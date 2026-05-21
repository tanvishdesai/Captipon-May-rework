import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Added for log_softmax in beam search
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score # Corrected import
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import random
import time
import csv
from pathlib import Path
import logging
from collections import defaultdict
from typing import Dict, List

import sentencepiece as spm # Import SentencePiece
import tempfile # For temporary file for SentencePiece training

# Step 0: Setup & Configuration
# -----------------------------------------------------------------------------
print("Step 0: Initializing Setup and Configuration...")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Hyperparameters ---
CONFIG = {
    'image_dir': '/kaggle/input/flickr8k/Images',
    'captions_file': '/kaggle/input/guj-captions/gujarati_captions.txt',
    'batch_size': 32,
    'embed_dim': 256,
    'hidden_dim': 512,
    'num_epochs': 20, 
    'learning_rate': 4e-4,
    'train_val_split': 0.8,
    'max_caption_length': 75, # Max length for PADDED captions (subword tokens)
    'model_save_path': "./best_lightweight_sp_beam_gujarati_captioner.pth",
    'csv_output_file': "./lightweight_sp_beam_predictions_all_images.csv", # Updated name to reflect content
    'plot_filename': "./lightweight_sp_beam_training_metrics.png",
    'sp_model_prefix': "lightweight_gujarati_sp", # Prefix for SentencePiece model files
    'sp_vocab_size': 8000, # Desired vocabulary size for SentencePiece
    'beam_width': 3, # Beam width for generation
}

# For reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Step 1: SentencePiece Vocabulary Class (Adapted from your Transformer script)
# -----------------------------------------------------------------------------
print("\nStep 1: Defining SentencePiece Vocabulary Class...")

class SentencePieceVocabulary:
    def __init__(self, model_prefix):
        self.model_prefix = model_prefix
        self.sp_model = spm.SentencePieceProcessor()
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        # Initialize IDs to None, they will be set after loading/building
        self.pad_idx = None
        self.sos_idx = None
        self.eos_idx = None
        self.unk_idx = None


    def build_vocabulary(self, sentence_list, vocab_size):
        print("Building SentencePiece vocabulary...")
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding='utf-8') as tmp_file:
            for sentence in sentence_list:
                tmp_file.write(sentence + "\n")
            temp_train_file_path = tmp_file.name

        model_filename_base = os.path.basename(self.model_prefix)
        try:
            spm.SentencePieceTrainer.train(
                input=temp_train_file_path,
                model_prefix=model_filename_base,
                vocab_size=vocab_size,
                user_defined_symbols=[], # Let SP handle special tokens via parameters below
                pad_id=0, pad_piece=self.pad_token,
                unk_id=1, unk_piece=self.unk_token,
                bos_id=2, bos_piece=self.sos_token, # BOS = Start Of Sentence
                eos_id=3, eos_piece=self.eos_token, # EOS = End Of Sentence
                model_type='bpe',
                character_coverage=1.0,
            )
        finally:
            if os.path.exists(temp_train_file_path):
                 os.remove(temp_train_file_path)
        self.load_model(f"{model_filename_base}.model") # Load after training

    def load_model(self, model_path):
        print(f"Loading SentencePiece model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model file not found: {model_path}")
        self.sp_model.load(model_path)
        self.pad_idx = self.sp_model.piece_to_id(self.pad_token)
        self.sos_idx = self.sp_model.piece_to_id(self.sos_token)
        self.eos_idx = self.sp_model.piece_to_id(self.eos_token)
        self.unk_idx = self.sp_model.piece_to_id(self.unk_token)

        print(f"Special token IDs: PAD={self.pad_idx}, SOS={self.sos_idx}, EOS={self.eos_idx}, UNK={self.unk_idx}")
        # Assertions to ensure IDs are standard
        assert self.pad_idx == 0, f"PAD ID is not 0, it's {self.pad_idx}"
        assert self.unk_idx == 1, f"UNK ID is not 1, it's {self.unk_idx}"
        assert self.sos_idx == 2, f"SOS ID is not 2, it's {self.sos_idx}"
        assert self.eos_idx == 3, f"EOS ID is not 3, it's {self.eos_idx}"
        print(f"SentencePiece vocabulary loaded/built. Size: {len(self)}")


    def __len__(self):
        return self.sp_model.get_piece_size() if self.sp_model and self.sp_model.get_piece_size() > 0 else 0

    def numericalize(self, text):
        if self.sos_idx is None: # Ensure model is loaded
            raise ValueError("SentencePiece model not loaded. Call load_model() or build_vocabulary() first.")
        tokens = self.sp_model.encode_as_ids(text)
        return [self.sos_idx] + tokens + [self.eos_idx]

    def textualize(self, indices):
        if self.sos_idx is None:
            raise ValueError("SentencePiece model not loaded.")
        # Filter out special tokens before decoding
        filtered_indices = [idx for idx in indices if idx not in [self.sos_idx, self.eos_idx, self.pad_idx]]
        return self.sp_model.decode_ids(filtered_indices)

    @staticmethod
    def tokenize_for_bleu(text, sp_model_instance):
        if not text or not sp_model_instance: return []
        return sp_model_instance.encode_as_pieces(text)


# Step 2: Model Architecture (LightweightCaptioningModel)
# -----------------------------------------------------------------------------
print("\nStep 2: Defining Model Architecture (LightweightCaptioningModel)...")
class LightweightCaptioningModel(nn.Module):
    def __init__(self, vocab, embed_dim=256, hidden_dim=512): # Takes vocab object
        super().__init__()
        self.vocab = vocab # Store vocab for special tokens
        self.vocab_size = len(vocab)

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_projection = nn.Linear(resnet.fc.in_features, hidden_dim) # Use resnet.fc.in_features
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=self.vocab.pad_idx)
        self.embed_projection = nn.Linear(embed_dim, hidden_dim)
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output = nn.Linear(hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, images, captions):
        batch_size = images.size(0)
        image_features = self.image_encoder(images)
        image_features = image_features.view(batch_size, -1)
        image_features = self.dropout(self.image_projection(image_features))

        embedded = self.dropout(self.embedding(captions))
        embedded = self.embed_projection(embedded)

        hidden = image_features.unsqueeze(0) # (1, batch_size, hidden_dim)

        output, _ = self.decoder(embedded, hidden)
        output = self.output(output)
        return output

    def generate_caption_beam_search(self, image_features_proj, beam_width=3, max_len=50):
        self.eval()
        batch_size = image_features_proj.size(0) # Should be 1 for single image generation
        assert batch_size == 1, "Beam search implemented for batch_size=1"

        hidden = image_features_proj.unsqueeze(0) # (1, 1, hidden_dim)

        k_prev_words = torch.full((beam_width, 1), self.vocab.sos_idx, dtype=torch.long, device=DEVICE)
        seqs = k_prev_words
        top_k_scores = torch.zeros(beam_width, 1, device=DEVICE)

        hidden_k = hidden.expand(1, beam_width, self.decoder.hidden_size).contiguous()

        complete_seqs = []
        complete_seqs_scores = []

        for step in range(max_len):
            inputs_current_step = seqs[:, -1].unsqueeze(1)
            embedded = self.embedding(inputs_current_step)
            embedded = self.embed_projection(embedded)

            output, hidden_k = self.decoder(embedded, hidden_k)
            logits = self.output(output.squeeze(1))
            log_probs = F.log_softmax(logits, dim=1)
            log_probs = top_k_scores.expand_as(log_probs) + log_probs

            if step == 0:
                top_k_scores, top_k_words = log_probs[0].topk(beam_width, 0, True, True)
                prev_beam_inds = torch.zeros(beam_width, dtype=torch.long, device=DEVICE)
            else:
                top_k_scores, top_k_words = log_probs.view(-1).topk(beam_width, 0, True, True)
                prev_beam_inds = torch.div(top_k_words, self.vocab_size, rounding_mode='floor')
            
            next_word_inds = top_k_words % self.vocab_size
            seqs = torch.cat([seqs[prev_beam_inds], next_word_inds.unsqueeze(1)], dim=1)
            hidden_k = hidden_k[:, prev_beam_inds, :]

            incomplete_inds = []
            for i in range(len(next_word_inds)): # Iterate up to current number of active beams
                if next_word_inds[i] == self.vocab.eos_idx:
                    complete_seqs.append(seqs[i, :].tolist())
                    complete_seqs_scores.append(top_k_scores[i].item())
                else:
                    incomplete_inds.append(i)
            
            if not incomplete_inds: break # All beams ended

            seqs = seqs[incomplete_inds]
            hidden_k = hidden_k[:, incomplete_inds, :]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            
            # Update beam_width for next iteration if some beams completed
            # This part requires careful handling if we want to maintain a fixed beam_width by exploring more.
            # Current logic reduces number of active beams. If all beams that were kept are incomplete,
            # then len(incomplete_inds) will be the new number of beams for the next step.
            # The topk operation will select 'beam_width' candidates (original config beam_width).
            # If number of incomplete_inds is less than original beam_width, topk will take all of them.
            # If more, it will take top 'beam_width'.
            # The important thing is `log_probs.view(-1).topk(beam_width, ...)` where beam_width is the original.

        if not complete_seqs:
            if seqs.nelement() > 0 :
                # Fallback: take the best currently incomplete sequence
                best_idx = top_k_scores.squeeze().argmax() if top_k_scores.numel() > 1 else 0
                best_seq = seqs[best_idx, :].tolist()
                if best_seq[-1] != self.vocab.eos_idx: best_seq.append(self.vocab.eos_idx)
                complete_seqs.append(best_seq)
                complete_seqs_scores.append(top_k_scores[best_idx].item() if top_k_scores.nelement() > 0 else -float('inf'))
            else:
                return self.vocab.textualize([self.vocab.sos_idx, self.vocab.eos_idx]) # Minimal valid caption

        if not complete_seqs_scores: return self.vocab.textualize([self.vocab.sos_idx, self.vocab.eos_idx])

        normalized_scores = [score / (len(seq)**0.7) for score, seq in zip(complete_seqs_scores, complete_seqs) if len(seq) > 0] # Length penalty
        if not normalized_scores: # All sequences might be empty if error
            return self.vocab.textualize([self.vocab.sos_idx, self.vocab.eos_idx])

        best_seq_idx = normalized_scores.index(max(normalized_scores))
        best_seq_indices = complete_seqs[best_seq_idx]
        
        return self.vocab.textualize(best_seq_indices)


# Step 3: Data Loading and Preprocessing
# -----------------------------------------------------------------------------
print("\nStep 3: Defining Data Loading and Preprocessing Utilities...")
class SimpleDataset(Dataset):
    def __init__(self, image_dir, vocab: SentencePieceVocabulary, max_length=50, transform=None):
        self.image_dir = Path(image_dir)
        self.vocab = vocab
        self.max_length = max_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.samples = [] # Will be populated by main logic after split
        print(f"Initializing SimpleDataset. Samples will be set externally.")
        if not self.vocab or len(self.vocab) == 0:
            raise ValueError("Vocabulary not provided or is empty for SimpleDataset.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            logging.error(f"Index {idx} out of bounds for SimpleDataset with {len(self.samples)} samples.")
            dummy_image = Image.new('RGB', (224, 224), color='magenta')
            caption_vec = torch.full((self.max_length,), self.vocab.pad_idx, dtype=torch.long)
            caption_vec[0] = self.vocab.sos_idx
            caption_vec[1] = self.vocab.eos_idx
            return {
                'image': self.transform(dummy_image),
                'caption': caption_vec,
                'raw_caption': "error dummy caption",
                'image_name': "error_image"
            }

        image_name, caption_text = self.samples[idx]
        image_path = self.image_dir / image_name
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            logging.warning(f"Image file not found at {image_path}. Using a dummy red image.")
            image = Image.new('RGB', (224, 224), color='red')
            caption_vec = torch.full((self.max_length,), self.vocab.pad_idx, dtype=torch.long)
            caption_vec[0] = self.vocab.sos_idx; caption_vec[1] = self.vocab.eos_idx
            return {
                'image': self.transform(image), 'caption': caption_vec,
                'raw_caption': "dummy caption for missing image", 'image_name': image_name
            }

        image = self.transform(image)
        numericalized_caption = self.vocab.numericalize(caption_text)

        padded_caption = torch.full((self.max_length,), self.vocab.pad_idx, dtype=torch.long)
        cap_len = len(numericalized_caption)

        if cap_len > self.max_length:
            padded_caption[:] = torch.tensor(numericalized_caption[:self.max_length-1] + [self.vocab.eos_idx], dtype=torch.long)
        else:
            padded_caption[:cap_len] = torch.tensor(numericalized_caption, dtype=torch.long)

        return {
            'image': image, 'caption': padded_caption,
            'raw_caption': caption_text, 'image_name': image_name
        }

class UniqueImageDataset(Dataset): # For validation
    def __init__(self, image_dir, image_to_captions: Dict[str, List[str]], vocab: SentencePieceVocabulary, transform, max_length):
        self.image_dir = Path(image_dir)
        self.image_to_captions = image_to_captions
        self.image_names = list(image_to_captions.keys())
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = self.image_dir / image_name
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except FileNotFoundError:
            logging.warning(f"Image file not found at {image_path} for UniqueImageDataset. Using dummy red image.")
            dummy_img_pil = Image.new('RGB', (224, 224), color='red')
            image = self.transform(dummy_img_pil)

        raw_captions_list = self.image_to_captions[image_name]
        return {
            'image': image, 'captions_raw_list': raw_captions_list, 'image_name': image_name
        }

def custom_collate_fn_unique_images(batch):
    images = torch.stack([item['image'] for item in batch])
    raw_captions_lists = [item['captions_raw_list'] for item in batch]
    image_names = [item['image_name'] for item in batch]
    return {'image': images, 'captions_raw_lists': raw_captions_lists, 'image_name': image_names}


# Step 4: Plotting Utilities
# -----------------------------------------------------------------------------
print("\nStep 4: Defining Plotting Utilities...")
def plot_metrics(train_losses, val_metrics_history, epoch_durations, val_epochs_ran, plot_filename):
    epochs_range_train = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_range_train, train_losses, 'bo-', label='Training Loss')
    plt.title('Training Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 2)
    if val_metrics_history and val_epochs_ran and val_metrics_history[0]:
        for i in range(1, 5):
            bleu_key = f'bleu-{i}'
            if bleu_key in val_metrics_history[0]:
                bleu_i_scores = [scores.get(bleu_key, 0.0) for scores in val_metrics_history] # Use .get for safety
                plt.plot(val_epochs_ran, bleu_i_scores, marker='o', linestyle='-', label=f'BLEU-{i}')
    plt.title('Validation BLEU Scores'); plt.xlabel('Epochs'); plt.ylabel('BLEU Score (0-1)')
    plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 3)
    if val_metrics_history and val_epochs_ran and val_metrics_history[0]:
        if 'meteor' in val_metrics_history[0]:
            meteor_scores = [scores.get('meteor', 0.0) for scores in val_metrics_history]
            plt.plot(val_epochs_ran, meteor_scores, marker='s', linestyle='-', label='METEOR')
        if 'rougeL' in val_metrics_history[0]:
            rougeL_scores = [scores.get('rougeL', 0.0) for scores in val_metrics_history]
            plt.plot(val_epochs_ran, rougeL_scores, marker='^', linestyle='-', label='ROUGE-L')
    plt.title('Validation METEOR & ROUGE-L Scores'); plt.xlabel('Epochs'); plt.ylabel('Score (%)')
    plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 4)
    if epoch_durations:
        epochs_range_epoch_dur = range(1, len(epoch_durations) + 1)
        plt.plot(epochs_range_epoch_dur, [d/60 for d in epoch_durations], 'go-', label='Epoch Duration (mins)')
    plt.title('Epoch Duration'); plt.xlabel('Epochs'); plt.ylabel('Duration (minutes)')
    plt.legend(); plt.grid(True)

    plt.tight_layout(); plt.savefig(plot_filename)
    print(f"Metrics plot saved as {plot_filename}")


# Step 5: CSV Generation Utility
# -----------------------------------------------------------------------------
print("\nStep 5: Defining CSV Generation Utility...")
def generate_predictions_csv(model, image_names_list, image_to_captions_map, vocab,
                             image_transform, max_pred_len, beam_width, img_dir, device, output_csv_file):
    print(f"\nGenerating predictions CSV for {len(image_names_list)} images to {output_csv_file}...")
    model.eval()
    results = []

    with torch.no_grad():
        for img_name in tqdm(image_names_list, desc="Generating predictions for CSV"):
            img_path = Path(img_dir) / img_name
            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = image_transform(image).unsqueeze(0).to(device)

                img_features = model.image_encoder(image_tensor)
                img_features = img_features.view(1, -1)
                img_features_proj = model.image_projection(img_features)

                generated_caption_text = model.generate_caption_beam_search(
                    img_features_proj, beam_width=beam_width, max_len=max_pred_len
                )

                original_captions_list = image_to_captions_map.get(img_name, [])
                # Join all original captions with a separator for the CSV for better comparison
                original_captions_str = " | ".join(original_captions_list) if original_captions_list else "N/A"


                results.append({
                    "image_id": img_name,
                    "original_captions": original_captions_str, # Changed field name
                    "generated_caption": generated_caption_text
                })
            except FileNotFoundError:
                logging.warning(f"Image file not found at {img_path}. Skipping for CSV.")
            except Exception as e:
                logging.error(f"Error processing image {img_name} for CSV: {e}. Skipping.")

    if results:
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["image_id", "original_captions", "generated_caption"] # Changed field name
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Predictions CSV saved to {output_csv_file}")
    else:
        print("No results to save to CSV.")


# Step 6: Training and Evaluation Utilities
# -----------------------------------------------------------------------------
print("\nStep 6: Defining Training and Evaluation Utilities...")

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs, vocab):
    model.train()
    total_loss = 0
    epoch_start_time = time.time()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", unit="batch")

    for batch_idx, batch_data in enumerate(progress_bar):
        images = batch_data['image'].to(device)
        captions = batch_data['caption'].to(device)

        outputs = model(images, captions[:, :-1])
        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            captions[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(loss=loss.item(), avg_loss_batch=total_loss/(batch_idx+1), lr=f"{current_lr:.1e}")

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    avg_epoch_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    print(f"Epoch {epoch+1} - Training Duration: {epoch_duration:.2f}s ({epoch_duration/60:.2f}m), Avg Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss, epoch_duration

def evaluate_model(model, val_loader, vocab: SentencePieceVocabulary, device, max_gen_len=50, beam_width=3):
    model.eval()
    all_references_tokenized = []
    all_hypotheses_tokenized = []
    all_hypotheses_str_for_rouge_meteor = [] # For full string hypotheses
    all_references_str_for_rouge_meteor = [] # For full string references


    # Ensure NLTK resources are available
    try: nltk.data.find('tokenizers/punkt.zip')
    except LookupError: nltk.download('punkt', quiet=True); print("NLTK punkt downloaded.")
    try: nltk.data.find('corpora/wordnet.zip')
    except LookupError: nltk.download('wordnet', quiet=True); print("NLTK wordnet downloaded.")
    try: nltk.data.find('corpora/omw-1.4.zip')
    except LookupError: nltk.download('omw-1.4', quiet=True); print("NLTK omw-1.4 downloaded.")


    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Evaluating"):
            images = batch_data['image'].to(device)
            references_batch_raw_text = batch_data['captions_raw_lists']

            batch_img_features = model.image_encoder(images)
            batch_img_features = batch_img_features.view(images.size(0), -1)
            batch_img_features_proj = model.image_projection(batch_img_features)

            for i in range(images.size(0)):
                single_img_feature_proj = batch_img_features_proj[i].unsqueeze(0)
                hyp_str = model.generate_caption_beam_search(
                    single_img_feature_proj, beam_width=beam_width, max_len=max_gen_len
                )
                
                # For BLEU (SentencePiece subwords)
                all_hypotheses_tokenized.append(vocab.tokenize_for_bleu(hyp_str, vocab.sp_model))
                refs_for_image_raw = references_batch_raw_text[i]
                refs_for_image_tokenized_bleu = [vocab.tokenize_for_bleu(ref, vocab.sp_model) for ref in refs_for_image_raw]
                all_references_tokenized.append(refs_for_image_tokenized_bleu)

                # For METEOR/ROUGE (space-tokenized decoded strings)
                all_hypotheses_str_for_rouge_meteor.append(hyp_str) # hyp_str is already decoded
                all_references_str_for_rouge_meteor.append(refs_for_image_raw)


    bleu_scores_dict = {}
    if not all_hypotheses_tokenized or not all_references_tokenized:
        print("Warning: No hypotheses or references for BLEU calculation.")
        for i in range(1, 5): bleu_scores_dict[f'bleu-{i}'] = 0.0
    else:
        for i in range(1, 5):
            weights = tuple(1.0/i for _ in range(i))
            try:
                bleu_scores_dict[f'bleu-{i}'] = corpus_bleu(all_references_tokenized, all_hypotheses_tokenized, weights=weights)
            except ZeroDivisionError: bleu_scores_dict[f'bleu-{i}'] = 0.0
            except Exception as e:
                logging.error(f"Error BLEU-{i}: {e}"); bleu_scores_dict[f'bleu-{i}'] = 0.0

    # METEOR and ROUGE calculations
    meteor_scores_list = []
    rouge1_scores_list, rouge2_scores_list, rougeL_scores_list = [], [], []
    
    if all_references_str_for_rouge_meteor and all_hypotheses_str_for_rouge_meteor:
        rouge_calculator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        for refs_raw_single_img, hyp_raw_single_img in zip(all_references_str_for_rouge_meteor, all_hypotheses_str_for_rouge_meteor):
            # Tokenize for METEOR (NLTK expects list of words)
            hyp_tokens_meteor = nltk.word_tokenize(hyp_raw_single_img)
            refs_tokens_meteor = [nltk.word_tokenize(ref) for ref in refs_raw_single_img]
            
            try:
                meteor_scores_list.append(meteor_score(refs_tokens_meteor, hyp_tokens_meteor))
            except Exception as e: logging.error(f"METEOR error: {e}"); meteor_scores_list.append(0.0)

            # ROUGE (uses raw strings)
            current_r1, current_r2, current_rL = [], [], []
            if refs_raw_single_img:
                for ref_str in refs_raw_single_img:
                    try:
                        scores = rouge_calculator.score(ref_str, hyp_raw_single_img)
                        current_r1.append(scores['rouge1'].fmeasure)
                        current_r2.append(scores['rouge2'].fmeasure)
                        current_rL.append(scores['rougeL'].fmeasure)
                    except Exception as e: logging.error(f"ROUGE error calculating for one ref: {e}");
                rouge1_scores_list.append(np.mean(current_r1) if current_r1 else 0.0)
                rouge2_scores_list.append(np.mean(current_r2) if current_r2 else 0.0)
                rougeL_scores_list.append(np.mean(current_rL) if current_rL else 0.0)
            else: # No references for this image
                rouge1_scores_list.append(0.0); rouge2_scores_list.append(0.0); rougeL_scores_list.append(0.0)

    metrics = {
        **bleu_scores_dict,
        'meteor': np.mean(meteor_scores_list) * 100 if meteor_scores_list else 0.0,
        'rouge1': np.mean(rouge1_scores_list) * 100 if rouge1_scores_list else 0.0,
        'rouge2': np.mean(rouge2_scores_list) * 100 if rouge2_scores_list else 0.0,
        'rougeL': np.mean(rougeL_scores_list) * 100 if rougeL_scores_list else 0.0,
    }

    if all_hypotheses_str_for_rouge_meteor and all_references_str_for_rouge_meteor:
        print("Sample predictions from evaluation:")
        for i in range(min(3, len(all_hypotheses_str_for_rouge_meteor))):
            print(f"  Sample {i+1}:")
            print(f"    References (raw): {all_references_str_for_rouge_meteor[i][:2]}") # Print first 2 raw refs
            print(f"    Hypothesis (raw): {all_hypotheses_str_for_rouge_meteor[i]}")
    return metrics, all_hypotheses_str_for_rouge_meteor


# Step 7: Main Execution Block
# -----------------------------------------------------------------------------
def main():
    print("\nStep 7: Starting Main Execution...")
    global CONFIG, DEVICE

    # --- 7.1 Load Data and Build/Load Vocabulary ---
    print("\n--- 7.1 Loading Data and Building/Loading SP Vocabulary ---")
    # Ensure sp_model_prefix is just a filename base for local storage
    sp_model_filename_base = os.path.basename(CONFIG['sp_model_prefix'])
    sp_model_file = f"{sp_model_filename_base}.model"
    sp_vocab_file = f"{sp_model_filename_base}.vocab" # SentencePiece also creates a .vocab file

    vocab = SentencePieceVocabulary(sp_model_filename_base) # Pass only base prefix

    all_captions_for_vocab_build = []
    if not os.path.exists(CONFIG['captions_file']):
        print(f"Error: Captions file {CONFIG['captions_file']} not found. Aborting."); return

    with open(CONFIG['captions_file'], 'r', encoding='utf-8-sig') as f:
        for line in f:
            try:
                _, caption = line.strip().split('\t', 1)
                all_captions_for_vocab_build.append(caption)
            except ValueError: continue
    if not all_captions_for_vocab_build:
        print("Error: No captions loaded from file for vocab building. Aborting."); return

    if os.path.exists(sp_model_file) and os.path.exists(sp_vocab_file) :
        vocab.load_model(sp_model_file)
    else:
        print(f"SP model file {sp_model_file} or {sp_vocab_file} not found. Building vocabulary...")
        vocab.build_vocabulary(all_captions_for_vocab_build, vocab_size=CONFIG['sp_vocab_size'])

    if len(vocab) == 0: print("Error: SP Vocabulary is empty after loading/building. Aborting."); return


    # --- 7.2 Prepare Datasets ---
    print("\n--- 7.2 Preparing Datasets ---")
    all_image_to_captions = defaultdict(list)
    valid_image_names = set()
    base_image_dir = Path(CONFIG['image_dir'])

    with open(CONFIG['captions_file'], 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Aggregating captions per image"):
            try:
                image_name_full, caption = line.strip().split('\t', 1)
                image_name = image_name_full.split('#')[0] # Get base image name
                if (base_image_dir / image_name).exists():
                    all_image_to_captions[image_name].append(caption)
                    valid_image_names.add(image_name)
                # else:
                #     logging.debug(f"Image file {image_name} not found in {base_image_dir}. Skipping.")
            except ValueError: continue
    
    unique_image_names_with_files = list(valid_image_names)
    if not unique_image_names_with_files: print("Error: No valid images found with captions. Aborting."); return
    
    # Shuffle for splitting train/val. The original order is lost here for unique_image_names_with_files
    # If a consistent order is needed for CSV later, make a sorted copy *before* shuffling.
    # For this request, shuffled is fine for the "all images" CSV.
    random.shuffle(unique_image_names_with_files)


    split_idx = int(CONFIG['train_val_split'] * len(unique_image_names_with_files))
    train_image_names = unique_image_names_with_files[:split_idx]
    val_image_names = unique_image_names_with_files[split_idx:]
    print(f"Total unique images with files: {len(unique_image_names_with_files)}")
    print(f"Train unique images: {len(train_image_names)}, Val unique images: {len(val_image_names)}")

    train_samples_list = []
    for img_name in train_image_names:
        for cap_text in all_image_to_captions[img_name]:
            train_samples_list.append((img_name, cap_text))
    
    # Define transforms
    common_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = SimpleDataset(CONFIG['image_dir'], vocab, CONFIG['max_caption_length'], transform=common_transform)
    train_dataset.samples = train_samples_list
    print(f"Train dataset (flattened image-caption pairs): {len(train_dataset)}")

    val_image_to_captions_map = {name: all_image_to_captions[name] for name in val_image_names}
    val_dataset = UniqueImageDataset(CONFIG['image_dir'], val_image_to_captions_map, vocab, common_transform, CONFIG['max_caption_length'])
    print(f"Validation dataset (unique images): {len(val_dataset)}")

    if len(train_dataset) == 0 and CONFIG['train_val_split'] > 0 : # Only error if split implies training data
         print("Warning: Training dataset is empty but was expected. Check split or data.");
         # If split is 0, this is fine.
         if CONFIG['train_val_split'] == 0 and len(val_dataset) > 0:
             print("Proceeding with validation/testing only as train_val_split is 0.")
         else:
             print("Error: Training dataset is empty. Aborting."); return


    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, collate_fn=custom_collate_fn_unique_images)
    
    print(f"Train loader: {len(train_loader)} batches. Val loader: {len(val_loader) if val_loader else 'N/A'} batches.")


    # --- 7.3 Initialize Model, Optimizer, Criterion ---
    print("\n--- 7.3 Initializing Model, Optimizer, Criterion ---")
    model = LightweightCaptioningModel(
        vocab=vocab,
        embed_dim=CONFIG['embed_dim'],
        hidden_dim=CONFIG['hidden_dim']
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    print(f"Model initialized. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 7.4 Training Loop ---
    print("\n--- 7.4 Starting Training Loop ---")
    train_losses_hist, val_metrics_hist, epoch_durations_hist, val_epochs_run = [], [], [], []
    best_bleu4 = -1.0 # Initialize to a value that will be beaten

    if len(train_loader) > 0: # Only train if there's training data
        for epoch in range(CONFIG['num_epochs']):
            print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
            train_loss, epoch_duration = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch, CONFIG['num_epochs'], vocab)
            train_losses_hist.append(train_loss)
            epoch_durations_hist.append(epoch_duration)

            if val_loader:
                val_metrics, _ = evaluate_model(model, val_loader, vocab, DEVICE,
                                            max_gen_len=CONFIG['max_caption_length'],
                                            beam_width=CONFIG['beam_width'])
                val_metrics_hist.append(val_metrics)
                val_epochs_run.append(epoch + 1)
                print(f"Epoch {epoch+1} Validation Metrics (BLEU 0-1, others 0-100):")
                for name, val in val_metrics.items(): print(f"  {name}: {val:.4f}")

                current_bleu4 = val_metrics.get('bleu-4', 0.0)
                if current_bleu4 > best_bleu4:
                    print(f"BLEU-4 improved from {best_bleu4:.4f} to {current_bleu4:.4f}. Saving model...")
                    best_bleu4 = current_bleu4
                    torch.save({
                        'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'vocab_config': {'prefix': sp_model_filename_base, 'model_file': sp_model_file},
                        'config': CONFIG, 'metrics': val_metrics
                    }, CONFIG['model_save_path'])
            else: # No validation loader, save periodically or at end
                if (epoch + 1) % 5 == 0 or epoch == CONFIG['num_epochs'] -1 :
                    save_path_no_val = f"{CONFIG['model_save_path'].replace('.pth', '')}_epoch{epoch+1}_no_val.pth"
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'vocab_config': {'prefix': sp_model_filename_base, 'model_file': sp_model_file},
                                'config': CONFIG }, save_path_no_val)
                    if not os.path.exists(CONFIG['model_save_path']) or best_bleu4 < 0: # If no val model saved yet
                        CONFIG['model_save_path'] = save_path_no_val # Update best model path
                        print(f"Saved model (no validation) to {save_path_no_val}")
    else:
        print("Skipping training loop as train_loader is empty.")
        # If not training, try to load a pre-existing model if specified or available
        if os.path.exists(CONFIG['model_save_path']):
            print(f"Training skipped. Using existing model: {CONFIG['model_save_path']}")
        else:
            # Try to find a _no_val model if primary doesn't exist
            potential_fallback = f"{CONFIG['model_save_path'].replace('.pth', '')}_epoch{CONFIG['num_epochs']}_no_val.pth"
            if os.path.exists(potential_fallback):
                 CONFIG['model_save_path'] = potential_fallback
                 print(f"Training skipped. Using existing model: {CONFIG['model_save_path']}")
            else:
                 print("Training skipped and no existing model found at specified paths. CSV/Example generation might fail.")


    # --- 7.5 Plot Metrics ---
    if train_losses_hist: # Only plot if training happened
        print("\n--- 7.5 Plotting Metrics ---")
        plot_metrics(train_losses_hist, val_metrics_hist, epoch_durations_hist, val_epochs_run, CONFIG['plot_filename'])

    # --- 7.6 Load Best Model and Generate Predictions CSV for ALL IMAGES ---
    print("\n--- 7.6 Generating Predictions CSV for All Processed Images ---")
    model_load_path = CONFIG['model_save_path']
    if not os.path.exists(model_load_path):
        potential_fallback = f"{CONFIG['model_save_path'].replace('.pth', '')}_epoch{CONFIG['num_epochs']}_no_val.pth"
        if os.path.exists(potential_fallback):
            model_load_path = potential_fallback
            print(f"Using fallback model: {model_load_path}")
        else:
            model_load_path = None
            print(f"No model found at {CONFIG['model_save_path']} or fallback for CSV generation.")

    if model_load_path and unique_image_names_with_files: # Check if we have a model and images
        checkpoint = torch.load(model_load_path, map_location=DEVICE)
        
        # Reload vocab based on checkpoint's config
        # Checkpoint stores base prefix, construct full sp_model_file path as done before
        chkpt_sp_model_filename_base = checkpoint['vocab_config']['prefix']
        chkpt_sp_model_file = checkpoint['vocab_config']['model_file'] # This should be the path used when saving

        vocab_csv = SentencePieceVocabulary(chkpt_sp_model_filename_base)
        try:
            vocab_csv.load_model(chkpt_sp_model_file)
        except FileNotFoundError:
            print(f"Error: SP model file {chkpt_sp_model_file} for CSV generation not found. Ensure it's in CWD or path is correct.")
            print("Skipping CSV generation.")
        else:
            config_csv = checkpoint.get('config', CONFIG) # Use checkpoint's config if available

            final_model = LightweightCaptioningModel(vocab_csv, config_csv['embed_dim'], config_csv['hidden_dim']).to(DEVICE)
            final_model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"Generating CSV for {len(unique_image_names_with_files)} unique images using model from {model_load_path}.")
            generate_predictions_csv(
                final_model,
                unique_image_names_with_files,    # Full list of unique image names
                all_image_to_captions,            # Captions for all these images
                vocab_csv,
                common_transform,                 # Use the same transform as validation/training
                config_csv.get('max_caption_length', CONFIG['max_caption_length']),
                config_csv.get('beam_width', CONFIG['beam_width']),
                config_csv.get('image_dir', CONFIG['image_dir']),
                DEVICE,
                CONFIG['csv_output_file']
            )
    else:
        if not model_load_path: print("No model loaded, skipping CSV generation.")
        if not unique_image_names_with_files: print("No images to process for CSV.")


    # --- 7.7 Example Usage ---
    print("\n--- 7.7 Example Usage (using validation set sample if available) ---")
    # This part still uses val_dataset. If val_dataset is empty, it will skip.
    if model_load_path and val_dataset and len(val_dataset) > 0:
        # Ensure final_model and vocab_csv are loaded if CSV part was skipped or failed
        if 'final_model' not in locals() or final_model is None:
            print("Loading model for example usage...")
            checkpoint_ex = torch.load(model_load_path, map_location=DEVICE)
            
            chkpt_ex_sp_model_filename_base = checkpoint_ex['vocab_config']['prefix']
            chkpt_ex_sp_model_file = checkpoint_ex['vocab_config']['model_file']

            vocab_ex = SentencePieceVocabulary(chkpt_ex_sp_model_filename_base)
            try:
                vocab_ex.load_model(chkpt_ex_sp_model_file)
            except FileNotFoundError:
                 print(f"Error: SP model file {chkpt_ex_sp_model_file} for example usage not found. Skipping example."); vocab_ex = None
            
            if vocab_ex:
                config_ex = checkpoint_ex.get('config', CONFIG)
                example_model = LightweightCaptioningModel(vocab_ex, config_ex['embed_dim'], config_ex['hidden_dim']).to(DEVICE)
                example_model.load_state_dict(checkpoint_ex['model_state_dict'])
        elif 'final_model' in locals() and final_model is not None and 'vocab_csv' in locals() and vocab_csv is not None :
             example_model = final_model # Use model from CSV generation if successful
             vocab_ex = vocab_csv
             config_ex = config_csv if 'config_csv' in locals() else CONFIG
        else:
            example_model = None; vocab_ex = None; print("Model or vocab not available for example.")

        if example_model and vocab_ex:
            example_model.eval()
            sample_idx = random.randint(0, len(val_dataset)-1)
            sample_data = val_dataset[sample_idx]
            sample_img_tensor = sample_data['image'].unsqueeze(0).to(DEVICE)
            
            print(f"Original Captions (sample image {sample_data['image_name']} from validation set):")
            for i, cap_txt in enumerate(sample_data['captions_raw_list'][:3]): print(f"  Ref {i+1}: {cap_txt}")

            with torch.no_grad():
                img_feat_ex = example_model.image_encoder(sample_img_tensor)
                img_feat_ex = img_feat_ex.view(1, -1)
                img_feat_proj_ex = example_model.image_projection(img_feat_ex)
                gen_caption_ex = example_model.generate_caption_beam_search(
                    img_feat_proj_ex,
                    beam_width=config_ex.get('beam_width', CONFIG['beam_width']),
                    max_len=config_ex.get('max_caption_length', CONFIG['max_caption_length'])
                )
            print(f"Generated Caption (sample with beam search): {gen_caption_ex}")
    else:
        print("Skipping example usage: No model loaded or validation dataset is empty.")

    print("\nScript finished.")

if __name__ == "__main__":
    # Create dummy files for local testing if they don't exist (Kaggle paths are different)
    

    main()