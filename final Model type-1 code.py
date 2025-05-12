import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Ensure transforms is imported
import torchvision.models as models
import collections
from PIL import Image
import nltk
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import random
import math
import sentencepiece as spm # Import SentencePiece
import tempfile # For temporary file for SentencePiece training
import time # For timing
import csv # For CSV output

# Step 0: Setup & Configuration
# -----------------------------------------------------------------------------
print("Step 0: Initializing Setup and Configuration...")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Hyperparameters ---
IMAGE_DIR = "/kaggle/input/flickr8k/Images"
CAPTION_FILE = "/kaggle/input/guj-captions/gujarati_captions.txt"
MODEL_SAVE_PATH = "./best_transformer_caption_model_augmented_v2.pth" # Changed model name
SP_MODEL_PREFIX = "gujarati_caption_sp" # Prefix for SentencePiece model files
SP_VOCAB_SIZE = 8000 # Desired vocabulary size for SentencePiece
CSV_OUTPUT_FILE = "all_images_predictions_augmented_v2.csv" # CSV output filename

# Model Hyperparameters
d_model = 512            # Dimension of embeddings and Transformer model (formerly embed_size & decoder_dim)
encoder_dim = 2048      # Dimension of encoder output (ResNet50)
nhead = 8               # Number of heads in MultiHeadAttention
num_transformer_decoder_layers = 6 # Number of Transformer decoder layers
dim_feedforward = 2048  # Dimension of feedforward network in Transformer
transformer_dropout = 0.15 # Dropout rate in Transformer

# Training params
num_epochs = 20 # Keep or adjust as needed
batch_size = 32 # Can be reduced if Transformer uses more memory
learning_rate = 1e-4 # May need tuning for Transformer
grad_clip_norm = 5.0
max_caption_length = 65 # Max length for PADDED captions (subword tokens might be more)
patience_early_stop = 5
lr_patience = 2
eval_frequency = 5 # How often to run validation (every N epochs)

# For reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Step 1: SentencePiece Vocabulary Class
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

    def build_vocabulary(self, sentence_list, vocab_size):
        print("Building SentencePiece vocabulary...")
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding='utf-8') as tmp_file:
            for sentence in sentence_list:
                tmp_file.write(sentence + "\n")
            temp_train_file_path = tmp_file.name
        
        model_filename = os.path.basename(self.model_prefix)
        spm.SentencePieceTrainer.train(
            input=temp_train_file_path,
            model_prefix=model_filename,
            vocab_size=vocab_size,
            user_defined_symbols=[],
            pad_id=0, pad_piece=self.pad_token,
            unk_id=1, unk_piece=self.unk_token,
            bos_id=2, bos_piece=self.sos_token,
            eos_id=3, eos_piece=self.eos_token,
            model_type='bpe', 
            character_coverage=1.0,
        )
        os.remove(temp_train_file_path) 
        
        self.sp_model.load(f"{model_filename}.model")
        
        self.pad_idx = self.sp_model.piece_to_id(self.pad_token)
        self.sos_idx = self.sp_model.piece_to_id(self.sos_token)
        self.eos_idx = self.sp_model.piece_to_id(self.eos_token)
        self.unk_idx = self.sp_model.piece_to_id(self.unk_token)

        print(f"Special token IDs: PAD={self.pad_idx}, SOS={self.sos_idx}, EOS={self.eos_idx}, UNK={self.unk_idx}")
        assert self.pad_idx == 0, f"PAD ID is not 0, it's {self.pad_idx}"
        assert self.unk_idx == 1, f"UNK ID is not 1, it's {self.unk_idx}"
        assert self.sos_idx == 2, f"SOS ID is not 2, it's {self.sos_idx}"
        assert self.eos_idx == 3, f"EOS ID is not 3, it's {self.eos_idx}"

        print(f"SentencePiece vocabulary built. Size: {len(self)}")

    def __len__(self):
        return self.sp_model.get_piece_size() if self.sp_model else 0

    def numericalize(self, text):
        tokens = self.sp_model.encode_as_ids(text)
        return [self.sos_idx] + tokens + [self.eos_idx]

    def textualize(self, indices):
        filtered_indices = [idx for idx in indices if idx not in [self.sos_idx, self.eos_idx, self.pad_idx]]
        return self.sp_model.decode_ids(filtered_indices)
    
    @staticmethod
    def tokenize_for_bleu(text, sp_model):
        if not text: return []
        return sp_model.encode_as_pieces(text)


# Step 2: Data Loading and Preprocessing
# -----------------------------------------------------------------------------
print("\nStep 2: Defining Data Loading and Preprocessing Utilities...")

def load_captions(filepath):
    print(f"Loading captions from: {filepath}")
    captions_dict = collections.defaultdict(list)
    all_captions_for_vocab = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    parts = line.split('\t', 1)
                    if len(parts) != 2:
                        print(f"Warning: Skipping malformed line {line_num+1}: '{line}'")
                        continue
                    img_id_part, caption_text = parts
                    img_name = img_id_part.split('#')[0] 
                    captions_dict[img_name].append(caption_text)
                    all_captions_for_vocab.append(caption_text)
                except Exception as e:
                    print(f"Warning: Error parsing line {line_num+1}: '{line}'. Error: {e}")
                    continue
        print(f"Loaded captions for {len(captions_dict)} unique images.")
        if not captions_dict:
            raise ValueError("No captions loaded. Check caption file format and path.")
        return captions_dict, all_captions_for_vocab
    except FileNotFoundError:
        print(f"Error: Caption file not found at {filepath}"); raise
    except Exception as e:
        print(f"An error occurred while loading captions: {e}"); raise

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_dict, image_names, vocab, transform, max_len):
        self.root_dir = root_dir
        self.df = [] 
        for img_name in image_names:
            if img_name in captions_dict:
                for caption in captions_dict[img_name]:
                    self.df.append((img_name, caption))
            # else: print(f"Warning: Image {img_name} found in split list but not in captions_dict.")
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len
        print(f"Dataset initialized with {len(self.df)} image-caption pairs.")
        if not self.df: raise ValueError("Dataset is empty.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name, caption_text = self.df[idx]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}. Using dummy image.")
            image = Image.new('RGB', (256, 256), color = 'red') # Dummy image
            # Create a minimal valid caption if image is missing
            caption_vec = torch.full((self.max_len,), self.vocab.pad_idx, dtype=torch.long)
            caption_vec[0] = self.vocab.sos_idx
            caption_vec[1] = self.vocab.eos_idx # Minimal valid caption: <sos> <eos> <pad>...
            return self.transform(image), caption_vec, torch.tensor(2, dtype=torch.long)


        image = self.transform(image)
        numericalized_caption = self.vocab.numericalize(caption_text)
        caption_len = len(numericalized_caption)
        
        padded_caption = torch.full((self.max_len,), self.vocab.pad_idx, dtype=torch.long)
        if caption_len > self.max_len:
            padded_caption[:] = torch.tensor(numericalized_caption[:self.max_len], dtype=torch.long)
            if padded_caption[-1] != self.vocab.eos_idx:
                 padded_caption[-1] = self.vocab.eos_idx 
            caption_len = self.max_len
        else:
            padded_caption[:caption_len] = torch.tensor(numericalized_caption, dtype=torch.long)
        return image, padded_caption, torch.tensor(caption_len, dtype=torch.long)


# Step 3: Model Architecture
# -----------------------------------------------------------------------------
print("\nStep 3: Defining Model Architecture...")
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
    def forward(self, images):
        features = self.resnet(images)
        batch_size = features.size(0)
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch_size, -1, features.size(-1))
        return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward,
                 encoder_feature_dim, dropout=0.1, max_seq_length=100):
        super(TransformerDecoderModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_length)
        self.encoder_projection = nn.Linear(encoder_feature_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(DEVICE)

    def _create_padding_mask(self, sequence, pad_idx):
        return (sequence == pad_idx).to(DEVICE)

    def forward(self, encoder_out, tgt_captions, pad_idx):
        memory = self.encoder_projection(encoder_out)
        tgt_emb = self.embedding(tgt_captions) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_seq_len = tgt_captions.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len)
        tgt_padding_mask = self._create_padding_mask(tgt_captions, pad_idx)
        output = self.transformer_decoder(tgt=tgt_emb, memory=memory,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_padding_mask)
        logits = self.fc_out(output)
        return logits, None

    def sample_beam_search(self, encoder_out, vocab, beam_width=5, max_sample_length=50):
        self.eval()
        batch_size = encoder_out.size(0)
        assert batch_size == 1, "Beam search currently supports batch_size=1"

        memory = self.encoder_projection(encoder_out)
        memory_k = memory.expand(beam_width, -1, -1)

        k_prev_words = torch.full((beam_width, 1), vocab.sos_idx, dtype=torch.long).to(DEVICE)
        seqs = k_prev_words
        top_k_scores = torch.zeros(beam_width, 1).to(DEVICE)
        complete_seqs, complete_seqs_scores = [], []

        for step in range(max_sample_length):
            tgt_emb = self.embedding(seqs) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            current_seq_len = seqs.size(1)
            tgt_mask = self._generate_square_subsequent_mask(current_seq_len).to(DEVICE)
            decoder_output = self.transformer_decoder(tgt_emb, memory_k, tgt_mask=tgt_mask)
            logits = self.fc_out(decoder_output[:, -1, :])
            log_probs = F.log_softmax(logits, dim=1)
            log_probs = top_k_scores.expand_as(log_probs) + log_probs

            if step == 0:
                top_k_scores, top_k_words = log_probs[0].topk(beam_width, 0, True, True)
                prev_beam_inds = torch.arange(beam_width).to(DEVICE)
            else:
                top_k_scores, top_k_words = log_probs.view(-1).topk(beam_width, 0, True, True)
                prev_beam_inds = torch.div(top_k_words, self.vocab_size, rounding_mode='floor')
            
            next_word_inds = top_k_words % self.vocab_size
            seqs = torch.cat([seqs[prev_beam_inds], next_word_inds.unsqueeze(1)], dim=1)
            
            is_eos = (next_word_inds == vocab.eos_idx)
            incomplete_inds = []
            for i in range(is_eos.size(0)):
                if is_eos[i]:
                    complete_seqs.append(seqs[i, :].tolist())
                    complete_seqs_scores.append(top_k_scores[i].item())
                else:
                    incomplete_inds.append(i)
            
            beam_width_new = len(incomplete_inds)
            if beam_width_new == 0: break

            seqs = seqs[incomplete_inds]
            memory_k = memory_k[prev_beam_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            
            if beam_width_new != memory_k.size(0): memory_k = memory_k[:beam_width_new]

        if not complete_seqs:
            if seqs.nelement() > 0:
                 best_idx = top_k_scores.squeeze().argmax()
                 complete_seqs.append(seqs[best_idx].tolist())
                 complete_seqs_scores.append(top_k_scores[best_idx].item())
            else: return []

        if not complete_seqs_scores: return []
        
        best_seq_idx = complete_seqs_scores.index(max(complete_seqs_scores))
        best_seq = complete_seqs[best_seq_idx]
        sampled_ids = [idx for idx in best_seq if idx not in [vocab.sos_idx, vocab.eos_idx, vocab.pad_idx]]
        return sampled_ids

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, images, captions, pad_idx):
        encoder_out = self.encoder(images)
        outputs, _ = self.decoder(encoder_out, captions, pad_idx)
        return outputs, None
    def generate_caption_beam_search(self, image, vocab, beam_width=5, max_sample_length=50):
        self.eval()
        with torch.no_grad():
            if image.dim() == 3: image = image.unsqueeze(0)
            image = image.to(DEVICE)
            encoder_out = self.encoder(image)
            sampled_ids = self.decoder.sample_beam_search(encoder_out, vocab, beam_width, max_sample_length)
            return vocab.textualize(sampled_ids)

# Step 4: Training and Evaluation Utilities
# -----------------------------------------------------------------------------
print("\nStep 4: Defining Training and Evaluation Utilities...")
def train_one_epoch(model, train_loader, optimizer, criterion, vocab, grad_clip_norm, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    epoch_start_time = time.time()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", unit="batch")
    for i, (images, captions, _) in enumerate(progress_bar):
        images, captions = images.to(DEVICE), captions.to(DEVICE)
        decoder_input_captions, targets = captions[:, :-1], captions[:, 1:]
        optimizer.zero_grad()
        outputs, _ = model(images, decoder_input_captions, vocab.pad_idx)
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(loss=loss.item(), avg_loss_batch=total_loss/(i+1), lr=f"{current_lr:.1e}")
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch {epoch+1} - Training Duration: {epoch_duration:.2f} seconds ({epoch_duration/60:.2f} minutes)")
    return total_loss / len(train_loader), epoch_duration

def evaluate_model(model, val_loader, criterion, vocab, epoch, num_epochs):
    model.eval()
    total_loss = 0.0
    references_corpus, hypotheses_corpus = [], []
    print(f"Epoch {epoch+1}/{num_epochs} [Validation] - Generating captions and calculating BLEU...")
    with torch.no_grad():
        for batch_idx, (images, captions_batch, _) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", unit="batch")):
            images, captions_batch = images.to(DEVICE), captions_batch.to(DEVICE)
            decoder_input_captions_val, targets_val = captions_batch[:, :-1], captions_batch[:, 1:]
            outputs, _ = model(images, decoder_input_captions_val, vocab.pad_idx)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets_val.reshape(-1))
            total_loss += loss.item()
            for i in range(images.size(0)):
                image_single = images[i].unsqueeze(0)
                ref_text = vocab.textualize(captions_batch[i].tolist())
                references_corpus.append([vocab.tokenize_for_bleu(ref_text, vocab.sp_model)])
                
                generated_caption_text = model.module.generate_caption_beam_search(image_single, vocab, beam_width=3, max_sample_length=max_caption_length) \
                                         if isinstance(model, nn.DataParallel) else \
                                         model.generate_caption_beam_search(image_single, vocab, beam_width=3, max_sample_length=max_caption_length)
                hypotheses_corpus.append(vocab.tokenize_for_bleu(generated_caption_text, vocab.sp_model))
                if batch_idx == 0 and i < 1:
                    print(f"  Sample Eval {i}: Ref: {ref_text} || Hyp: {generated_caption_text}")
    avg_val_loss = total_loss / len(val_loader)
    bleu_scores = {f'BLEU-{i}': corpus_bleu(references_corpus, hypotheses_corpus, weights=tuple(1/i for _ in range(i))) for i in range(1, 5)}
    print(f"Validation Results - Epoch {epoch+1}: Avg Loss: {avg_val_loss:.4f}")
    for name, score in bleu_scores.items(): print(f"{name}: {score:.4f}")
    return avg_val_loss, bleu_scores

# Step 5: Plotting Utilities
# -----------------------------------------------------------------------------
print("\nStep 5: Defining Plotting Utilities...")
def plot_metrics(train_losses, val_losses, bleu_scores_history, epoch_durations, val_plot_epochs_list, plot_filename):
    epochs_range_train = range(1, len(train_losses) + 1)
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range_train, train_losses, 'bo-', label='Training Loss')
    if val_losses and val_plot_epochs_list:
        plt.plot(val_plot_epochs_list, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 3, 2)
    if bleu_scores_history and val_plot_epochs_list:
        for i in range(1, 5):
            bleu_i_scores = [scores[f'BLEU-{i}'] for scores in bleu_scores_history]
            if bleu_i_scores:
                plt.plot(val_plot_epochs_list, bleu_i_scores, marker='o', linestyle='-', label=f'BLEU-{i}')
    plt.title('Validation BLEU Scores'); plt.xlabel('Epochs'); plt.ylabel('BLEU Score')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 3, 3)
    if epoch_durations:
        epochs_range_epoch_dur = range(1, len(epoch_durations) + 1)
        plt.plot(epochs_range_epoch_dur, [d/60 for d in epoch_durations], 'go-', label='Epoch Duration (mins)')
        plt.title('Epoch Duration'); plt.xlabel('Epochs'); plt.ylabel('Duration (minutes)')
        plt.legend(); plt.grid(True)
    
    plt.tight_layout(); plt.savefig(plot_filename); plt.show()
    print(f"Metrics plot saved as {plot_filename}")

# Step 6: CSV Generation Utility
# -----------------------------------------------------------------------------
print("\nStep 6: Defining CSV Generation Utility...")
def generate_predictions_csv(model, image_names_list, captions_data_dict, vocab, image_transform,
                             max_pred_len, img_dir, device, output_csv_file):
    print(f"\nGenerating predictions CSV for {len(image_names_list)} images to {output_csv_file}...")
    model.eval()
    results = [] 

    with torch.no_grad():
        for img_name in tqdm(image_names_list, desc="Generating predictions for CSV"):
            img_path = os.path.join(img_dir, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = image_transform(image).unsqueeze(0).to(device)

                generated_caption_text = model.module.generate_caption_beam_search(image_tensor, vocab, beam_width=5, max_sample_length=max_pred_len) \
                                         if isinstance(model, nn.DataParallel) else \
                                         model.generate_caption_beam_search(image_tensor, vocab, beam_width=5, max_sample_length=max_pred_len)
                
                original_captions_list = captions_data_dict.get(img_name, [])
                original_caption_text = original_captions_list[0] if original_captions_list else "N/A"

                results.append({
                    "image_id": img_name,
                    "original_caption": original_caption_text,
                    "generated_caption": generated_caption_text
                })
            except FileNotFoundError:
                print(f"Warning: Image file not found at {img_path}. Skipping for CSV.")
            except Exception as e:
                print(f"Warning: Error processing image {img_name} for CSV: {e}. Skipping.")
    if results:
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["image_id", "original_caption", "generated_caption"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Predictions CSV saved to {output_csv_file}")
    else:
        print("No results to save to CSV.")


# Step 7: Main Execution Block
# -----------------------------------------------------------------------------
def main():
    print("\nStep 7: Starting Main Execution...")

    # --- 7.1 Load Data and Build Vocabulary ---
    print("\n--- 7.1 Loading Data and Building Vocabulary ---")
    captions_data, all_captions_for_vocab = load_captions(CAPTION_FILE)
    vocab = SentencePieceVocabulary(SP_MODEL_PREFIX)
    sp_model_file = f"{os.path.basename(SP_MODEL_PREFIX)}.model"
    if os.path.exists(sp_model_file):
        print(f"Loading existing SentencePiece model: {sp_model_file}")
        vocab.sp_model.load(sp_model_file)
        vocab.pad_idx = vocab.sp_model.piece_to_id(vocab.pad_token)
        vocab.sos_idx = vocab.sp_model.piece_to_id(vocab.sos_token)
        vocab.eos_idx = vocab.sp_model.piece_to_id(vocab.eos_token)
        vocab.unk_idx = vocab.sp_model.piece_to_id(vocab.unk_token)
        print(f"Loaded SP vocab. Size: {len(vocab)}. IDs: P={vocab.pad_idx},S={vocab.sos_idx},E={vocab.eos_idx},U={vocab.unk_idx}")
    else:
        vocab.build_vocabulary(all_captions_for_vocab, vocab_size=SP_VOCAB_SIZE)
    print(f"Vocabulary size: {len(vocab)}")

    train_transform = transforms.Compose([
        transforms.Resize(288), transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("Defined training and validation image transforms.")

    # --- 7.2 Split Data ---
    print("\n--- 7.2 Splitting Data ---")
    unique_image_names = list(captions_data.keys())
    random.shuffle(unique_image_names)
    if not os.path.isdir(IMAGE_DIR): print(f"Error: IMAGE_DIR '{IMAGE_DIR}' not found."); return
    
    verified_image_names = [img_name for img_name in tqdm(unique_image_names, desc="Verifying images") 
                            if os.path.exists(os.path.join(IMAGE_DIR, img_name))]
    if not verified_image_names: print("Error: No valid image files found. Aborting."); return
    print(f"Found {len(verified_image_names)} images common to captions and directory.")
    
    split_idx = int(0.8 * len(verified_image_names))
    train_image_names, val_image_names = verified_image_names[:split_idx], verified_image_names[split_idx:]
    print(f"Train images: {len(train_image_names)}, Val images: {len(val_image_names)}")

    train_dataset = FlickrDataset(IMAGE_DIR, captions_data, train_image_names, vocab, train_transform, max_caption_length)
    val_dataset = FlickrDataset(IMAGE_DIR, captions_data, val_image_names, vocab, val_transform, max_caption_length)
    if len(train_dataset) == 0 or len(val_dataset) == 0: print("Error: Training or validation dataset is empty."); return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train loader: {len(train_loader)} batches. Val loader: {len(val_loader)} batches.")

    # --- 7.3 Initialize Model, Optimizer, Criterion, Scheduler ---
    print("\n--- 7.3 Initializing Model, Optimizer, Criterion, Scheduler ---")
    encoder = EncoderCNN().to(DEVICE)
    decoder = TransformerDecoderModel(len(vocab), d_model, nhead, num_transformer_decoder_layers, 
                                      dim_feedforward, encoder_dim, transformer_dropout, max_caption_length).to(DEVICE)
    model = ImageCaptioningModel(encoder, decoder).to(DEVICE)
    if torch.cuda.device_count() > 1: print(f"Using {torch.cuda.device_count()} GPUs!"); model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, label_smoothing=0.1) 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.05) 
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=lr_patience, verbose=True)
    print(f"Model initialized. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 7.4 Training Loop ---
    print("\n--- 7.4 Starting Training Loop ---")
    train_losses, val_losses, bleu_scores_history, epoch_durations_list = [], [], [], []
    recorded_val_epochs = [] # To store epoch numbers where validation was performed
    best_bleu4, epochs_no_improve = 0.0, 0
    overall_train_start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, epoch_duration = train_one_epoch(model, train_loader, optimizer, criterion, vocab, grad_clip_norm, epoch, num_epochs)
        train_losses.append(train_loss)
        epoch_durations_list.append(epoch_duration)
        print(f"Epoch {epoch+1} - Average Training Loss: {train_loss:.4f}")
        
        if ((epoch+1) % eval_frequency == 0) or epoch == num_epochs - 1 :
            val_loss, current_bleu_scores = evaluate_model(model, val_loader, criterion, vocab, epoch, num_epochs)
            val_losses.append(val_loss)
            bleu_scores_history.append(current_bleu_scores)
            recorded_val_epochs.append(epoch + 1)
            
            current_bleu4 = current_bleu_scores['BLEU-4']
            lr_scheduler.step(current_bleu4)
            if current_bleu4 > best_bleu4:
                print(f"BLEU-4 improved from {best_bleu4:.4f} to {current_bleu4:.4f}. Saving model...")
                best_bleu4 = current_bleu4
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), MODEL_SAVE_PATH)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"BLEU-4 did not improve. Best: {best_bleu4:.4f}. No improvement for {epochs_no_improve} eval cycles.")
    
            if epochs_no_improve * eval_frequency >= patience_early_stop: # epochs_no_improve counts eval cycles
                print(f"Early stopping after {epochs_no_improve * eval_frequency} epochs without BLEU-4 improvement."); break
            
    overall_train_end_time = time.time()
    total_training_time = overall_train_end_time - overall_train_start_time
    print(f"\nTraining finished. Total training time: {total_training_time // 3600:.0f}h {(total_training_time % 3600) // 60:.0f}m {total_training_time % 60:.2f}s")

    # --- 7.5 Plot Metrics ---
    if train_losses: # Ensure there's something to plot
        print("\n--- 7.5 Plotting Metrics ---")
        plot_filename = f"training_metrics_transformer_augmented_v2_ep{num_epochs}.png"
        plot_metrics(train_losses, val_losses, bleu_scores_history, epoch_durations_list, recorded_val_epochs, plot_filename)
    else:
        print("\n--- 7.5 Skipping Plotting: No metrics recorded. ---")

    # --- 7.6 Load Best Model and Generate Predictions CSV for ALL verified images ---
    print("\n--- 7.6 Generating Full Predictions CSV ---")
    if os.path.exists(MODEL_SAVE_PATH) and verified_image_names:
        encoder_final = EncoderCNN().to(DEVICE)
        decoder_final = TransformerDecoderModel(len(vocab), d_model, nhead, num_transformer_decoder_layers, 
                                                dim_feedforward, encoder_dim, transformer_dropout, max_caption_length).to(DEVICE)
        final_model = ImageCaptioningModel(encoder_final, decoder_final).to(DEVICE)
        
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        if all(key.startswith('module.') for key in state_dict.keys()): # Saved from DataParallel
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items(): new_state_dict[k[7:]] = v
            final_model.load_state_dict(new_state_dict)
        else: # Saved from a non-DataParallel model
            final_model.load_state_dict(state_dict)
        
        final_model.eval()
        
        generate_predictions_csv(
            model=final_model,
            image_names_list=verified_image_names, # Use all verified images
            captions_data_dict=captions_data,      # Full captions dictionary
            vocab=vocab,
            image_transform=val_transform,         # Use validation transform for consistency
            max_pred_len=max_caption_length,
            img_dir=IMAGE_DIR,
            device=DEVICE,
            output_csv_file=CSV_OUTPUT_FILE
        )
    else:
        print(f"Skipping full CSV generation: Model '{MODEL_SAVE_PATH}' not found or no verified images.")

    # --- 7.7 Example Usage (Sample from validation set) ---
    print("\n--- 7.7 Example Usage (Generating caption for a sample validation image) ---")
    if os.path.exists(MODEL_SAVE_PATH) and len(val_dataset) > 0:
        # Model is already loaded as final_model if CSV generation ran, or load it again
        if 'final_model' not in locals() or final_model is None: # If CSV gen was skipped
            encoder_ex = EncoderCNN().to(DEVICE)
            decoder_ex = TransformerDecoderModel(len(vocab), d_model, nhead, num_transformer_decoder_layers, 
                                                    dim_feedforward, encoder_dim, transformer_dropout, max_caption_length).to(DEVICE)
            final_model = ImageCaptioningModel(encoder_ex, decoder_ex).to(DEVICE)
            state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            if all(key.startswith('module.') for key in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items(): new_state_dict[k[7:]] = v
                final_model.load_state_dict(new_state_dict)
            else:
                final_model.load_state_dict(state_dict)
            final_model.eval()

        sample_idx = random.randint(0, len(val_dataset)-1)
        sample_img_tensor, sample_caption_tensor, _ = val_dataset[sample_idx]
        original_caption_text = vocab.textualize(sample_caption_tensor.tolist())
        
        # Display the image (optional, if matplotlib is available for this)
        plt.figure(figsize=(5,5))
        plt.imshow(sample_img_tensor.permute(1, 2, 0).cpu().numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) # Unnormalize for display
        plt.title("Sample Image for Captioning")
        plt.axis('off'); plt.show()

        print(f"Original Caption (sample): {original_caption_text}")
        generated_caption = final_model.generate_caption_beam_search(sample_img_tensor, vocab, beam_width=5, max_sample_length=max_caption_length)
        print(f"Generated Caption (sample): {generated_caption}")
    else:
        print("Skipping example usage: Best model not saved or validation set empty.")

    print("\nScript finished.")

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' not found. Downloading...")
        nltk.download('punkt', quiet=True)
    main()