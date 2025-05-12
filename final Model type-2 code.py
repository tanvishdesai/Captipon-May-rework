import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import torchvision.models as models

import sentencepiece as spm
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import os
import re
import math
import time
import collections
import random
import csv

# Ensure reproducibility (optional, but good practice for debugging)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
   torch.cuda.manual_seed_all(42)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

# === Configuration ===
print("Step 0: Configuration Setup")
CONFIG = {
    "img_dir": "/kaggle/input/flickr8k/Images",
    "caption_file": "/kaggle/input/guj-captions/gujarati_captions.txt",
    "spm_model_prefix": "gujarati_spm_flickr8k",
    "spm_vocab_size": 8000,

    "batch_size": 32,
    "epochs": 30, # Can be adjusted
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience_early_stopping": 5, # For BLEU-4
    "grad_clip_value": 1.0,

    "image_size": 256,
    "d_model": 512,
    "n_heads": 8,
    "num_decoder_layers": 6,
    "dim_feedforward": 2048,
    "dropout_rate": 0.1,
    "max_seq_len": 50,

    "beam_width_generation": 5, # Beam width for caption generation
    "length_penalty_alpha": 0.7, # For beam search score normalization

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    "pad_token_id": 0, # Placeholder, will be updated by SPM
    "sos_token_id": 1, # Placeholder
    "eos_token_id": 2, # Placeholder
    "unk_token_id": 3, # Placeholder
    "actual_vocab_size": 8000 # Placeholder, will be updated by SPM
}
print(f"Using device: {CONFIG['device']}")

# Create dummy files if they don't exist (for local testing)
# REMOVE OR COMMENT OUT THIS SECTION WHEN RUNNING WITH ACTUAL DATA ON KAGGLE
IS_KAGGLE_ENV = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
if not IS_KAGGLE_ENV: # Only run dummy data creation if not on Kaggle
    if not os.path.exists(CONFIG['img_dir']):
        print(f"Warning: Image directory '{CONFIG['img_dir']}' not found. Creating dummy directory.")
        os.makedirs(CONFIG['img_dir'], exist_ok=True)
        try:
            Image.new('RGB', (100, 100), color = 'red').save(os.path.join(CONFIG['img_dir'], "dummy1.jpg"))
            Image.new('RGB', (100, 100), color = 'blue').save(os.path.join(CONFIG['img_dir'], "dummy2.jpg"))
        except Exception as e:
            print(f"Could not create dummy images: {e}")

    if not os.path.exists(CONFIG['caption_file']):
        print(f"Warning: Caption file '{CONFIG['caption_file']}' not found. Creating dummy file.")
        os.makedirs(os.path.dirname(CONFIG['caption_file']), exist_ok=True)
        dummy_captions_content = (
            "dummy1.jpg#0\tઆ એક લાલ છબી છે .\n"
            "dummy1.jpg#1\tલાલ ચોરસ ની છબી .\n"
            "dummy2.jpg#0\tવાદળી છબી અહીં છે .\n"
            "dummy2.jpg#1\tઆ વાદળી પૃષ્ઠભૂમિ છે .\n"
        )
        with open(CONFIG['caption_file'], "w", encoding="utf-8") as f:
            f.write(dummy_captions_content)
# END OF DUMMY DATA SECTION

# === SentencePiece Tokenizer Training ===
def train_sentencepiece_model(text_file_path, model_prefix, vocab_size_config):
    print("Step 1a: Training/Loading SentencePiece model...")
    model_file = f"{model_prefix}.model"
    
    if os.path.exists(model_file):
        print(f"SentencePiece model {model_file} already exists. Loading it.")
    else:
        print(f"Training new SentencePiece model. This may take a few minutes...")
        temp_caption_file = "temp_gujarati_captions_for_spm.txt"
        all_captions_for_spm = []
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        all_captions_for_spm.append(parts[1])
            
            with open(temp_caption_file, 'w', encoding='utf-8') as f:
                for caption in all_captions_for_spm:
                    f.write(caption + '\n')
            if not all_captions_for_spm:
                raise ValueError("No captions found to train SentencePiece model.")

            spm.SentencePieceTrainer.train(
                input=temp_caption_file,
                model_prefix=model_prefix,
                vocab_size=vocab_size_config,
                model_type='unigram',
                user_defined_symbols=['<PAD>'],
                pad_id=0, # Try to enforce PAD ID to be 0 if SentencePiece version supports this hint
                unk_id=3, # Standard: unk=0, bos=1, eos=2. If PAD is user-defined, it often becomes 3.
                          # SPM default is unk=0, bos=1, eos=2. <PAD> as user_defined_symbols[0] makes it 3.
                          # If we enforce pad_id=0, then unk, bos, eos might shift. Let's verify.
                # After SPM training, we will query actual IDs.
            )
            print(f"SentencePiece model trained and saved as {model_prefix}.model and {model_prefix}.vocab")
            os.remove(temp_caption_file)
        except Exception as e:
            print(f"Error during SentencePiece training: {e}")
            if os.path.exists(temp_caption_file): os.remove(temp_caption_file)
            raise

    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    
    CONFIG['unk_token_id'] = sp.unk_id()
    CONFIG['sos_token_id'] = sp.bos_id() 
    CONFIG['eos_token_id'] = sp.eos_id()
    
    # Try to get PAD ID for '<PAD>' symbol. User defined symbols are typically after standard ones.
    try:
        CONFIG['pad_token_id'] = sp.piece_to_id('<PAD>')
    except Exception: # Should not happen due to user_defined_symbols
        print("CRITICAL: '<PAD>' symbol not found in SPM vocab. This is unexpected.")
        # Fallback, highly undesirable
        # Check if 0 is unassigned or taken by UNK (SPM often uses 0 for UNK by default if not specified)
        if sp.unk_id() != 0 and sp.bos_id() != 0 and sp.eos_id() != 0 :
            CONFIG['pad_token_id'] = 0 # A common convention for padding
            print("Warning: Assuming PAD ID is 0 as a fallback. Verify this is correct for your SPM model.")
        else: # If 0 is taken by unk, bos, or eos, this is problematic.
            # Try to assign pad_id to 3 if it was not assigned to unk.
            if sp.unk_id() !=3 and sp.bos_id() !=3 and sp.eos_id() !=3 :
                 CONFIG['pad_token_id'] = 3
                 print("Warning: Assuming PAD ID is 3 as a fallback. Verify this is correct.")
            else:
                 # This case means unk,bos,eos, and <PAD> are not uniquely assigned.
                 # This is a setup error.
                 print("CRITICAL: PAD ID could not be resolved without conflict.")
                 # Default to 0 and hope for the best, but this needs fixing.
                 CONFIG['pad_token_id'] = 0


    CONFIG['actual_vocab_size'] = sp.get_piece_size() 
    
    print(f"SPM Loaded. Effective Vocab size: {CONFIG['actual_vocab_size']}")
    print(f"  UNK ID: {CONFIG['unk_token_id']} ('{sp.id_to_piece(CONFIG['unk_token_id'])}')")
    print(f"  SOS ID: {CONFIG['sos_token_id']} ('{sp.id_to_piece(CONFIG['sos_token_id'])}')")
    print(f"  EOS ID: {CONFIG['eos_token_id']} ('{sp.id_to_piece(CONFIG['eos_token_id'])}')")
    print(f"  PAD ID: {CONFIG['pad_token_id']} ('{sp.id_to_piece(CONFIG['pad_token_id'])}')")
    
    special_ids = {CONFIG['unk_token_id'], CONFIG['sos_token_id'], CONFIG['eos_token_id'], CONFIG['pad_token_id']}
    if len(special_ids) < 4:
        print("CRITICAL WARNING: Special token IDs are overlapping! This will cause issues.")
        print(f"  Set of IDs: {special_ids}")
        # Resolve common conflict: if unk_id is 0 and pad_id is also 0.
        # This means SPM didn't use user_defined_symbols as expected or pad_id trainer arg didn't work.
        # We want PAD to be 0 for nn.Embedding padding_idx.
        # If sp.unk_id() is 0, and sp.piece_to_id('<PAD>') is also 0, this is bad.
        # It means <PAD> IS the unk_token. This is not ideal.
        # The user_defined_symbols=['<PAD>'] should give <PAD> a distinct ID (usually 3).
        # If pad_id=0 hint in trainer worked, then unk_id should not be 0.
        # Let's assume the printout reflects the true state and proceed.
        # The CrossEntropyLoss ignore_index will use CONFIG['pad_token_id'].

    return sp

# === Dataset and DataLoader ===
class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_dict, image_keys, sp_processor, transform, max_len):
        self.root_dir = root_dir
        self.captions_dict = captions_dict # This is all_captions_dict from main
        self.image_keys = image_keys # List of image filenames for this split
        self.sp = sp_processor
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        img_key = self.image_keys[idx]
        img_path = os.path.join(self.root_dir, img_key)
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            print(f"Error: Image not found at {img_path}. Using a black image placeholder.")
            image = torch.zeros((3, CONFIG['image_size'], CONFIG['image_size']))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Using a black image placeholder.")
            image = torch.zeros((3, CONFIG['image_size'], CONFIG['image_size']))

        # For training/validation, pick one random caption
        # For inference (CSV generation), this caption is not strictly used by inference_collate_fn
        # but dataset must return it to match structure.
        captions_for_image = self.captions_dict.get(img_key)
        if not captions_for_image: # Should not happen if image_keys are from caption file
            print(f"Warning: No captions found for image {img_key}. Using empty caption.")
            chosen_caption_text = ""
        else:
            chosen_caption_text = random.choice(captions_for_image)

        tokenized_caption = [CONFIG['sos_token_id']] + \
                            self.sp.encode_as_ids(chosen_caption_text) + \
                            [CONFIG['eos_token_id']]
        
        caption_tensor = torch.tensor(tokenized_caption, dtype=torch.long)
        
        if len(caption_tensor) > self.max_len:
            caption_tensor = caption_tensor[:self.max_len-1] # Truncate
            caption_tensor[-1] = CONFIG['eos_token_id'] # Ensure EOS at the end
        
        return image, caption_tensor, img_key

def collate_fn(batch):
    images, captions, img_keys = zip(*batch)
    images = torch.stack(images, 0)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=CONFIG['pad_token_id'])
    return images, captions_padded, img_keys

# === Model Architecture ===
class ImageEncoder(nn.Module):
    def __init__(self, d_model, fine_tune_last_n_blocks=2):
        super().__init__()
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        for param in effnet.parameters():
            param.requires_grad = False
            
        if fine_tune_last_n_blocks > 0:
            for i in range(len(effnet.features) - fine_tune_last_n_blocks, len(effnet.features)):
                for param in effnet.features[i].parameters():
                    param.requires_grad = True
        
        self.features = effnet.features
        self.avgpool = effnet.avgpool 
        self.fc = nn.Linear(effnet.classifier[1].in_features, d_model)
        
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, images):
        x = self.features(images)  
        x = self.avgpool(x)        
        x = torch.flatten(x, 1)    
        x = self.fc(x)             
        return x 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=CONFIG['max_seq_len']):
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

class CaptioningTransformer(nn.Module):
    def __init__(self, image_encoder, vocab_size, d_model, n_heads, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.image_encoder = image_encoder
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=CONFIG['pad_token_id'])
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=CONFIG['max_seq_len'])
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        if CONFIG['pad_token_id'] is not None:
             self.embedding.weight.data[CONFIG['pad_token_id']].zero_()
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, images, captions):
        image_features = self.image_encoder(images) 
        memory = image_features.unsqueeze(1)

        tgt_for_decoder = captions[:, :-1] # Input to decoder (shifted right)
        
        tgt_emb = self.embedding(tgt_for_decoder) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        tgt_seq_len = tgt_for_decoder.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(CONFIG['device'])
        tgt_padding_mask = (tgt_for_decoder == CONFIG['pad_token_id']).to(CONFIG['device'])

        decoder_output = self.transformer_decoder(
            tgt=tgt_emb, 
            memory=memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        output_logits = self.fc_out(decoder_output)
        return output_logits

    def generate_caption(self, image, sp_processor, 
                         max_len_generation=CONFIG['max_seq_len'], 
                         beam_width=CONFIG['beam_width_generation'],
                         length_penalty_alpha=CONFIG['length_penalty_alpha']):
        self.eval()
        k = beam_width

        with torch.no_grad():
            image_feature = self.image_encoder(image.unsqueeze(0)) 
            memory = image_feature.unsqueeze(1) # (1, 1, d_model) for decoder memory

            # Start with SOS token for all beams
            # Each beam: (sequence_tensor, log_probability_score)
            initial_beam = (torch.LongTensor([[CONFIG['sos_token_id']]]).to(CONFIG['device']), 0.0)
            live_beams = [initial_beam] 
            completed_beams = []

            for _ in range(max_len_generation -1): # -1 because SOS is already counted
                if not live_beams:
                    break
                
                all_next_candidates = [] # (sequence_tensor, score)

                for current_seq_tensor, current_score in live_beams:
                    last_token_in_seq = current_seq_tensor[0, -1].item()
                    
                    # If beam already ended with EOS or is max length, move to completed
                    if last_token_in_seq == CONFIG['eos_token_id'] or current_seq_tensor.size(1) >= max_len_generation:
                        completed_beams.append((current_seq_tensor, current_score))
                        continue 

                    # Prepare input for decoder
                    tgt_emb = self.embedding(current_seq_tensor) * math.sqrt(self.d_model)
                    pos_encoded_tgt = self.pos_encoder(tgt_emb)
                    
                    current_len = current_seq_tensor.size(1)
                    gen_tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_len).to(CONFIG['device'])

                    decoder_output = self.transformer_decoder(
                        tgt=pos_encoded_tgt, 
                        memory=memory,
                        tgt_mask=gen_tgt_mask
                    )
                    last_token_logits = self.fc_out(decoder_output[:, -1, :]) # (1, vocab_size)
                    log_probs = F.log_softmax(last_token_logits, dim=-1) # (1, vocab_size)

                    # Get top k next tokens for this beam
                    top_k_log_probs, top_k_indices = torch.topk(log_probs.squeeze(0), k, dim=-1)

                    for i in range(k): # For each of the k choices
                        next_token_id = top_k_indices[i].unsqueeze(0) # Shape (1)
                        next_log_prob = top_k_log_probs[i].item()

                        new_seq_tensor = torch.cat([current_seq_tensor, next_token_id.unsqueeze(0)], dim=1) # (1, current_len + 1)
                        new_score = current_score + next_log_prob
                        all_next_candidates.append((new_seq_tensor, new_score))
                
                # If all live beams were moved to completed, nothing to expand further
                if not all_next_candidates and not any(b[0][0, -1].item() != CONFIG['eos_token_id'] and b[0].size(1) < max_len_generation for b in live_beams):
                    live_beams = [] # Clear live beams if all were processed into completed_beams or no candidates generated
                else:
                    # Prune all_next_candidates to get new live_beams and update completed_beams
                    # Sort all candidates by score (higher is better for log_probs)
                    sorted_candidates = sorted(all_next_candidates, key=lambda x: x[1], reverse=True)
                    
                    live_beams = [] # Reset live_beams for this step
                    for seq_cand, score_cand in sorted_candidates:
                        if seq_cand[0, -1].item() == CONFIG['eos_token_id'] or seq_cand.size(1) >= max_len_generation:
                            completed_beams.append((seq_cand, score_cand))
                        else:
                            if len(live_beams) < k: # Keep only k best live beams
                                live_beams.append((seq_cand, score_cand))
                        
                        # Optimization: if we have k live and k completed, we might have enough candidates
                        # However, standard beam search usually fills up k live beams first from all candidates.
                        if len(live_beams) >= k and len(completed_beams) >= k*5: # Heuristic to prune search space early
                            break 
                    
                    # Ensure live_beams does not exceed k
                    live_beams = live_beams[:k]


                # Sort completed beams by normalized score and keep top k
                if completed_beams:
                    completed_beams = sorted(
                        completed_beams, 
                        key=lambda x: x[1] / (x[0].size(1)**length_penalty_alpha if x[0].size(1) > 0 else 1.0), 
                        reverse=True
                    )
                    completed_beams = completed_beams[:k] # Keep top k completed

                # Stop if top k completed beams are found and their scores are better than live ones (advanced)
                # Or if live_beams is empty
                if len(live_beams) == 0 and len(completed_beams) > 0:
                    break


            # After loop, select best from completed_beams or from live_beams if no completed
            final_beam_tensor = None
            if completed_beams:
                # Best is the first one due to sorting with length penalty
                final_beam_tensor = completed_beams[0][0]
            elif live_beams: # If no beam reached EOS but we have live ones
                # Sort live beams with length penalty as well
                live_beams = sorted(
                    live_beams, 
                    key=lambda x: x[1] / (x[0].size(1)**length_penalty_alpha if x[0].size(1) > 0 else 1.0), 
                    reverse=True
                )
                final_beam_tensor = live_beams[0][0]
            else: # Should only happen if max_len_generation is extremely small (e.g., 1)
                final_beam_tensor = torch.LongTensor([[CONFIG['sos_token_id'], CONFIG['eos_token_id']]]).to(CONFIG['device'])
            
            final_tokens = final_beam_tensor.squeeze(0).tolist()
            
            # Decode tokens to text
            clean_tokens = []
            for tok_id in final_tokens:
                if tok_id == CONFIG['sos_token_id']:
                    continue
                if tok_id == CONFIG['eos_token_id']:
                    break 
                if tok_id == CONFIG['pad_token_id']: # Should not be generated
                    continue
                clean_tokens.append(tok_id)
            
            generated_caption_text = sp_processor.decode_ids(clean_tokens)
            return generated_caption_text, final_tokens


# === Training & Evaluation Utilities ===
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model, model_path="best_model.pth"): # Pass model to save
        improvement = False
        if self.best_score is None:
            self.best_score = score
            improvement = True
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                improvement = True
            else:
                self.counter += 1
        elif self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                improvement = True
            else:
                self.counter += 1
        
        if improvement:
            if self.verbose:
                print(f"Metric improved ({self.best_score:.6f}). Saving model to {model_path}")
            torch.save(model.state_dict(), model_path) # Save the model here
        
        if self.counter >= self.patience:
            if self.verbose: print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            self.early_stop = True
        return improvement


def calculate_bleu_scores_nltk(references_dict, candidates_dict, sp_processor):
    actual_references_tokenized_str = [] 
    predicted_candidates_tokenized_str = []

    for img_key, pred_text in candidates_dict.items():
        if img_key in references_dict:
            refs_for_image_text = references_dict[img_key]
            # Tokenize references and prediction into pieces (subwords) for NLTK BLEU
            tokenized_refs_str = [sp_processor.encode_as_pieces(ref) for ref in refs_for_image_text]
            tokenized_pred_str = sp_processor.encode_as_pieces(pred_text)
            
            actual_references_tokenized_str.append(tokenized_refs_str)
            predicted_candidates_tokenized_str.append(tokenized_pred_str)

    if not predicted_candidates_tokenized_str: return 0.0, 0.0, 0.0, 0.0

    chencherry = SmoothingFunction()
    bleu1 = corpus_bleu(actual_references_tokenized_str, predicted_candidates_tokenized_str, weights=(1.0, 0, 0, 0), smoothing_function=chencherry.method1)
    bleu2 = corpus_bleu(actual_references_tokenized_str, predicted_candidates_tokenized_str, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
    bleu3 = corpus_bleu(actual_references_tokenized_str, predicted_candidates_tokenized_str, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
    bleu4 = corpus_bleu(actual_references_tokenized_str, predicted_candidates_tokenized_str, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
    
    return bleu1, bleu2, bleu3, bleu4

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, grad_clip_value):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for images, captions_padded, _ in progress_bar:
        images, captions_padded = images.to(device), captions_padded.to(device)
        optimizer.zero_grad()
        
        # Target for loss is captions_padded[:, 1:]
        # Model output should align with this target
        output_logits = model(images, captions_padded) # captions_padded includes SOS, EOS
        
        # output_logits: (batch, seq_len-1, vocab_size)
        # captions_padded[:, 1:]: (batch, seq_len-1)
        loss = criterion(output_logits.reshape(-1, output_logits.size(-1)), 
                         captions_padded[:, 1:].reshape(-1)) # Exclude SOS from target
        loss.backward()
        if grad_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()
        
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, epoch, sp_processor, all_captions_master_dict):
    model.eval()
    total_loss = 0
    candidate_captions_dict = {} # {img_key: generated_text}
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False)

    with torch.no_grad():
        for images, captions_padded, img_keys_batch in progress_bar:
            images, captions_padded = images.to(device), captions_padded.to(device)
            output_logits = model(images, captions_padded)
            loss = criterion(output_logits.reshape(-1, output_logits.size(-1)), 
                             captions_padded[:, 1:].reshape(-1))
            total_loss += loss.item()

            # Generate captions for BLEU score calculation
            for i in range(images.size(0)):
                single_image = images[i] # This is a tensor for one image
                img_key = img_keys_batch[i]
                
                # generate_caption now includes beam_width and length_penalty_alpha from CONFIG by default
                generated_text, _ = model.generate_caption(single_image, sp_processor)
                candidate_captions_dict[img_key] = generated_text
    
    avg_loss = total_loss / len(dataloader)
    bleu1, bleu2, bleu3, bleu4 = calculate_bleu_scores_nltk(all_captions_master_dict, candidate_captions_dict, sp_processor)
    return avg_loss, bleu1, bleu2, bleu3, bleu4


# === CSV Generation Utility ===
def generate_output_csv(model_path, sp_processor, all_captions_master_dict, all_img_keys, config, output_csv_path="output_captions.csv"):
    print(f"\nGenerating output CSV with all image captions...")
    
    # Define dataset/loader for all images
    full_dataset_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = FlickrDataset(
        config['img_dir'], 
        all_captions_master_dict,
        all_img_keys,
        sp_processor, 
        full_dataset_transform, 
        config['max_seq_len']
    )
    
    def inference_collate_fn(batch):
        images, _, img_keys = zip(*batch) # We don't need captions from dataset here
        images_stacked = torch.stack(images, 0)
        return images_stacked, img_keys

    full_loader = DataLoader(
        full_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=inference_collate_fn,
        num_workers=2,
        pin_memory=True if config['device'].type == 'cuda' else False
    )

    # Load the best model
    image_enc = ImageEncoder(config['d_model'])
    loaded_model = CaptioningTransformer(
        image_encoder=image_enc, vocab_size=config['actual_vocab_size'], 
        d_model=config['d_model'], n_heads=config['n_heads'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'], dropout=config['dropout_rate']
    )
    try:
        loaded_model.load_state_dict(torch.load(model_path, map_location=config['device']))
    except FileNotFoundError:
        print(f"ERROR: Model file {model_path} not found. Cannot generate CSV.")
        return
        
    loaded_model.to(config['device'])
    loaded_model.eval()

    with torch.no_grad(), open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "original_captions", "generated_caption"])

        progress_bar = tqdm(full_loader, desc="Generating CSV Output")
        for images_batch, img_keys_batch in progress_bar:
            images_batch = images_batch.to(config['device'])
            
            for i in range(images_batch.size(0)):
                single_image = images_batch[i]
                img_key = img_keys_batch[i]
                
                generated_text, _ = loaded_model.generate_caption(
                    single_image, 
                    sp_processor,
                    max_len_generation=config['max_seq_len'],
                    beam_width=config['beam_width_generation'],
                    length_penalty_alpha=config['length_penalty_alpha']
                )
                
                original_captions_list = all_captions_master_dict.get(img_key, ["N/A"])
                original_captions_str = " | ".join(original_captions_list) # Join multiple refs with pipe
                
                writer.writerow([img_key, original_captions_str, generated_text])

    print(f"Output CSV saved to {output_csv_path}")


# === Main Script Execution ===
def main():
    print("--- Gujarati Image Captioning Training ---")
    
    print("\nStep 1: Configurations Loaded.")
    print(f"Device: {CONFIG['device']}, Batch Size: {CONFIG['batch_size']}, LR: {CONFIG['learning_rate']}")
    print(f"Model: d_model={CONFIG['d_model']}, n_heads={CONFIG['n_heads']}, dec_layers={CONFIG['num_decoder_layers']}")
    print(f"Beam width for generation: {CONFIG['beam_width_generation']}")

    sp_processor = train_sentencepiece_model(CONFIG['caption_file'], CONFIG['spm_model_prefix'], CONFIG['spm_vocab_size'])

    print("\nStep 2: Loading and Preprocessing Data...")
    all_captions_dict = collections.defaultdict(list)
    all_image_filenames_from_captions = set()

    try:
        with open(CONFIG['caption_file'], 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) != 2:
                    print(f"Warning: Malformed line #{line_idx+1}: '{line}'. Skipping.")
                    continue
                img_info, caption_text = parts
                img_filename = img_info.split('#')[0]
                all_captions_dict[img_filename].append(caption_text)
                all_image_filenames_from_captions.add(img_filename)
    except FileNotFoundError:
        print(f"CRITICAL: Caption file '{CONFIG['caption_file']}' not found. Exiting.")
        return
    
    print(f"Total unique images found in caption file: {len(all_image_filenames_from_captions)}")
    if not all_image_filenames_from_captions:
        print("CRITICAL: No image filenames loaded from caption file. Check file format and path.")
        return
    
    # Ensure all images mentioned in captions actually exist (optional, good for robustness)
    # This can be slow if img_dir is large and not all images are used.
    # For now, we trust the caption file.
    
    image_keys_list_all = sorted(list(all_image_filenames_from_captions))
    random.Random(42).shuffle(image_keys_list_all) 
    
    train_split_idx = int(len(image_keys_list_all) * 0.8)
    val_split_idx = int(len(image_keys_list_all) * 0.9)
    
    train_image_keys = image_keys_list_all[:train_split_idx]
    val_image_keys = image_keys_list_all[train_split_idx:val_split_idx]
    # test_image_keys = image_keys_list_all[val_split_idx:] # If you want a test set
    
    print(f"Training set size: {len(train_image_keys)}, Validation set size: {len(val_image_keys)}")

    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FlickrDataset(CONFIG['img_dir'], all_captions_dict, train_image_keys, sp_processor, train_transform, CONFIG['max_seq_len'])
    val_dataset = FlickrDataset(CONFIG['img_dir'], all_captions_dict, val_image_keys, sp_processor, val_transform, CONFIG['max_seq_len'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True if CONFIG['device'].type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True if CONFIG['device'].type == 'cuda' else False)
    print("DataLoaders created.")

    print("\nStep 3: Defining Model...")
    image_enc = ImageEncoder(CONFIG['d_model'])
    caption_model = CaptioningTransformer(
        image_encoder=image_enc, vocab_size=CONFIG['actual_vocab_size'], 
        d_model=CONFIG['d_model'], n_heads=CONFIG['n_heads'],
        num_decoder_layers=CONFIG['num_decoder_layers'],
        dim_feedforward=CONFIG['dim_feedforward'], dropout=CONFIG['dropout_rate']
    ).to(CONFIG['device'])
    
    num_params = sum(p.numel() for p in caption_model.parameters() if p.requires_grad)
    print(f"Model defined. Trainable parameters: {num_params / 1e6:.2f}M")

    print("\nStep 4: Setting up Optimizer, Loss, and Scheduler...")
    optimizer = optim.AdamW(caption_model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG['pad_token_id'])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=CONFIG['patience_early_stopping'] // 2, verbose=True)
    
    best_bleu4_model_path = "best_caption_model_bleu4.pth"
    early_stopper = EarlyStopping(patience=CONFIG['patience_early_stopping'], mode='max', verbose=True)

    print("Optimizer, Loss, Scheduler, EarlyStopper initialized.")

    print("\nStep 5: Starting Training...")
    history = collections.defaultdict(list)
    # history = {'train_loss': [], 'val_loss': [], 'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': [], 
    #            'epoch_duration': [], 'gpu_mem_peak_epoch': []}


    for epoch in range(CONFIG['epochs']):
        if CONFIG['device'].type == 'cuda':
            torch.cuda.reset_peak_memory_stats(CONFIG['device']) # Reset for current epoch peak measurement
        
        start_time_epoch = time.time()
        
        train_loss = train_one_epoch(caption_model, train_loader, optimizer, criterion, CONFIG['device'], epoch, CONFIG['grad_clip_value'])
        val_loss, bleu1, bleu2, bleu3, bleu4 = evaluate(caption_model, val_loader, criterion, CONFIG['device'], epoch, sp_processor, all_captions_dict)
        
        epoch_duration = time.time() - start_time_epoch
        history['epoch_duration'].append(epoch_duration)

        if CONFIG['device'].type == 'cuda':
            peak_mem_epoch = torch.cuda.max_memory_allocated(CONFIG['device']) / (1024**2) # In MB
            history['gpu_mem_peak_epoch'].append(peak_mem_epoch)
        else:
            history['gpu_mem_peak_epoch'].append(0)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Time: {epoch_duration:.2f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        print(f"  BLEU Scores: B1={bleu1:.4f}, B2={bleu2:.4f}, B3={bleu3:.4f}, B4={bleu4:.4f}")
        if CONFIG['device'].type == 'cuda':
            print(f"  Peak GPU Memory: {history['gpu_mem_peak_epoch'][-1]:.2f} MB")


        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['bleu1'].append(bleu1)
        history['bleu2'].append(bleu2)
        history['bleu3'].append(bleu3)
        history['bleu4'].append(bleu4)

        lr_scheduler.step(bleu4)
        
        # EarlyStopper now saves the model internally if score improves
        early_stopper(bleu4, caption_model, best_bleu4_model_path)

        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
    
    print("Training finished.")
    if os.path.exists(best_bleu4_model_path):
        print(f"Best BLEU-4 during training: {early_stopper.best_score:.4f} (model saved to {best_bleu4_model_path})")
    else:
        print(f"No model saved (possibly training ended before improvement or error). Best BLEU-4 was {early_stopper.best_score if early_stopper.best_score is not None else 'N/A'}")


    print("\nStep 6: Plotting Results...")
    if not history['train_loss']: 
        print("No history to plot. Skipping plotting.")
    else:
        epochs_ran = range(1, len(history['train_loss']) + 1)
        plt.figure(figsize=(18, 12))

        plt.subplot(2, 2, 1)
        plt.plot(epochs_ran, history['train_loss'], label='Train Loss', marker='.')
        plt.plot(epochs_ran, history['val_loss'], label='Validation Loss', marker='.')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curves')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(epochs_ran, history['bleu1'], label='BLEU-1', marker='.')
        plt.plot(epochs_ran, history['bleu2'], label='BLEU-2', marker='.')
        plt.plot(epochs_ran, history['bleu3'], label='BLEU-3', marker='.')
        plt.plot(epochs_ran, history['bleu4'], label='BLEU-4', marker='.')
        plt.xlabel('Epochs'); plt.ylabel('BLEU Score'); plt.legend(); plt.title('BLEU Scores')
        plt.grid(True)

        if history['epoch_duration']:
            plt.subplot(2, 2, 3)
            plt.plot(epochs_ran, history['epoch_duration'], label='Epoch Duration (s)', marker='.', color='green')
            plt.xlabel('Epochs'); plt.ylabel('Time (seconds)'); plt.legend(); plt.title('Epoch Duration')
            plt.grid(True)

        if CONFIG['device'].type == 'cuda' and any(m > 0 for m in history['gpu_mem_peak_epoch']):
            gpu_mems_to_plot = [m for m in history['gpu_mem_peak_epoch'] if m > 0]
            epochs_for_gpu_plot = [e for i, e in enumerate(epochs_ran) if history['gpu_mem_peak_epoch'][i] > 0]
            if gpu_mems_to_plot:
                plt.subplot(2, 2, 4)
                plt.plot(epochs_for_gpu_plot, gpu_mems_to_plot, label='Peak GPU Memory (MB)', marker='.', color='purple')
                plt.xlabel('Epochs'); plt.ylabel('Memory (MB)'); plt.legend(); plt.title('Peak GPU Memory per Epoch')
                plt.grid(True)
        elif CONFIG['device'].type != 'cuda':
             print("GPU Memory plot skipped (not running on CUDA or no data).")


        plt.tight_layout()
        plt.savefig("training_and_resource_plots.png")
        print("Plots saved to training_and_resource_plots.png")

    # Step 7: Generate CSV output using the best model
    if os.path.exists(best_bleu4_model_path):
        generate_output_csv(
            model_path=best_bleu4_model_path,
            sp_processor=sp_processor,
            all_captions_master_dict=all_captions_dict, # Original dictionary with all captions
            all_img_keys=image_keys_list_all, # List of all image keys used in the dataset
            config=CONFIG,
            output_csv_path="all_images_captions_output.csv"
        )
    else:
        print(f"Skipping CSV generation as best model file '{best_bleu4_model_path}' was not found.")

    print("\n--- Script Finished ---")

if __name__ == '__main__':
    main()