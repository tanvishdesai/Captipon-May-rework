
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Ensure transforms is imported
import torchvision.models as models
import collections
from PIL import Image
import nltk # For word tokenization in stats
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
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

# --- File Paths & General Config ---
IMAGE_DIR = "/kaggle/input/flickr8k/Images" # Change to your image directory
CAPTION_FILE = "/kaggle/input/multi-lingual-flickr8k/swedish_captions_8k.txt" # Change to your caption file
# CAPTION_FILE = "/kaggle/input/rus-captions/russian_captions.txt" # Example for Russian
# CAPTION_FILE = "/kaggle/input/tel-captions/telugu_captions.txt" # Example for Telugu

MODEL_SAVE_PATH = "./best_transformer_caption_model_swedish.pth"
SP_MODEL_PREFIX = "caption_sp_diag_v1" # Prefix for SentencePiece model files
CSV_OUTPUT_FILE = "all_images_predictions_swedish-type-1.csv"
PLOT_FILENAME_TEMPLATE = "training_metrics_diag_v1_ep{epochs}_vocab{vocab_size}.png"


# --- SentencePiece Vocabulary Config ---
# Set SP_VOCAB_SIZE_TARGET to a specific integer (e.g., 8000) to aim for that size.
# Set to 0 or negative to use heuristic based on corpus unique words.
SP_VOCAB_SIZE_TARGET = 8000  # Target vocab size for SP training, or 0 for heuristic
MIN_HEURISTIC_VOCAB = 4000      # Min vocab size if using heuristic
MAX_HEURISTIC_VOCAB = 20000     # Max vocab size if using heuristic
HEURISTIC_UNIQUE_WORD_FACTOR = 0.6 # Factor of unique space-split "words" for heuristic vocab size

# --- Model Hyperparameters ---
d_model = 512
encoder_dim = 2048
nhead = 8
num_transformer_decoder_layers = 6
dim_feedforward = 2048
transformer_dropout = 0.15

# --- Training params ---
num_epochs = 20 # Adjust as needed
batch_size = 32
learning_rate = 1e-4
grad_clip_norm = 5.0
max_caption_length = 70 # Max length for PADDED captions (subword tokens)
patience_early_stop = 7 # In terms of eval_frequency cycles
lr_patience = 3 # In terms of eval_frequency cycles
eval_frequency = 1 # Evaluate every epoch for more granular data, or increase for speed

# For reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Download NLTK's punkt tokenizer for word counting in stats if not present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Could not check/download NLTK 'punkt': {e}. Word tokenization for stats might be basic.")


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
        # Placeholder for IDs, will be set after loading/training
        self.pad_idx, self.sos_idx, self.eos_idx, self.unk_idx = 0, 0, 0, 0


    def build_vocabulary(self, sentence_list, vocab_size):
        print(f"Building SentencePiece vocabulary with target size: {vocab_size}...")
        # Ensure sentence_list is not empty
        if not sentence_list:
            raise ValueError("Cannot build vocabulary from an empty sentence list.")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding='utf-8') as tmp_file:
            for sentence in sentence_list:
                tmp_file.write(sentence + "\n")
            temp_train_file_path = tmp_file.name
        
        model_filename_base = os.path.basename(self.model_prefix)
        
        # Check if model_prefix includes a directory. If so, create it.
        model_dir = os.path.dirname(self.model_prefix)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            
        # Use model_prefix directly if it's just a name, or combine dir and base.
        # SentencePieceTrainer expects model_prefix to be a path prefix.
        spm_model_path_prefix = self.model_prefix 

        try:
            spm.SentencePieceTrainer.train(
                input=temp_train_file_path,
                model_prefix=spm_model_path_prefix, # Use the potentially path-inclusive prefix
                vocab_size=vocab_size,
                user_defined_symbols=[], # Already handled by special token args
                pad_id=0, pad_piece=self.pad_token,
                unk_id=1, unk_piece=self.unk_token,
                bos_id=2, bos_piece=self.sos_token,
                eos_id=3, eos_piece=self.eos_token,
                model_type='bpe', 
                character_coverage=0.9995, # Common practice
                input_sentence_size=2000000, # Handle large datasets
                shuffle_input_sentence=True,
            )
        except Exception as e:
            print(f"Error during SentencePiece training: {e}")
            print(f"  Input file: {temp_train_file_path}")
            print(f"  Model prefix: {spm_model_path_prefix}")
            raise
        finally:
            if os.path.exists(temp_train_file_path):
                 os.remove(temp_train_file_path)
        
        self.sp_model.load(f"{spm_model_path_prefix}.model")
        self._set_special_token_ids()
        print(f"SentencePiece vocabulary built. Actual size: {len(self)}")

    def load_vocabulary(self, model_file_path=None):
        if model_file_path is None:
            model_file_path = f"{self.model_prefix}.model"
        
        if not os.path.exists(model_file_path):
            print(f"SentencePiece model file not found at {model_file_path}")
            return False
            
        self.sp_model.load(model_file_path)
        self._set_special_token_ids()
        print(f"SentencePiece vocabulary loaded. Actual size: {len(self)}")
        return True

    def _set_special_token_ids(self):
        self.pad_idx = self.sp_model.piece_to_id(self.pad_token)
        self.sos_idx = self.sp_model.piece_to_id(self.sos_token)
        self.eos_idx = self.sp_model.piece_to_id(self.eos_token)
        self.unk_idx = self.sp_model.piece_to_id(self.unk_token)

        print(f"Special token IDs: PAD={self.pad_idx}, SOS={self.sos_idx}, EOS={self.eos_idx}, UNK={self.unk_idx}")
        # Default IDs for SentencePiece with these settings
        expected_pad_idx, expected_unk_idx, expected_bos_idx, expected_eos_idx = 0, 1, 2, 3
        if self.pad_idx != expected_pad_idx:
            print(f"Warning: PAD ID is {self.pad_idx}, expected {expected_pad_idx}")
        if self.unk_idx != expected_unk_idx:
            print(f"Warning: UNK ID is {self.unk_idx}, expected {expected_unk_idx}")
        if self.sos_idx != expected_bos_idx: # BOS in SP corresponds to SOS here
            print(f"Warning: SOS (BOS) ID is {self.sos_idx}, expected {expected_bos_idx}")
        if self.eos_idx != expected_eos_idx:
            print(f"Warning: EOS ID is {self.eos_idx}, expected {expected_eos_idx}")


    def __len__(self):
        return self.sp_model.get_piece_size() if self.sp_model and self.sp_model.get_piece_size() > 0 else 0

    def numericalize(self, text):
        tokens = self.sp_model.encode_as_ids(text)
        return [self.sos_idx] + tokens + [self.eos_idx]

    def textualize(self, indices):
        # Filter out special tokens before decoding for cleaner text
        filtered_indices = [idx for idx in indices if idx not in [self.sos_idx, self.eos_idx, self.pad_idx]]
        return self.sp_model.decode_ids(filtered_indices)
    
    @staticmethod
    def tokenize_for_bleu(text, sp_model_instance): # Ensure sp_model_instance is passed
        if not text or not sp_model_instance: return []
        return sp_model_instance.encode_as_pieces(text)

    def get_corpus_stats(self, sentence_list):
        if not sentence_list or not self.sp_model or self.sp_model.get_piece_size() == 0 :
            return 0, 0, 0
        
        total_sentences = len(sentence_list)
        total_words = 0
        total_subwords = 0
        
        print("Calculating corpus stats (subword analysis)...")
        for sentence in tqdm(sentence_list, desc="Analyzing corpus for subword stats"):
            try:
                # Use NLTK for word tokenization if available and works
                words_in_sentence = nltk.word_tokenize(sentence)
            except Exception: # Fallback for languages where punkt might not be ideal or if nltk fails
                words_in_sentence = sentence.split() # Basic space-based tokenization
            
            total_words += len(words_in_sentence)
            
            subword_pieces = self.sp_model.encode_as_pieces(sentence)
            total_subwords += len(subword_pieces)
            
        avg_subwords_per_sentence = total_subwords / total_sentences if total_sentences > 0 else 0
        avg_subwords_per_word = total_subwords / total_words if total_words > 0 else 0
        avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0
        
        return avg_subwords_per_sentence, avg_subwords_per_word, avg_words_per_sentence

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
                        print(f"Warning: Skipping malformed line {line_num+1}: '{line}' (expected tab separation)")
                        continue
                    img_id_part, caption_text = parts
                    # Assuming img_id_part format like "image_name.jpg#caption_index"
                    img_name = img_id_part.split('#')[0] 
                    captions_dict[img_name].append(caption_text)
                    all_captions_for_vocab.append(caption_text)
                except Exception as e:
                    print(f"Warning: Error parsing line {line_num+1}: '{line}'. Error: {e}")
                    continue
        print(f"Loaded captions for {len(captions_dict)} unique images from {len(all_captions_for_vocab)} total caption lines.")
        if not captions_dict:
            raise ValueError("No captions loaded. Check caption file format and path.")
        return captions_dict, all_captions_for_vocab
    except FileNotFoundError:
        print(f"Error: Caption file not found at {filepath}"); raise
    except Exception as e:
        print(f"An error occurred while loading captions: {e}"); raise

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_dict, image_names, vocab, transform, max_len, dataset_name="Dataset"):
        self.root_dir = root_dir
        self.df = [] 
        self.dataset_name = dataset_name
        self.truncated_captions_count = 0
        self.captions_processed = 0

        for img_name in image_names:
            if img_name in captions_dict:
                for caption_text in captions_dict[img_name]:
                    self.df.append((img_name, caption_text))
        
        if not self.df and image_names: # If image_names were provided but no pairs were formed
            print(f"Warning: Dataset '{self.dataset_name}' initialized but no image-caption pairs formed from {len(image_names)} provided image names.")
            print("This could be due to a mismatch between image names in the split and keys in captions_dict.")
        elif not self.df:
             print(f"Warning: Dataset '{self.dataset_name}' initialized with no image-caption pairs (image_names list might be empty).")


        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len
        # print(f"Dataset '{self.dataset_name}' initialized with {len(self.df)} image-caption pairs.")
        # if not self.df and len(image_names) > 0 : # Only raise error if we expected data
        #     raise ValueError(f"Dataset '{self.dataset_name}' is empty despite receiving image names. Check data paths and caption file integrity.")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self.captions_processed +=1
        img_name, caption_text = self.df[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path} for dataset '{self.dataset_name}'. Using dummy image.")
            image = Image.new('RGB', (256, 256), color = 'red') 
            # Minimal valid caption: <sos> <eos> <pad>...
            caption_vec = torch.full((self.max_len,), self.vocab.pad_idx, dtype=torch.long)
            caption_vec[0] = self.vocab.sos_idx
            caption_vec[1] = self.vocab.eos_idx 
            return self.transform(image), caption_vec, torch.tensor(2, dtype=torch.long) # length 2

        image = self.transform(image)
        numericalized_caption = self.vocab.numericalize(caption_text)
        caption_len = len(numericalized_caption)
        
        padded_caption = torch.full((self.max_len,), self.vocab.pad_idx, dtype=torch.long)
        
        actual_len_for_tensor = caption_len
        if caption_len > self.max_len:
            padded_caption[:] = torch.tensor(numericalized_caption[:self.max_len], dtype=torch.long)
            # Ensure last token is EOS if truncated
            if padded_caption[-1] != self.vocab.eos_idx and self.vocab.eos_idx is not None:
                 padded_caption[-1] = self.vocab.eos_idx
            self.truncated_captions_count += 1
            actual_len_for_tensor = self.max_len
        else:
            padded_caption[:caption_len] = torch.tensor(numericalized_caption, dtype=torch.long)
        
        return image, padded_caption, torch.tensor(actual_len_for_tensor, dtype=torch.long)

    def get_truncation_stats(self):
        total_items = len(self.df) # total image-caption pairs
        if total_items == 0:
            return f"Dataset '{self.dataset_name}' is empty. No truncation stats."
        
        # The self.captions_processed count might be more accurate if getitem was called multiple times by dataloader workers
        # However, for a summary after init, len(self.df) * num_captions_per_image (approx) is total captions.
        # For simplicity, let's consider one pass through unique items in df for stats,
        # assuming truncation applies per caption.
        # The current self.truncated_captions_count is incremented per __getitem__ call.
        # To report after initialization, we need to iterate or estimate.
        # Let's report based on current value, assuming it's called after some processing or at end of epoch.
        # A more accurate way is to calculate this during __init__ or a separate scan.
        # For now, we'll rely on the counter being updated during dataloading.
        # A better place for this would be after an epoch or full dataset iteration.
        # Let's just report the current count and total potential items.
        
        # The issue: get_truncation_stats called after init, but before dataloading.
        # So, let's simulate one pass to get these stats accurately after init.
        simulated_truncated_count = 0
        for _, cap_text in self.df:
            num_cap = self.vocab.numericalize(cap_text)
            if len(num_cap) > self.max_len:
                simulated_truncated_count += 1
        
        percentage_truncated = (simulated_truncated_count / total_items) * 100 if total_items > 0 else 0
        return (f"Dataset '{self.dataset_name}': {simulated_truncated_count}/{total_items} captions "
                f"({percentage_truncated:.2f}%) would exceed max_len={self.max_len} and be truncated.")


# Step 3: Model Architecture (largely unchanged, added comments for clarity)
# -----------------------------------------------------------------------------
print("\nStep 3: Defining Model Architecture...")
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2] # Exclude final FC and AvgPool
        self.resnet = nn.Sequential(*modules)
        # Output will be (batch_size, 2048, H/32, W/32) e.g. (B, 2048, 8, 8) for 256x256 input
    def forward(self, images):
        features = self.resnet(images) # (B, C, H_feat, W_feat)
        batch_size = features.size(0)
        # Reshape for Transformer: (B, H_feat*W_feat, C)
        features = features.permute(0, 2, 3, 1) # (B, H_feat, W_feat, C)
        features = features.view(batch_size, -1, features.size(-1)) # (B, Num_Pixels, encoder_dim)
        return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000): # max_len for pre-calculation
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x): # x is (Batch, Seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward,
                 encoder_feature_dim, dropout=0.1, max_seq_length=100): # max_seq_length for PositionalEncoding
        super(TransformerDecoderModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_length)
        # Project encoder features to d_model if they are different
        self.encoder_projection = nn.Linear(encoder_feature_dim, d_model) if encoder_feature_dim != d_model else nn.Identity()
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size # Needed for beam search logic

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _create_padding_mask(self, sequence, pad_idx): # Not used by nn.TransformerDecoder's tgt_key_padding_mask
        # This mask is for self-attention padding, nn.TransformerDecoder handles it internally if tgt_key_padding_mask is provided
        return (sequence == pad_idx).to(DEVICE)


    def forward(self, encoder_out, tgt_captions, pad_idx):
        # encoder_out: (B, Num_Pixels, encoder_dim)
        # tgt_captions: (B, Seq_len)
        
        memory = self.encoder_projection(encoder_out) # (B, Num_Pixels, d_model)
        
        tgt_emb = self.embedding(tgt_captions) * math.sqrt(self.d_model) # (B, Seq_len, d_model)
        tgt_emb = self.pos_encoder(tgt_emb) # (B, Seq_len, d_model)
        
        tgt_seq_len = tgt_captions.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len) # (Seq_len, Seq_len)
        
        # True for positions that should be masked (padded)
        tgt_padding_mask = (tgt_captions == pad_idx) # (B, Seq_len)

        output = self.transformer_decoder(tgt=tgt_emb, memory=memory,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_padding_mask,
                                          # memory_key_padding_mask can be added if encoder_out has padding
                                          )
        logits = self.fc_out(output) # (B, Seq_len, vocab_size)
        return logits, None # Keep signature consistent with older attention models

    def sample_beam_search(self, encoder_out, vocab, beam_width=5, max_sample_length=50):
        self.eval() # Ensure model is in eval mode
        batch_size = encoder_out.size(0)
        assert batch_size == 1, "Beam search currently supports batch_size=1 for simplicity"

        memory = self.encoder_projection(encoder_out) # (1, Num_Pixels, d_model)
        # Expand memory to beam_width
        memory_k = memory.expand(beam_width, -1, -1) # (beam_width, Num_Pixels, d_model)

        # Start with <sos> token for all beams
        k_prev_words = torch.full((beam_width, 1), vocab.sos_idx, dtype=torch.long).to(DEVICE) # (beam_width, 1)
        seqs = k_prev_words # Stores current sequences for each beam
        
        # Scores for each beam, initially 0 for <sos> path, others -inf
        top_k_scores = torch.zeros(beam_width, 1).to(DEVICE) 
        top_k_scores[1:] = float('-inf')


        complete_seqs = []
        complete_seqs_scores = []

        for step in range(max_sample_length):
            # Get embeddings for current sequences
            # For the first step, seqs is (beam_width, 1). For subsequent, (beam_width, current_len)
            tgt_emb = self.embedding(seqs) * math.sqrt(self.d_model) # (beam_width, current_len, d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            
            current_seq_len = seqs.size(1)
            tgt_mask = self._generate_square_subsequent_mask(current_seq_len).to(DEVICE) # (current_len, current_len)
            
            # No padding mask needed for generated sequences as they are dense until EOS
            decoder_output = self.transformer_decoder(tgt_emb, memory_k, tgt_mask=tgt_mask) # (beam_width, current_len, d_model)
            
            # Get logits for the last token prediction
            logits = self.fc_out(decoder_output[:, -1, :]) # (beam_width, vocab_size)
            log_probs = F.log_softmax(logits, dim=1) # (beam_width, vocab_size)
            
            # Add current scores to log_probs: (beam_width, vocab_size)
            # Each row i in log_probs now represents log_probs for words extending sequence i
            log_probs = top_k_scores.expand_as(log_probs) + log_probs 
            
            if step == 0: # First step, only consider extensions from the single <sos> input (effective beam_width=1)
                # top_k_scores was (beam_width,1), log_probs is (beam_width, vocab_size)
                # we only care about the first row of log_probs (that came from the initial SOS with score 0)
                # and pick top_k from its vocab_size entries
                top_k_scores, top_k_words = log_probs[0].topk(beam_width, 0, True, True) # (beam_width), (beam_width)
                prev_beam_inds = torch.zeros(beam_width, dtype=torch.long).to(DEVICE) # All extend from beam 0
            else:
                # Flatten log_probs to (beam_width * vocab_size) to find overall top k
                top_k_scores, top_k_words = log_probs.view(-1).topk(beam_width, 0, True, True) # (beam_width), (beam_width)
                # Determine which beam each top word came from
                prev_beam_inds = torch.div(top_k_words, self.vocab_size, rounding_mode='floor') # (beam_width)
            
            next_word_inds = top_k_words % self.vocab_size # (beam_width)
            
            # Update sequences:
            # seqs was (beam_width_old, current_len)
            # We pick prev_beam_inds from seqs, making it (beam_width_new, current_len)
            # Then cat next_word_inds (beam_width_new, 1)
            seqs = torch.cat([seqs[prev_beam_inds], next_word_inds.unsqueeze(1)], dim=1) # (beam_width, current_len + 1)
            
            # Identify completed sequences (those ending in <eos>)
            is_eos = (next_word_inds == vocab.eos_idx)
            incomplete_inds = [] # Indices of beams that are not complete
            
            for i in range(is_eos.size(0)): # Iterate through current top beams
                if is_eos[i]:
                    complete_seqs.append(seqs[i, :].tolist())
                    complete_seqs_scores.append(top_k_scores[i].item()) # Score of this completed sequence
                else:
                    incomplete_inds.append(i) # This beam is not finished
            
            beam_width_new = len(incomplete_inds)
            if beam_width_new == 0: # All beams ended in EOS
                break

            # Prune down to active beams
            seqs = seqs[incomplete_inds]
            memory_k = memory_k[prev_beam_inds[incomplete_inds]] # memory_k also needs to be re-arranged based on prev_beam_inds of *active* new beams
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1) # (beam_width_new, 1)
            
            # Ensure memory_k's first dimension matches the new beam_width for the next iteration
            if beam_width_new != memory_k.size(0): # Should not happen if logic is correct
                 memory_k = memory_k[:beam_width_new]


        if not complete_seqs: # If no sequences completed (e.g., all reached max_length)
            if seqs.nelement() > 0: # Check if seqs is not empty
                 # Add all current (incomplete) sequences as candidates
                 for i in range(seqs.size(0)):
                    complete_seqs.append(seqs[i, :].tolist())
                    complete_seqs_scores.append(top_k_scores[i].item()) # Use their current scores
            else: # seqs is empty, means something went wrong or max_sample_length=0
                 return [] # Return empty list if no sequences generated

        if not complete_seqs_scores: return [] # Should be redundant if complete_seqs has items
        
        # Normalize scores by length (optional, can help with very short vs long sequences)
        # For now, just pick highest raw score.
        best_seq_idx = complete_seqs_scores.index(max(complete_seqs_scores))
        best_seq = complete_seqs[best_seq_idx]
        
        # Do not filter SOS/EOS/PAD here, textualize will do it
        return best_seq


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, pad_idx):
        encoder_out = self.encoder(images)
        outputs, _ = self.decoder(encoder_out, captions, pad_idx) # Pass pad_idx
        return outputs, None # Attn_weights are None for Transformer

    def generate_caption_beam_search(self, image, vocab, beam_width=5, max_sample_length=50):
        self.eval() # Ensure model is in eval mode
        with torch.no_grad():
            if image.dim() == 3: # If single image (C, H, W)
                image = image.unsqueeze(0) # Add batch dimension -> (1, C, H, W)
            image = image.to(DEVICE)
            
            encoder_out = self.encoder(image) # (1, Num_Pixels, encoder_dim)
            # sampled_ids is a list of token IDs including SOS/EOS
            sampled_ids_with_special_tokens = self.decoder.sample_beam_search(encoder_out, vocab, beam_width, max_sample_length)
            
            # textualize will handle stripping SOS, EOS, PAD
            generated_caption_text = vocab.textualize(sampled_ids_with_special_tokens)
            return generated_caption_text

# Step 4: Training and Evaluation Utilities
# -----------------------------------------------------------------------------
print("\nStep 4: Defining Training and Evaluation Utilities...")
def train_one_epoch(model, train_loader, optimizer, criterion, vocab, grad_clip_norm, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    epoch_start_time = time.time()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", unit="batch")
    
    for i, (images, captions, _) in enumerate(progress_bar): # caption_lengths not directly used by Transformer here
        images, captions = images.to(DEVICE), captions.to(DEVICE)
        
        # Prepare inputs for Transformer decoder:
        # Decoder input: <sos> token1 token2 ... tokenN
        # Target: token1 token2 ... tokenN <eos>
        decoder_input_captions = captions[:, :-1]
        targets = captions[:, 1:]
        
        optimizer.zero_grad()
        # Pass pad_idx to model for creating padding mask
        outputs, _ = model(images, decoder_input_captions, vocab.pad_idx) # outputs: (B, Seq_len-1, Vocab_size)
        
        # Reshape for CrossEntropyLoss: (B * (Seq_len-1), Vocab_size) and (B * (Seq_len-1))
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(loss=loss.item(), avg_loss_batch=total_loss/(i+1), lr=f"{current_lr:.1e}")
        
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch {epoch+1} - Training Duration: {epoch_duration:.2f}s ({epoch_duration/60:.2f}m)")
    return total_loss / len(train_loader), epoch_duration

def evaluate_model(model, val_loader, criterion, vocab, epoch, num_epochs):
    model.eval()
    total_loss = 0.0
    references_corpus = [] # List of lists of reference token lists
    hypotheses_corpus = [] # List of hypothesis token lists
    
    print(f"Epoch {epoch+1}/{num_epochs} [Validation] - Generating captions and calculating BLEU...")
    smoothing_fn = SmoothingFunction().method1 # For BLEU smoothing

    with torch.no_grad():
        for batch_idx, (images, captions_batch, _) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", unit="batch")):
            images, captions_batch = images.to(DEVICE), captions_batch.to(DEVICE)
            
            decoder_input_captions_val = captions_batch[:, :-1]
            targets_val = captions_batch[:, 1:]
            
            outputs, _ = model(images, decoder_input_captions_val, vocab.pad_idx)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets_val.reshape(-1))
            total_loss += loss.item()

            # For BLEU score, generate captions for each image in the batch
            for i in range(images.size(0)):
                image_single = images[i].unsqueeze(0) # (1, C, H, W)
                
                # Original caption tensor for this image (already on DEVICE)
                # Need to convert IDs to text, then to SP pieces for BLEU reference
                ref_ids_with_special = captions_batch[i].tolist()
                ref_text = vocab.textualize(ref_ids_with_special) # Textualize removes special tokens
                # For BLEU, tokenize the textualized reference
                # IMPORTANT: For corpus_bleu, references_corpus expects list of lists of reference tokens
                # If only one reference per image: [[ref1_toks], [ref2_toks], ...]
                references_corpus.append([vocab.tokenize_for_bleu(ref_text, vocab.sp_model)])

                # Generate hypothesis
                # Access generate_caption_beam_search from the underlying model if DataParallel is used
                caption_generating_model = model.module if isinstance(model, nn.DataParallel) else model
                generated_caption_text = caption_generating_model.generate_caption_beam_search(
                    image_single, vocab, beam_width=3, max_sample_length=max_caption_length
                )
                hypotheses_corpus.append(vocab.tokenize_for_bleu(generated_caption_text, vocab.sp_model))

                # Diagnostic: Print a few samples from the first validation batch
                if batch_idx == 0 and i < 2: # Print 2 samples
                    print(f"\n  --- Sample Eval {i+1} (Epoch {epoch+1}) ---")
                    print(f"    Original Reference (Text): {ref_text}")
                    print(f"    Ref Tokens (for BLEU): {' '.join(references_corpus[-1][0][:15])}...") # Show first 15 tokens
                    print(f"    Generated Hypothesis (Text): {generated_caption_text}")
                    print(f"    Hyp Tokens (for BLEU): {' '.join(hypotheses_corpus[-1][:15])}...") # Show first 15 tokens
                    print(f"  --- End Sample ---")

    avg_val_loss = total_loss / len(val_loader)
    
    bleu_scores = {}
    if references_corpus and hypotheses_corpus:
        for i in range(1, 5): # BLEU-1 to BLEU-4
            weights = tuple(1/i for _ in range(i))
            try:
                bleu_i_score = corpus_bleu(references_corpus, hypotheses_corpus, weights=weights, smoothing_function=smoothing_fn)
                bleu_scores[f'BLEU-{i}'] = bleu_i_score
            except ZeroDivisionError: # Can happen if no n-grams match
                print(f"Warning: ZeroDivisionError calculating BLEU-{i}. Setting score to 0.")
                bleu_scores[f'BLEU-{i}'] = 0.0
            except ValueError as e: # Can happen with very short sequences / no overlap
                 print(f"Warning: ValueError calculating BLEU-{i}: {e}. Setting score to 0.")
                 bleu_scores[f'BLEU-{i}'] = 0.0


    print(f"Validation Results - Epoch {epoch+1}: Avg Loss: {avg_val_loss:.4f}")
    for name, score in bleu_scores.items(): 
        print(f"{name}: {score:.4f}")
        
    return avg_val_loss, bleu_scores


# Step 5: Plotting Utilities
# -----------------------------------------------------------------------------
print("\nStep 5: Defining Plotting Utilities...")
def plot_metrics(train_losses, val_losses, bleu_scores_history, epoch_durations, 
                 val_plot_epochs_list, plot_filename, run_config_summary=""):
    epochs_range_train = range(1, len(train_losses) + 1)
    
    fig, axs = plt.subplots(1, 3, figsize=(22, 6)) # Adjusted for title
    fig.suptitle(f"Training Metrics\n{run_config_summary}", fontsize=10)


    # Subplot 1: Losses
    axs[0].plot(epochs_range_train, train_losses, 'bo-', label='Training Loss')
    if val_losses and val_plot_epochs_list:
        axs[0].plot(val_plot_epochs_list, val_losses, 'ro-', label='Validation Loss')
    axs[0].set_title('Training and Validation Loss'); axs[0].set_xlabel('Epochs'); axs[0].set_ylabel('Loss')
    axs[0].legend(); axs[0].grid(True)

    # Subplot 2: BLEU Scores
    if bleu_scores_history and val_plot_epochs_list:
        for i in range(1, 5):
            # Check if BLEU-i exists in scores (it might not if calculation failed)
            bleu_i_scores = [scores.get(f'BLEU-{i}', 0) for scores in bleu_scores_history] # Default to 0 if key missing
            if any(s > 0 for s in bleu_i_scores): # Only plot if there are valid scores
                axs[1].plot(val_plot_epochs_list, bleu_i_scores, marker='o', linestyle='-', label=f'BLEU-{i}')
    axs[1].set_title('Validation BLEU Scores'); axs[1].set_xlabel('Epochs'); axs[1].set_ylabel('BLEU Score')
    axs[1].legend(); axs[1].grid(True)

    # Subplot 3: Epoch Durations
    if epoch_durations:
        epochs_range_epoch_dur = range(1, len(epoch_durations) + 1)
        axs[2].plot(epochs_range_epoch_dur, [d/60 for d in epoch_durations], 'go-', label='Epoch Duration (mins)')
        # Display average epoch duration
        avg_epoch_duration_mins = np.mean(epoch_durations) / 60
        axs[2].set_title(f'Epoch Duration (Avg: {avg_epoch_duration_mins:.2f} mins/epoch)')
        axs[2].set_xlabel('Epochs'); axs[2].set_ylabel('Duration (minutes)')
        axs[2].legend(); axs[2].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.savefig(plot_filename)
    plt.show()
    print(f"Metrics plot saved as {plot_filename}")

# Step 6: CSV Generation Utility (largely unchanged)
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

                caption_generating_model = model.module if isinstance(model, nn.DataParallel) else model
                generated_caption_text = caption_generating_model.generate_caption_beam_search(
                    image_tensor, vocab, beam_width=5, max_sample_length=max_pred_len
                )
                
                original_captions_list = captions_data_dict.get(img_name, [])
                # Join multiple original captions with a delimiter for CSV, or take the first one
                original_caption_text = " | ".join(original_captions_list) if original_captions_list else "N/A"


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
        print("No results to save to CSV for CSV generation.")


# Step 7: Main Execution Block
# -----------------------------------------------------------------------------
def main():
    print("\nStep 7: Starting Main Execution...")
    main_start_time = time.time()

    # --- 7.1 Load Data and Build/Load Vocabulary ---
    print("\n--- 7.1 Loading Data and Building/Loading Vocabulary ---")
    if not os.path.exists(CAPTION_FILE):
        print(f"FATAL: Caption file not found: {CAPTION_FILE}"); return
    captions_data, all_captions_for_vocab = load_captions(CAPTION_FILE)
    if not all_captions_for_vocab:
        print("FATAL: No captions were loaded from the caption file. Exiting."); return

    vocab = SentencePieceVocabulary(SP_MODEL_PREFIX)
    sp_model_file_path = f"{SP_MODEL_PREFIX}.model" # SentencePiece saves .model and .vocab

    actual_sp_vocab_size = 0
    if os.path.exists(sp_model_file_path):
        print(f"Attempting to load existing SentencePiece model from: {sp_model_file_path}")
        if vocab.load_vocabulary(sp_model_file_path):
            actual_sp_vocab_size = len(vocab)
            print(f"Successfully loaded SP vocab. Actual size from model: {actual_sp_vocab_size}")
        else: # Fallback to building if load fails despite file existing (e.g. corrupted)
            print(f"Failed to load existing SP model, will attempt to build a new one.")
    
    if actual_sp_vocab_size == 0: # If model not found or failed to load
        print("SentencePiece model not found or failed to load. Building a new one.")
        
        target_sp_training_vocab_size = SP_VOCAB_SIZE_TARGET
        if SP_VOCAB_SIZE_TARGET <= 0: # Heuristic mode for vocab size
            print(f"SP_VOCAB_SIZE_TARGET ({SP_VOCAB_SIZE_TARGET}) <= 0. Using heuristic for SP vocab size.")
            # Simple word tokenization for heuristic. SP is robust.
            unique_corpus_tokens = set(tok for s in all_captions_for_vocab for tok in s.split())
            num_unique_corpus_tokens = len(unique_corpus_tokens)
            print(f"  Number of unique space-split tokens in corpus: {num_unique_corpus_tokens}")
            
            heuristic_size = int(num_unique_corpus_tokens * HEURISTIC_UNIQUE_WORD_FACTOR)
            target_sp_training_vocab_size = max(MIN_HEURISTIC_VOCAB, min(MAX_HEURISTIC_VOCAB, heuristic_size))
            print(f"  Heuristically determined target SP vocab size for training: {target_sp_training_vocab_size} (bounds: {MIN_HEURISTIC_VOCAB}-{MAX_HEURISTIC_VOCAB})")
        else:
            print(f"  Using configured SP_VOCAB_SIZE_TARGET: {SP_VOCAB_SIZE_TARGET} for SP training.")

        vocab.build_vocabulary(all_captions_for_vocab, vocab_size=target_sp_training_vocab_size)
        actual_sp_vocab_size = len(vocab)
        print(f"Built new SP vocab. Target during SP train: {target_sp_training_vocab_size}, Actual size post-train: {actual_sp_vocab_size}")

    if actual_sp_vocab_size == 0:
         print("FATAL: Vocabulary size is 0 after attempting to load/build. Cannot proceed."); return
    
    # --- Vocabulary Diagnostics ---
    avg_subwords_sent, avg_subwords_word, avg_words_sent = vocab.get_corpus_stats(all_captions_for_vocab)
    print(f"Vocabulary Corpus Stats: Avg Words/Sentence: {avg_words_sent:.2f}, Avg Subwords/Sentence: {avg_subwords_sent:.2f}, Avg Subwords/Word (NLTK-based): {avg_subwords_word:.2f}")
    run_summary_config_string = f"ActualVocab: {actual_sp_vocab_size}, AvgSubwords/Word: {avg_subwords_word:.2f}, MaxCapLen: {max_caption_length}"


    # --- Image Transforms ---
    train_transform = transforms.Compose([
        transforms.Resize(288), transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("Defined training and validation image transforms with augmentations.")

    # --- 7.2 Split Data ---
    print("\n--- 7.2 Splitting Data ---")
    if not os.path.isdir(IMAGE_DIR): 
        print(f"FATAL: IMAGE_DIR '{IMAGE_DIR}' not found or not a directory."); return
    
    all_image_files_in_dir = {f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))}
    # Filter unique_image_names from captions_data to only those existing in IMAGE_DIR
    # And also ensure these images actually have captions.
    
    # Images that have captions
    images_with_captions = list(captions_data.keys())
    
    # Images that exist on disk AND have captions
    verified_image_names = [img_name for img_name in tqdm(images_with_captions, desc="Verifying images (exist & have captions)") 
                            if img_name in all_image_files_in_dir]

    if not verified_image_names: 
        print("FATAL: No valid image files found that also have captions. Aborting."); return
    print(f"Found {len(verified_image_names)} images common to caption data and image directory.")
    
    random.shuffle(verified_image_names) # Shuffle before splitting
    
    # Ensure splits are not empty and handle small datasets
    num_verified = len(verified_image_names)
    if num_verified < 10: # Arbitrary small number for basic split
        print(f"Warning: Very small dataset ({num_verified} images). Using all for train and val.")
        train_image_names, val_image_names = verified_image_names, verified_image_names
    else:
        split_idx = int(0.8 * num_verified)
        if split_idx == 0 : split_idx = 1 # Ensure val set has at least one if train gets all but one
        if split_idx == num_verified: split_idx = num_verified -1 # Ensure val set has at least one

        train_image_names, val_image_names = verified_image_names[:split_idx], verified_image_names[split_idx:]

    print(f"Data Split: Train images: {len(train_image_names)}, Val images: {len(val_image_names)}")
    if not train_image_names or not val_image_names:
        print("FATAL: One of the data splits is empty. Aborting. Check dataset size and split logic."); return

    train_dataset = FlickrDataset(IMAGE_DIR, captions_data, train_image_names, vocab, train_transform, max_caption_length, dataset_name="Train")
    val_dataset = FlickrDataset(IMAGE_DIR, captions_data, val_image_names, vocab, val_transform, max_caption_length, dataset_name="Validation")
    
    # --- Dataset Diagnostics ---
    print(train_dataset.get_truncation_stats())
    print(val_dataset.get_truncation_stats())
    
    if len(train_dataset) == 0: 
        print("FATAL: Training dataset is empty after initialization. Aborting."); return
    if len(val_dataset) == 0 and len(val_image_names) > 0 : # Only fatal if we expected val images
        print("FATAL: Validation dataset is empty despite val_image_names not being empty. Aborting."); return
    elif len(val_dataset) == 0:
        print("Warning: Validation dataset is empty (val_image_names might have been empty or no pairs formed). Validation will be skipped.")


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True) if len(val_dataset) > 0 else None
    print(f"Train loader: {len(train_loader)} batches. Val loader: {len(val_loader) if val_loader else 0} batches.")

    # --- 7.3 Initialize Model, Optimizer, Criterion, Scheduler ---
    print("\n--- 7.3 Initializing Model, Optimizer, Criterion, Scheduler ---")
    encoder = EncoderCNN() # Output_dim is encoder_dim (2048 for ResNet50 features)
    decoder = TransformerDecoderModel(
        vocab_size=actual_sp_vocab_size, d_model=d_model, nhead=nhead, 
        num_decoder_layers=num_transformer_decoder_layers, 
        dim_feedforward=dim_feedforward, encoder_feature_dim=encoder_dim, 
        dropout=transformer_dropout, max_seq_length=max_caption_length
    )
    model = ImageCaptioningModel(encoder, decoder).to(DEVICE)
    
    num_encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    num_decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"Model initialized. Encoder params: {num_encoder_params:,}, Decoder params: {num_decoder_params:,}, Total: {num_encoder_params+num_decoder_params:,}")

    if torch.cuda.device_count() > 1: 
        print(f"Using {torch.cuda.device_count()} GPUs! Applying DataParallel.")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, label_smoothing=0.1) 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4) # Added weight decay
    # Scheduler uses BLEU-4 from validation
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=lr_patience, verbose=True)


    # --- 7.4 Training Loop ---
    print("\n--- 7.4 Starting Training Loop ---")
    train_losses, val_losses_history, bleu_scores_history, epoch_durations_list = [], [], [], []
    recorded_val_epochs = [] 
    best_bleu4, epochs_no_improve = 0.0, 0
    overall_train_start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss_epoch, epoch_duration = train_one_epoch(model, train_loader, optimizer, criterion, vocab, grad_clip_norm, epoch, num_epochs)
        train_losses.append(train_loss_epoch)
        epoch_durations_list.append(epoch_duration)
        print(f"Epoch {epoch+1} - Average Training Loss: {train_loss_epoch:.4f}")
        
        # Validation
        if  ((epoch == num_epochs - 1) or epoch == 10):
            val_loss, current_bleu_scores = evaluate_model(model, val_loader, criterion, vocab, epoch, num_epochs)
            val_losses_history.append(val_loss)
            bleu_scores_history.append(current_bleu_scores)
            recorded_val_epochs.append(epoch + 1)
            
            current_bleu4 = current_bleu_scores.get('BLEU-4', 0.0) # Default to 0 if not found
            lr_scheduler.step(current_bleu4) # Step scheduler based on BLEU-4

            if current_bleu4 > best_bleu4:
                print(f"BLEU-4 improved from {best_bleu4:.4f} to {current_bleu4:.4f}. Saving model to {MODEL_SAVE_PATH}")
                best_bleu4 = current_bleu4
                # Save the underlying model state_dict if using DataParallel
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), MODEL_SAVE_PATH)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"BLEU-4 ({current_bleu4:.4f}) did not improve from best ({best_bleu4:.4f}). No improvement for {epochs_no_improve} eval cycle(s).")
    
            if epochs_no_improve >= patience_early_stop:
                print(f"Early stopping triggered after {epochs_no_improve} eval cycles ({epochs_no_improve * eval_frequency} epochs) without BLEU-4 improvement.")
                break
        elif not val_loader:
            print(f"Epoch {epoch+1} - No validation loader. Saving model based on epoch.")
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), MODEL_SAVE_PATH) # Save model every epoch if no validation

    overall_train_end_time = time.time()
    total_training_time = overall_train_end_time - overall_train_start_time
    print(f"\nTraining finished. Total training time: {total_training_time // 3600:.0f}h {(total_training_time % 3600) // 60:.0f}m {total_training_time % 60:.2f}s")

    # --- 7.5 Plot Metrics ---
    if train_losses:
        print("\n--- 7.5 Plotting Metrics ---")
        # Use the actual vocab size in the plot filename for clarity
        plot_filename = PLOT_FILENAME_TEMPLATE.format(epochs=epoch+1, vocab_size=actual_sp_vocab_size)
        plot_metrics(train_losses, val_losses_history, bleu_scores_history, epoch_durations_list, 
                     recorded_val_epochs, plot_filename, run_summary_config_string)
    else:
        print("\n--- 7.5 Skipping Plotting: No training metrics recorded (e.g., num_epochs was 0). ---")

    # --- 7.6 Load Best Model and Generate Predictions CSV for ALL verified images ---
    print("\n--- 7.6 Generating Full Predictions CSV using Best Model ---")
    if os.path.exists(MODEL_SAVE_PATH) and verified_image_names:
        # Initialize a fresh model instance for loading the best weights
        encoder_final = EncoderCNN()
        decoder_final = TransformerDecoderModel(
            vocab_size=actual_sp_vocab_size, d_model=d_model, nhead=nhead,
            num_decoder_layers=num_transformer_decoder_layers, dim_feedforward=dim_feedforward,
            encoder_feature_dim=encoder_dim, dropout=transformer_dropout, max_seq_length=max_caption_length
        )
        final_model = ImageCaptioningModel(encoder_final, decoder_final).to(DEVICE)
        
        print(f"Loading best model state_dict from {MODEL_SAVE_PATH}")
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        
        # Handle cases where model was saved with/without DataParallel's 'module.' prefix
        if all(key.startswith('module.') for key in state_dict.keys()) and not isinstance(final_model, nn.DataParallel):
            print("  Removing 'module.' prefix from state_dict keys for non-DataParallel model.")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v # remove `module.`
            final_model.load_state_dict(new_state_dict)
        elif not all(key.startswith('module.') for key in state_dict.keys()) and isinstance(final_model, nn.DataParallel):
            print("  Adding 'module.' prefix to state_dict keys for DataParallel model (this case is unusual).")
            # This case is less common, usually it's the other way around.
            # If final_model is already DataParallel, it expects 'module.' if state_dict comes from DP save.
            # If state_dict does NOT have 'module.', it means it was saved from a non-DP model.
            # DP's load_state_dict can sometimes handle this, but being explicit is safer.
            # However, common practice is to save model.module.state_dict().
            final_model.load_state_dict(state_dict, strict=False) # Use strict=False if unsure about prefixes
        else:
            final_model.load_state_dict(state_dict)
        
        final_model.eval()
        
        generate_predictions_csv(
            model=final_model, # Pass the model instance itself
            image_names_list=verified_image_names, # Use all verified images for final CSV
            captions_data_dict=captions_data,
            vocab=vocab,
            image_transform=val_transform, # Use validation transform for consistency in CSV
            max_pred_len=max_caption_length,
            img_dir=IMAGE_DIR,
            device=DEVICE,
            output_csv_file=CSV_OUTPUT_FILE
        )
    else:
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"Skipping full CSV generation: Best model file '{MODEL_SAVE_PATH}' not found.")
        if not verified_image_names:
            print(f"Skipping full CSV generation: No verified images available.")


    # --- 7.7 Example Usage (Sample from validation set if it exists) ---
    print("\n--- 7.7 Example Usage (Generating caption for a sample validation image) ---")
    if os.path.exists(MODEL_SAVE_PATH) and val_dataset and len(val_dataset) > 0:
        # Ensure final_model is loaded (it should be if CSV generation ran)
        if 'final_model' not in locals() or final_model is None:
            print("Loading model for example usage as it wasn't loaded for CSV.")
            encoder_ex = EncoderCNN()
            decoder_ex = TransformerDecoderModel(
                vocab_size=actual_sp_vocab_size, d_model=d_model, nhead=nhead,
                num_decoder_layers=num_transformer_decoder_layers, dim_feedforward=dim_feedforward,
                encoder_feature_dim=encoder_dim, dropout=transformer_dropout, max_seq_length=max_caption_length
            )
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
        # __getitem__ returns image_tensor, caption_ids_tensor, caption_length_tensor
        sample_img_tensor, sample_caption_ids, _ = val_dataset[sample_idx] 
        
        # To display image, unnormalize and permute channels
        # Mean and std used for normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = sample_img_tensor.cpu().numpy().transpose((1, 2, 0)) # C,H,W -> H,W,C
        img_display = std * img_display + mean # Unnormalize
        img_display = np.clip(img_display, 0, 1)

        plt.figure(figsize=(6,6))
        plt.imshow(img_display)
        original_caption_text = vocab.textualize(sample_caption_ids.tolist())
        plt.title(f"Original (Example): {original_caption_text[:80]}...", fontsize=9) # Show first 80 chars
        plt.axis('off'); plt.show()

        print(f"Original Caption (sample validation image): {original_caption_text}")
        generated_caption = final_model.generate_caption_beam_search(
            sample_img_tensor, vocab, beam_width=5, max_sample_length=max_caption_length
        )
        print(f"Generated Caption (sample validation image): {generated_caption}")
    else:
        if not os.path.exists(MODEL_SAVE_PATH):
             print("Skipping example usage: Best model file not saved.")
        if not (val_dataset and len(val_dataset) > 0):
             print("Skipping example usage: Validation dataset is empty or not available.")

    main_end_time = time.time()
    print(f"\nScript finished. Total execution time: {(main_end_time - main_start_time)/60:.2f} minutes.")
    print(f"Run Configuration Summary: {run_summary_config_string}")
    if best_bleu4 > 0: print(f"Best BLEU-4 achieved: {best_bleu4:.4f}")


if __name__ == "__main__":
    main()