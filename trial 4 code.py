import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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

# Step 0: Setup & Configuration
# -----------------------------------------------------------------------------
print("Step 0: Initializing Setup and Configuration...")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Hyperparameters ---
IMAGE_DIR = "/kaggle/input/flickr8k/Images"
CAPTION_FILE = "/kaggle/input/guj-captions/gujarati_captions.txt"
MODEL_SAVE_PATH = "./best_transformer_caption_model.pth" # Changed model name
SP_MODEL_PREFIX = "gujarati_caption_sp" # Prefix for SentencePiece model files
SP_VOCAB_SIZE = 8000 # Desired vocabulary size for SentencePiece

# Model Hyperparameters
d_model = 768            # Dimension of embeddings and Transformer model (formerly embed_size & decoder_dim)
encoder_dim = 2048      # Dimension of encoder output (ResNet50)
nhead = 8               # Number of heads in MultiHeadAttention
num_transformer_decoder_layers = 8 # Number of Transformer decoder layers
dim_feedforward = 3072  # Dimension of feedforward network in Transformer
transformer_dropout = 0.15 # Dropout rate in Transformer

# Training params
num_epochs = 20 # Keep or adjust as needed
batch_size = 32 # Can be reduced if Transformer uses more memory
learning_rate = 1e-4 # May need tuning for Transformer
grad_clip_norm = 5.0
max_caption_length = 25 # Max length for PADDED captions (subword tokens might be more)
patience_early_stop = 5
lr_patience = 2

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
        self.unk_token = "<unk>" # Note: SentencePiece uses <unk> by default

# In class SentencePieceVocabulary:

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
            
            # --- MODIFIED SECTION START ---
            # user_defined_symbols should be for any *additional* special tokens you might have,
            # NOT for the core PAD, UNK, SOS, EOS if they are being managed by *_id and *_piece.
            user_defined_symbols=[], # Empty if no other symbols beyond PAD, UNK, SOS, EOS.
            
            pad_id=0,          # We want ID 0 for padding
            unk_id=1,          # We want ID 1 for unknown
            bos_id=2,          # We want ID 2 for beginning-of-sentence (our <sos>)
            eos_id=3,          # We want ID 3 for end-of-sentence (our <eos>)
            
            pad_piece=self.pad_token,  # Use "<pad>" string for padding
            unk_piece=self.unk_token,  # Use "<unk>" string for unknown
            bos_piece=self.sos_token,  # Use "<sos>" string for BOS
            eos_piece=self.eos_token,  # Use "<eos>" string for EOS
            # --- MODIFIED SECTION END ---
            
            model_type='bpe', 
            character_coverage=1.0,
            # unk_surface = self.unk_token # This is older/redundant if unk_piece is set. Remove for clarity.
        )
        os.remove(temp_train_file_path) 
        
        self.sp_model.load(f"{model_filename}.model")
        
        # These lines correctly retrieve the IDs assigned by SentencePiece to your token strings.
        # Given the explicit *_id and *_piece settings above, these should match 0,2,3,1.
        self.pad_idx = self.sp_model.piece_to_id(self.pad_token)
        self.sos_idx = self.sp_model.piece_to_id(self.sos_token)
        self.eos_idx = self.sp_model.piece_to_id(self.eos_token)
        self.unk_idx = self.sp_model.piece_to_id(self.unk_token)

        print(f"Special token IDs: PAD={self.pad_idx}, SOS={self.sos_idx}, EOS={self.eos_idx}, UNK={self.unk_idx}")
        # These assertions correctly verify that the retrieved IDs are what we expect.
        assert self.pad_idx == 0, f"PAD ID is not 0, it's {self.pad_idx}"
        assert self.unk_idx == 1, f"UNK ID is not 1, it's {self.unk_idx}"
        assert self.sos_idx == 2, f"SOS ID is not 2, it's {self.sos_idx}"
        assert self.eos_idx == 3, f"EOS ID is not 3, it's {self.eos_idx}"

        print(f"SentencePiece vocabulary built. Size: {len(self)}")

    def __len__(self):
        return self.sp_model.get_piece_size() if self.sp_model else 0

    def numericalize(self, text):
        # Tokenize and add SOS/EOS
        tokens = self.sp_model.encode_as_ids(text)
        return [self.sos_idx] + tokens + [self.eos_idx]

    def textualize(self, indices):
        # Convert indices to text, filtering out special tokens for final output
        filtered_indices = [idx for idx in indices if idx not in [self.sos_idx, self.eos_idx, self.pad_idx]]
        return self.sp_model.decode_ids(filtered_indices)
    
    @staticmethod
    def tokenize_for_bleu(text, sp_model): # For BLEU scoring, need to tokenize based on SP model pieces
        if not text: return []
        return sp_model.encode_as_pieces(text)


# Step 2: Data Loading and Preprocessing (largely same, uses new Vocab)
# -----------------------------------------------------------------------------
print("\nStep 2: Defining Data Loading and Preprocessing Utilities...")
# load_captions function remains the same as in v1

def load_captions(filepath):
    print(f"Loading captions from: {filepath}")
    captions_dict = collections.defaultdict(list)
    all_captions_for_vocab = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = line.split('\t', 1)
                    if len(parts) != 2:
                        print(f"Warning: Skipping malformed line {line_num+1}: '{line}' (expected 2 parts after split)")
                        continue
                    
                    img_id_part, caption_text = parts
                    img_name = img_id_part.split('#')[0] 
                    
                    captions_dict[img_name].append(caption_text)
                    all_captions_for_vocab.append(caption_text) # Used for SP training
                except Exception as e:
                    print(f"Warning: Error parsing line {line_num+1}: '{line}'. Error: {e}")
                    continue
        print(f"Loaded captions for {len(captions_dict)} unique images.")
        if not captions_dict:
            raise ValueError("No captions loaded. Check caption file format and path.")
        return captions_dict, all_captions_for_vocab
    except FileNotFoundError:
        print(f"Error: Caption file not found at {filepath}")
        raise
    except Exception as e:
        print(f"An error occurred while loading captions: {e}")
        raise

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_dict, image_names, vocab, transform, max_len):
        self.root_dir = root_dir
        self.df = [] 
        for img_name in image_names:
            if img_name in captions_dict:
                for caption in captions_dict[img_name]:
                    self.df.append((img_name, caption))
            else:
                print(f"Warning: Image {img_name} found in split list but not in captions_dict.")

        self.vocab = vocab # This is now SentencePieceVocabulary instance
        self.transform = transform
        self.max_len = max_len
        print(f"Dataset initialized with {len(self.df)} image-caption pairs.")
        if not self.df:
            raise ValueError("Dataset is empty. Check image names and captions_dict.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name, caption_text = self.df[idx]
        img_path = os.path.join(self.root_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}. Check IMAGE_DIR.")
            dummy_image = Image.new('RGB', (256, 256), color = 'red')
            image = self.transform(dummy_image)
            caption_vec = torch.full((self.max_len,), self.vocab.pad_idx, dtype=torch.long)
            caption_vec[0] = self.vocab.sos_idx
            caption_vec[1] = self.vocab.eos_idx
            return image, caption_vec, torch.tensor(2)

        image = self.transform(image)
        
        numericalized_caption = self.vocab.numericalize(caption_text) # Uses SP
        caption_len = len(numericalized_caption)
        
        padded_caption = torch.full((self.max_len,), self.vocab.pad_idx, dtype=torch.long)
        if caption_len > self.max_len:
            padded_caption[:] = torch.tensor(numericalized_caption[:self.max_len], dtype=torch.long)
            # Ensure EOS is the last token if truncated this way (important for Transformer)
            if padded_caption[-1] != self.vocab.eos_idx :
                 padded_caption[-1] = self.vocab.eos_idx 
            caption_len = self.max_len
        else:
            padded_caption[:caption_len] = torch.tensor(numericalized_caption, dtype=torch.long)
            
        return image, padded_caption, torch.tensor(caption_len, dtype=torch.long)


# Step 3: Model Architecture (EncoderCNN, PositionalEncoding, TransformerDecoderModel)
# -----------------------------------------------------------------------------
print("\nStep 3: Defining Model Architecture...")

# EncoderCNN remains the same as in v1
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Output: (batch_size, encoder_dim, H/32, W/32)

    def forward(self, images):
        features = self.resnet(images)
        batch_size = features.size(0)
        features = features.permute(0, 2, 3, 1) # (batch_size, H', W', encoder_dim)
        # Flatten for Transformer: (batch_size, H'*W', encoder_dim)
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
        pe = pe.unsqueeze(0) # .transpose(0, 1) -> if not batch_first
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward,
                 encoder_feature_dim, dropout=0.1, max_seq_length=100):
        super(TransformerDecoderModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_length)
        
        # Layer to project encoder's feature dimension to d_model
        self.encoder_projection = nn.Linear(encoder_feature_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size # Store for beam search

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(DEVICE)

    def _create_padding_mask(self, sequence, pad_idx):
        # sequence: (batch_size, seq_len)
        return (sequence == pad_idx).to(DEVICE) # (batch_size, seq_len)

    def forward(self, encoder_out, tgt_captions, pad_idx):
        # encoder_out: (batch_size, num_pixels, encoder_feature_dim)
        # tgt_captions: (batch_size, seq_len) - target caption indices (for training, shifted)
        
        # Project encoder output
        # permute for nn.TransformerDecoder's memory: (num_pixels, batch_size, d_model) if not batch_first for memory
        # but if TransformerDecoderLayer is batch_first, memory should also be batch_first
        memory = self.encoder_projection(encoder_out) # (batch_size, num_pixels, d_model)

        # Prepare target for decoder
        tgt_emb = self.embedding(tgt_captions) * math.sqrt(self.d_model) # (batch_size, seq_len, d_model)
        tgt_emb = self.pos_encoder(tgt_emb) # (batch_size, seq_len, d_model)

        # Masks
        tgt_seq_len = tgt_captions.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len) # (seq_len, seq_len)
        tgt_padding_mask = self._create_padding_mask(tgt_captions, pad_idx) # (batch_size, seq_len)
        
        # Note: memory_key_padding_mask could be created if encoder_out can have padding.
        # For ResNet features of fixed size, it's often not needed unless specific images produce fewer.
        # Assuming fixed size num_pixels from encoder here.

        # Decoder forward
        # nn.TransformerDecoder expects tgt_mask of shape (L,L) and tgt_key_padding_mask of shape (N,L)
        # memory_key_padding_mask (N,S)
        output = self.transformer_decoder(tgt=tgt_emb, memory=memory,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_padding_mask)
                                          # memory_key_padding_mask=memory_padding_mask if you have one
                                          # (batch_size, seq_len, d_model)
        
        logits = self.fc_out(output) # (batch_size, seq_len, vocab_size)
        return logits, None # No alphas returned by default Transformer (can be added with custom layers)

    def sample_beam_search(self, encoder_out, vocab, beam_width=5, max_sample_length=50):
        # encoder_out: (1, num_pixels, encoder_feature_dim)
        self.eval() # Ensure model is in eval mode
        
        batch_size = encoder_out.size(0)
        assert batch_size == 1, "Beam search currently supports batch_size=1"

        # Project encoder output and expand for beams
        # memory: (1, num_pixels, d_model)
        memory = self.encoder_projection(encoder_out)
        # memory_k: (beam_width, num_pixels, d_model)
        memory_k = memory.expand(beam_width, -1, -1)

        # Initialize k beams
        # k_prev_words stores the last token of each beam sequence
        # Initially, all beams start with SOS token
        # seqs: (beam_width, 1) tensor of SOS_IDX
        k_prev_words = torch.full((beam_width, 1), vocab.sos_idx, dtype=torch.long).to(DEVICE)
        seqs = k_prev_words # (beam_width, current_seq_len)

        # top_k_scores: (beam_width, 1) tensor of scores
        top_k_scores = torch.zeros(beam_width, 1).to(DEVICE)

        complete_seqs = []
        complete_seqs_scores = []

        for step in range(max_sample_length):
            # Prepare input for this step for all k beams
            # tgt_emb: (beam_width, current_seq_len, d_model)
            tgt_emb = self.embedding(seqs) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)

            # No padding mask needed for tgt during inference as sequences are dynamically built
            # Subsequent mask is needed
            current_seq_len = seqs.size(1)
            tgt_mask = self._generate_square_subsequent_mask(current_seq_len).to(DEVICE)

            # decoder_output: (beam_width, current_seq_len, d_model)
            decoder_output = self.transformer_decoder(tgt_emb, memory_k, tgt_mask=tgt_mask)
            
            # Get logits for the *last* token of each sequence
            # logits: (beam_width, vocab_size)
            logits = self.fc_out(decoder_output[:, -1, :]) # Takes the last time step's output
            log_probs = F.log_softmax(logits, dim=1) # (beam_width, vocab_size)

            # Add current scores
            # log_probs: (beam_width, vocab_size)
            # top_k_scores: (beam_width, 1)
            log_probs = top_k_scores.expand_as(log_probs) + log_probs

            if step == 0: # First step, all beams are equal, select top k from first beam's output
                top_k_scores, top_k_words = log_probs[0].topk(beam_width, 0, True, True)
                prev_beam_inds = torch.arange(beam_width).to(DEVICE) # All come from the "first" (and only) beam
            else: # Subsequent steps
                top_k_scores, top_k_words = log_probs.view(-1).topk(beam_width, 0, True, True)
                # top_k_words are flat indices, convert to (beam_idx, word_idx)
                prev_beam_inds = torch.div(top_k_words, self.vocab_size, rounding_mode='floor') # Which beam it came from
            
            next_word_inds = top_k_words % self.vocab_size # (beam_width) - actual word indices

            # Update sequences
            # seqs: (beam_width, current_seq_len + 1)
            seqs = torch.cat([seqs[prev_beam_inds], next_word_inds.unsqueeze(1)], dim=1)
            
            # Identify completed sequences (ending with EOS)
            is_eos = (next_word_inds == vocab.eos_idx)
            
            incomplete_inds = [] # Indices of beams that are not yet complete
            for i in range(is_eos.size(0)):
                if is_eos[i]:
                    complete_seqs.append(seqs[i, :].tolist())
                    complete_seqs_scores.append(top_k_scores[i].item())
                else:
                    incomplete_inds.append(i)
            
            beam_width_new = len(incomplete_inds)
            if beam_width_new == 0: # All beams are complete
                break

            # Prune beams: select only incomplete ones
            seqs = seqs[incomplete_inds]
            memory_k = memory_k[prev_beam_inds[incomplete_inds]] # Reorder memory for new beam set
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            
            # If beam_width has changed, update it for the next expansion
            if beam_width_new != memory_k.size(0): # Should match
                 memory_k = memory_k[:beam_width_new] # Ensure correct size if any mismatch
                 
            if seqs.size(0) != beam_width_new : # Should also match
                # This indicates a potential issue if sizes don't match after pruning
                # For now, we assume they match if incomplete_inds is handled correctly
                pass


        if not complete_seqs: # If no sequence reached EOS within max_length
            # Fallback: take the best sequence from the current set of `seqs`
            if seqs.nelement() > 0: # Check if seqs is not empty
                 # Find the sequence with the highest score among the current `seqs`
                 # `top_k_scores` here corresponds to the scores of these `seqs`
                 best_idx = top_k_scores.squeeze().argmax()
                 complete_seqs.append(seqs[best_idx].tolist())
                 complete_seqs_scores.append(top_k_scores[best_idx].item())
            else: # No sequences generated at all (e.g. beam_width=0 or error)
                return []


        # Select the best sequence among all completed sequences
        if not complete_seqs_scores: return [] # Should not happen if fallback above works
        
        best_seq_idx = complete_seqs_scores.index(max(complete_seqs_scores))
        best_seq = complete_seqs[best_seq_idx]
        
        # Filter out SOS, EOS, PAD for the final output
        sampled_ids = [idx for idx in best_seq if idx not in [vocab.sos_idx, vocab.eos_idx, vocab.pad_idx]]
        return sampled_ids


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder # This is now TransformerDecoderModel

    def forward(self, images, captions, pad_idx):
        # captions here are tgt_captions (batch_size, seq_len), already shifted (e.g. excluding last token)
        encoder_out = self.encoder(images)
        # For training, input to decoder is captions[:, :-1] (excluding <EOS>)
        # The decoder's forward method will handle masking.
        outputs, _ = self.decoder(encoder_out, captions, pad_idx)
        return outputs, None # No alphas from standard transformer

    def generate_caption_beam_search(self, image, vocab, beam_width=5, max_sample_length=50):
        self.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(DEVICE)
            
            encoder_out = self.encoder(image) # (1, num_pixels, encoder_dim)
            sampled_ids = self.decoder.sample_beam_search(encoder_out, vocab, beam_width, max_sample_length)
            
            return vocab.textualize(sampled_ids) # Use new textualize method


# Step 4: Training and Evaluation Utilities (adapted for Transformer)
# -----------------------------------------------------------------------------
print("\nStep 4: Defining Training and Evaluation Utilities...")

def train_one_epoch(model, train_loader, optimizer, criterion, vocab, grad_clip_norm, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", unit="batch")
    for i, (images, captions, _) in enumerate(progress_bar): # lengths not directly used by Transformer in this setup
        images = images.to(DEVICE)
        captions = captions.to(DEVICE) # (batch_size, max_caption_length)
        
        # For Transformer:
        # Decoder input: <SOS> w1 w2 ... wn
        # Decoder target: w1 w2 ... wn <EOS>
        decoder_input_captions = captions[:, :-1] # (batch_size, max_caption_length - 1)
        targets = captions[:, 1:]                # (batch_size, max_caption_length - 1)

        optimizer.zero_grad()
        
        # outputs: (batch_size, max_caption_length - 1, vocab_size)
        outputs, _ = model(images, decoder_input_captions, vocab.pad_idx)
        
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(loss=loss.item(), avg_loss_batch=total_loss/(i+1), lr=f"{current_lr:.1e}")
        
    return total_loss / len(train_loader)


def evaluate_model(model, val_loader, criterion, vocab, epoch, num_epochs):
    model.eval()
    total_loss = 0.0
    references_corpus = []
    hypotheses_corpus = []
    
    print(f"Epoch {epoch+1}/{num_epochs} [Validation] - Generating captions and calculating BLEU...")
    with torch.no_grad():
        for batch_idx, (images, captions_batch, _) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", unit="batch")):
            images = images.to(DEVICE)
            captions_batch = captions_batch.to(DEVICE) # (batch, max_len)
            
            decoder_input_captions_val = captions_batch[:, :-1]
            targets_val = captions_batch[:, 1:]

            outputs, _ = model(images, decoder_input_captions_val, vocab.pad_idx)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets_val.reshape(-1))
            total_loss += loss.item()

            for i in range(images.size(0)):
                image_single = images[i].unsqueeze(0)
                
                # Ground truth (original, not shifted or sos/eos stripped for ref)
                ref_caption_indices = captions_batch[i].tolist()
                # Textualize using vocab, then tokenize for BLEU using SP model
                ref_text = vocab.textualize(ref_caption_indices) # Gets clean text
                ref_tokens = vocab.tokenize_for_bleu(ref_text, vocab.sp_model)
                references_corpus.append([ref_tokens])


                generated_caption_text = model.module.generate_caption_beam_search(image_single, vocab, beam_width=3, max_sample_length=max_caption_length) \
                                         if isinstance(model, nn.DataParallel) else \
                                         model.generate_caption_beam_search(image_single, vocab, beam_width=3, max_sample_length=max_caption_length)
                
                hyp_tokens = vocab.tokenize_for_bleu(generated_caption_text, vocab.sp_model)
                hypotheses_corpus.append(hyp_tokens)

                if batch_idx == 0 and i < 1: # Print first example from first val batch
                    print(f"  Sample Eval {i}:")
                    print(f"    Ref (text): {ref_text}")
                    print(f"    Hyp (text): {generated_caption_text}")

    avg_val_loss = total_loss / len(val_loader)
    
    bleu_scores = {}
    for i in range(1, 5):
        bleu_scores[f'BLEU-{i}'] = corpus_bleu(references_corpus, hypotheses_corpus, weights=tuple(1/i for _ in range(i)))
    
    print(f"Validation Results - Epoch {epoch+1}: Avg Loss: {avg_val_loss:.4f}")
    for name, score in bleu_scores.items():
        print(f"{name}: {score:.4f}")
        
    return avg_val_loss, bleu_scores

# Step 5: Plotting Utilities (remains the same as v1)
# -----------------------------------------------------------------------------
print("\nStep 5: Defining Plotting Utilities...")
def plot_metrics(train_losses, val_losses, bleu_scores_history):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    for i in range(1, 5):
        bleu_i_scores = [scores[f'BLEU-{i}'] for scores in bleu_scores_history]
        plt.plot(epochs_range, bleu_i_scores, marker='o', linestyle='-', label=f'BLEU-{i}')
    plt.title('Validation BLEU Scores'); plt.xlabel('Epochs'); plt.ylabel('BLEU Score')
    plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("training_metrics_transformer.png"); plt.show()
    print("Metrics plot saved as training_metrics_transformer.png")

# Step 6: Main Execution Block
# -----------------------------------------------------------------------------
def main():
    print("\nStep 6: Starting Main Execution...")

    # --- 6.1 Load Data and Build Vocabulary ---
    print("\n--- 6.1 Loading Data and Building Vocabulary ---")
    captions_data, all_captions_for_vocab = load_captions(CAPTION_FILE)
    
    vocab = SentencePieceVocabulary(SP_MODEL_PREFIX)
    # Check if SP model files exist, if so, load, else build
    sp_model_file = f"{os.path.basename(SP_MODEL_PREFIX)}.model"
    if os.path.exists(sp_model_file):
        print(f"Loading existing SentencePiece model: {sp_model_file}")
        vocab.sp_model.load(sp_model_file)
        vocab.pad_idx = vocab.sp_model.piece_to_id(vocab.pad_token)
        vocab.sos_idx = vocab.sp_model.piece_to_id(vocab.sos_token)
        vocab.eos_idx = vocab.sp_model.piece_to_id(vocab.eos_token)
        vocab.unk_idx = vocab.sp_model.piece_to_id(vocab.unk_token)
        print(f"Loaded SP vocab. Size: {len(vocab)}. Special IDs: PAD={vocab.pad_idx}, SOS={vocab.sos_idx}, EOS={vocab.eos_idx}, UNK={vocab.unk_idx}")

    else:
        vocab.build_vocabulary(all_captions_for_vocab, vocab_size=SP_VOCAB_SIZE)
    
    print(f"Vocabulary size: {len(vocab)}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 6.2 Split Data ---
    print("\n--- 6.2 Splitting Data ---")
    # ... (Data splitting logic from v1, make sure it's robust) ...
    unique_image_names = list(captions_data.keys())
    random.shuffle(unique_image_names)
    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: IMAGE_DIR '{IMAGE_DIR}' not found."); return
    
    verified_image_names = [img_name for img_name in tqdm(unique_image_names, desc="Verifying images") 
                            if os.path.exists(os.path.join(IMAGE_DIR, img_name))]
    if not verified_image_names:
        print("Error: No valid image files found. Aborting."); return
    print(f"Found {len(verified_image_names)} images common to captions and directory.")
    
    split_idx = int(0.8 * len(verified_image_names))
    train_image_names = verified_image_names[:split_idx]
    val_image_names = verified_image_names[split_idx:]
    
    print(f"Train images: {len(train_image_names)}, Val images: {len(val_image_names)}")

    train_dataset = FlickrDataset(IMAGE_DIR, captions_data, train_image_names, vocab, transform, max_caption_length)
    val_dataset = FlickrDataset(IMAGE_DIR, captions_data, val_image_names, vocab, transform, max_caption_length)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Training or validation dataset is empty."); return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train loader: {len(train_loader)} batches. Val loader: {len(val_loader)} batches.")


    # --- 6.3 Initialize Model, Optimizer, Criterion, Scheduler ---
    print("\n--- 6.3 Initializing Model, Optimizer, Criterion, Scheduler ---")
    encoder = EncoderCNN().to(DEVICE)
    # For TransformerDecoderModel: vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, encoder_feature_dim, dropout, max_seq_length
    decoder = TransformerDecoderModel(vocab_size=len(vocab),
                                      d_model=d_model,
                                      nhead=nhead,
                                      num_decoder_layers=num_transformer_decoder_layers,
                                      dim_feedforward=dim_feedforward,
                                      encoder_feature_dim=encoder_dim,
                                      dropout=transformer_dropout,
                                      max_seq_length=max_caption_length).to(DEVICE)
    model = ImageCaptioningModel(encoder, decoder).to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, label_smoothing=0.1) # Use pad_idx from SP vocab
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01) # Typical AdamW wd # Adam params often used for Transformers
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=lr_patience, verbose=True)
    
    print("Model, optimizer, criterion, and scheduler initialized.")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # --- 6.4 Training Loop ---
    print("\n--- 6.4 Starting Training Loop ---")
    train_losses, val_losses, bleu_scores_history = [], [], []
    best_bleu4, epochs_no_improve = 0.0, 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, vocab, grad_clip_norm, epoch, num_epochs)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1} - Average Training Loss: {train_loss:.4f}")
        if ((epoch+1) % 5 == 0):
            val_loss, current_bleu_scores = evaluate_model(model, val_loader, criterion, vocab, epoch, num_epochs)
            val_losses.append(val_loss)
            bleu_scores_history.append(current_bleu_scores)
            
            current_bleu4 = current_bleu_scores['BLEU-4']
            lr_scheduler.step(current_bleu4)
    
            if current_bleu4 > best_bleu4:
                print(f"BLEU-4 improved from {best_bleu4:.4f} to {current_bleu4:.4f}. Saving model...")
                best_bleu4 = current_bleu4
                # Save model state_dict correctly whether DataParallel or not
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(model_state, MODEL_SAVE_PATH)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"BLEU-4 did not improve. Best: {best_bleu4:.4f}. No improvement epochs: {epochs_no_improve}/{patience_early_stop}")
    
            if epochs_no_improve >= patience_early_stop:
                print(f"Early stopping after {patience_early_stop} epochs."); break
            
    print("\nTraining finished.")

    # --- 6.5 Plot Metrics ---
    if train_losses and val_losses and bleu_scores_history:
        print("\n--- 6.5 Plotting Metrics ---")
        # plot_metrics(train_losses, val_losses, bleu_scores_history)
    else:
        print("\n--- 6.5 Skipping Plotting: No metrics recorded. ---")


    # --- 6.6 Example Usage ---
    print("\n--- 6.6 Example Usage (Generating caption for a sample validation image) ---")
    if os.path.exists(MODEL_SAVE_PATH) and len(val_dataset) > 0:
        encoder_final = EncoderCNN().to(DEVICE)
        decoder_final = TransformerDecoderModel(len(vocab), d_model, nhead, num_transformer_decoder_layers, 
                                                dim_feedforward, encoder_dim, transformer_dropout, max_caption_length).to(DEVICE)
        final_model = ImageCaptioningModel(encoder_final, decoder_final).to(DEVICE)
        
        # Load state dict correctly, handling potential 'module.' prefix if saved from DataParallel
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        if all(key.startswith('module.') for key in state_dict.keys()) and not isinstance(final_model, nn.DataParallel):
             # Model was saved from DataParallel, loading into non-DataParallel model
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            final_model.load_state_dict(new_state_dict)
        elif not all(key.startswith('module.') for key in state_dict.keys()) and isinstance(final_model, nn.DataParallel):
            # Model was saved from non-DataParallel, loading into DataParallel model
            final_model.module.load_state_dict(state_dict)
        else: # Covers module->module and non-module->non-module
            final_model.load_state_dict(state_dict)
        
        final_model.eval()
        
        sample_img, sample_caption_tensor, _ = val_dataset[random.randint(0, len(val_dataset)-1)]
        original_caption_text = vocab.textualize(sample_caption_tensor.tolist())
        print(f"Original Caption: {original_caption_text}")

        # Use final_model directly for generation (already handles DataParallel internally if needed)
        generated_caption = final_model.generate_caption_beam_search(sample_img, vocab, beam_width=5, max_sample_length=max_caption_length)
        print(f"Generated Caption: {generated_caption}")
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
