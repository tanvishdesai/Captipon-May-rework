import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

from PIL import Image
import nltk # For BLEU score
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import collections
import random
import math

# Step 0: Setup & Configuration
# -----------------------------------------------------------------------------
print("Step 0: Initializing Setup and Configuration...")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Hyperparameters (Adjust as needed) ---
IMAGE_DIR = "/kaggle/input/flickr8k/Images"  # <<< CHANGE THIS TO YOUR IMAGE DIRECTORY
CAPTION_FILE = "/kaggle/input/guj-captions/gujarati_captions.txt" # <<< CHANGE THIS TO YOUR CAPTION FILE
MODEL_SAVE_PATH = "./best_caption_model.pth"

embed_size = 256        # Dimension of word embeddings
attention_dim = 512     # Dimension of attention layer
encoder_dim = 2048      # Dimension of encoder output (ResNet50 last conv layer channels)
decoder_dim = 512       # Dimension of decoder LSTM hidden states
num_layers_decoder = 2  # Number of layers in LSTM decoder
dropout_p = 0.5

# Training params
num_epochs = 20
batch_size = 32
learning_rate = 1e-4
grad_clip_norm = 5.0    # Gradient clipping
vocab_threshold = 2     # Minimum word frequency to be included in vocabulary
max_caption_length = 50 # Max length for padding captions
patience_early_stop = 5 # Patience for early stopping
lr_patience = 2         # Patience for learning rate scheduler

# For reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Step 1: Vocabulary Class
# -----------------------------------------------------------------------------
print("\nStep 1: Defining Vocabulary Class...")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.word_counts = collections.Counter()

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize_gujarati(text):
        return text.lower().split(' ') # Simple space tokenizer

    def build_vocabulary(self, sentence_list):
        print("Building vocabulary...")
        idx = 4 # Start indexing from 4
        for sentence in tqdm(sentence_list, desc="Counting words"):
            for word in self.tokenize_gujarati(sentence):
                self.word_counts[word] += 1

        for word, count in tqdm(self.word_counts.items(), desc="Creating vocab mapping"):
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        print(f"Vocabulary built. Size: {len(self)}")

    def numericalize(self, text):
        tokenized_text = self.tokenize_gujarati(text)
        numericalized = [self.stoi["<SOS>"]]
        numericalized.extend([self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text])
        numericalized.append(self.stoi["<EOS>"])
        return numericalized

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
                if not line:
                    continue
                try:
                    # Format: image_name.jpg#id<space>caption
                    parts = line.split('\t', 1)
                    if len(parts) != 2:
                        print(f"Warning: Skipping malformed line {line_num+1}: '{line}' (expected 2 parts after split)")
                        continue
                    
                    img_id_part, caption_text = parts
                    img_name = img_id_part.split('#')[0] # Get base image name
                    
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
        print(f"Error: Caption file not found at {filepath}")
        raise
    except Exception as e:
        print(f"An error occurred while loading captions: {e}")
        raise

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_dict, image_names, vocab, transform, max_len):
        self.root_dir = root_dir
        self.df = [] # Store (img_path, caption_text) pairs
        for img_name in image_names:
            if img_name in captions_dict:
                for caption in captions_dict[img_name]:
                    self.df.append((img_name, caption))
            else:
                print(f"Warning: Image {img_name} found in split list but not in captions_dict.")

        self.vocab = vocab
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
            # Return a dummy image and caption to avoid crashing DataLoader
            # This should ideally be handled by ensuring all files exist before training
            dummy_image = Image.new('RGB', (256, 256), color = 'red')
            image = self.transform(dummy_image)
            caption_vec = torch.zeros(self.max_len, dtype=torch.long)
            caption_vec[0] = self.vocab.stoi["<SOS>"]
            caption_vec[1] = self.vocab.stoi["<EOS>"]
            return image, caption_vec, torch.tensor(2) # length 2 for SOS, EOS

        image = self.transform(image)
        
        numericalized_caption = self.vocab.numericalize(caption_text)
        caption_len = len(numericalized_caption)
        
        # Pad caption
        padded_caption = torch.full((self.max_len,), self.vocab.stoi["<PAD>"], dtype=torch.long)
        if caption_len > self.max_len:
            padded_caption[:] = torch.tensor(numericalized_caption[:self.max_len], dtype=torch.long)
            caption_len = self.max_len
        else:
            padded_caption[:caption_len] = torch.tensor(numericalized_caption, dtype=torch.long)
            
        return image, padded_caption, torch.tensor(caption_len, dtype=torch.long)


# Step 3: Model Architecture
# -----------------------------------------------------------------------------
print("\nStep 3: Defining Model Architecture (Encoder, Attention, Decoder)...")

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2] # Remove final avgpool and fc layer
        self.resnet = nn.Sequential(*modules)
        # The output of self.resnet will be (batch_size, encoder_dim, H/32, W/32)
        # For ResNet50, encoder_dim is 2048. H/32, W/32 usually 7x7 or 8x8 for 224/256 input.

    def forward(self, images):
        features = self.resnet(images)  # (batch_size, encoder_dim, H', W')
        batch_size = features.size(0)
        num_channels = features.size(1) # encoder_dim
        # Permute and flatten for attention: (batch_size, H'*W', encoder_dim)
        features = features.permute(0, 2, 3, 1) 
        features = features.view(batch_size, -1, num_channels)
        return features

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (batch_size, num_pixels, encoder_dim)
        # decoder_hidden: (batch_size, decoder_dim) - this is h_{t-1} from LSTM's last layer
        
        att1 = self.encoder_att(encoder_out) # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1) # (batch_size, 1, attention_dim)
        
        # Additive attention
        # att1 + att2 will broadcast att2 across num_pixels
        # (batch_size, num_pixels, attention_dim)
        combined_att = self.relu(att1 + att2) 
        
        # (batch_size, num_pixels, 1)
        e = self.full_att(combined_att) 
        
        # (batch_size, num_pixels)
        alpha = self.softmax(e.squeeze(2)) 
        
        # (batch_size, encoder_dim)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        
        return attention_weighted_encoding, alpha

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, decoder_dim, vocab_size, encoder_dim, num_layers, dropout_p):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        
        # LSTM input: concatenated embedding and attention_weighted_encoding
        self.lstm = nn.LSTM(embed_size + encoder_dim, decoder_dim, num_layers, 
                            batch_first=True, dropout=dropout_p if num_layers > 1 else 0)
        
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_p)

        # Layers to initialize LSTM hidden/cell states from mean encoder output
        self.init_h = nn.Linear(encoder_dim, decoder_dim) # For one layer LSTM
        self.init_c = nn.Linear(encoder_dim, decoder_dim) # For one layer LSTM
        # If num_layers > 1, you might need to adjust init_h/c or how they are expanded.
        # For simplicity, we'll make them produce (num_layers, batch_size, decoder_dim)

    def init_hidden_state(self, encoder_out):
        # encoder_out: (batch_size, num_pixels, encoder_dim)
        mean_encoder_out = encoder_out.mean(dim=1) # (batch_size, encoder_dim)
        
        # Initialize hidden state for each layer
        # For a single layer LSTM:
        h0 = self.init_h(mean_encoder_out) # (batch_size, decoder_dim)
        c0 = self.init_c(mean_encoder_out) # (batch_size, decoder_dim)

        # Reshape for LSTM: (num_layers, batch_size, decoder_dim)
        h = h0.unsqueeze(0).repeat(self.num_layers, 1, 1) 
        c = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h, c

    def forward(self, encoder_out, captions, caption_lengths):
        # encoder_out: (batch_size, num_pixels, encoder_dim)
        # captions: (batch_size, max_caption_length)
        # caption_lengths: (batch_size) - actual lengths, not used explicitly here with fixed loop
        
        batch_size = encoder_out.size(0)
        
        embeddings = self.embedding(captions) # (batch_size, max_caption_length, embed_size)
        
        h, c = self.init_hidden_state(encoder_out) # (num_layers, batch_size, decoder_dim)
        
        # We don't feed <EOS> into LSTM, and predict up to max_caption_length-1 words
        decode_len = captions.size(1) - 1
        
        predictions = torch.zeros(batch_size, decode_len, self.vocab_size).to(DEVICE)
        alphas = torch.zeros(batch_size, decode_len, encoder_out.size(1)).to(DEVICE) # num_pixels

        for t in range(decode_len):
            current_embedding_t = embeddings[:, t, :] # (batch_size, embed_size)
            
            # h[-1] gives the hidden state of the last LSTM layer (if multi-layer)
            # For single layer LSTM, h[0] or h.squeeze(0)
            # h.shape is (num_layers, batch_size, decoder_dim), so h[-1] is (batch_size, decoder_dim)
            attention_weighted_encoding, alpha_t = self.attention(encoder_out, h[-1])
            
            # Concatenate embedding and context vector
            lstm_input_t = torch.cat((current_embedding_t, attention_weighted_encoding), dim=1) # (batch_size, embed_size + encoder_dim)
            
            # LSTM expects input of (batch_size, 1, input_size) if batch_first=True and processing one step
            # or (1, batch_size, input_size) if batch_first=False
            # Or, use LSTMCell
            # Here, nn.LSTM can take (batch_size, seq_len, input_size), so unsqueeze(1) for seq_len=1
            lstm_output_t, (h, c) = self.lstm(lstm_input_t.unsqueeze(1), (h, c))
            # lstm_output_t is (batch_size, 1, decoder_dim)
            
            preds_t = self.fc(self.dropout(lstm_output_t.squeeze(1))) # (batch_size, vocab_size)
            
            predictions[:, t, :] = preds_t
            alphas[:, t, :] = alpha_t
            
        return predictions, alphas

    def sample(self, encoder_out, vocab, max_sample_length=20):
        # Used for inference/evaluation
        batch_size = encoder_out.size(0)
        assert batch_size == 1, "Sampling currently supports batch_size=1"

        h, c = self.init_hidden_state(encoder_out) # (num_layers, 1, decoder_dim)
        
        # Start with <SOS> token
        current_word_idx = torch.tensor([vocab.stoi["<SOS>"]]).to(DEVICE) # (1)
        
        sampled_ids = []
        
        for _ in range(max_sample_length):
            current_embedding = self.embedding(current_word_idx) # (1, embed_size)
            
            attention_weighted_encoding, _ = self.attention(encoder_out, h[-1]) # (1, encoder_dim)
            
            lstm_input = torch.cat((current_embedding, attention_weighted_encoding), dim=1) # (1, embed_size + encoder_dim)
            
            lstm_output, (h, c) = self.lstm(lstm_input.unsqueeze(1), (h, c)) # (1, 1, decoder_dim)
            
            preds = self.fc(lstm_output.squeeze(1)) # (1, vocab_size)
            
            # Greedy decoding
            predicted_idx = preds.argmax(1) # (1)
            
            if predicted_idx.item() == vocab.stoi["<EOS>"]:
                break
            
            sampled_ids.append(predicted_idx.item())
            current_word_idx = predicted_idx
            
        return sampled_ids
        
    def sample_beam_search(self, encoder_out, vocab, beam_width=3, max_sample_length=50):
        # encoder_out: (1, num_pixels, encoder_dim) - ensure batch_size is 1 for this
        k = beam_width
        vocab_size = self.vocab_size

        # Tensor to store top k previous words at each step; now they are just <SOS>
        k_prev_words = torch.full((k, 1), vocab.stoi["<SOS>"], dtype=torch.long).to(DEVICE)  # (k, 1)

        # Tensor to store top k sequences; now they are just <SOS>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they are just 0
        top_k_scores = torch.zeros(k, 1).to(DEVICE)  # (k, 1)

        # Lists to store completed sequences and their scores
        complete_seqs = []
        complete_seqs_scores = []

        # Start decoding
        # Initialize hidden and cell states from encoder_out
        # Note: init_hidden_state expects batch_size in encoder_out, so we might need to tile or adjust
        # For beam search, we need to manage k independent LSTM states.
        
        # Tile encoder_out and initialize h, c for k beams
        encoder_out_k = encoder_out.expand(k, -1, -1) # (k, num_pixels, encoder_dim)
        h, c = self.init_hidden_state(encoder_out_k) # (num_layers, k, decoder_dim)
        
        step = 1
        while True:
            embeddings = self.embedding(k_prev_words).squeeze(1) # (k, embed_size)
            
            # h_last_layer for attention should be (k, decoder_dim)
            att_weighted_encoding, _ = self.attention(encoder_out_k, h[-1]) # (k, encoder_dim)
            
            lstm_input = torch.cat([embeddings, att_weighted_encoding], dim=1) # (k, embed_size + encoder_dim)
            
            # LSTM expects input (k, 1, input_size) and h/c (num_layers, k, decoder_dim)
            # lstm_output will be (k, 1, decoder_dim)
            lstm_output, (h, c) = self.lstm(lstm_input.unsqueeze(1), (h, c))
            
            preds = self.fc(lstm_output.squeeze(1))  # (k, vocab_size)
            log_probs = F.log_softmax(preds, dim=1)  # (k, vocab_size)

            # Add current scores
            # top_k_scores has scores of sequences ending at previous step
            # log_probs has scores for current step words
            log_probs = top_k_scores.expand_as(log_probs) + log_probs # (k, vocab_size)

            # For the first step, all k sequences will have the same score (0)
            # and we pick k best words from vocab_size
            if step == 1:
                top_k_scores, top_k_words = log_probs[0].topk(k, 0, True, True)  # (k)
            else:
                # Unroll and find top k scores and their details
                top_k_scores, top_k_words = log_probs.view(-1).topk(k, 0, True, True)  # (k)

            # Convert unrolled indices to actual indices of scores, words
            prev_seq_inds = torch.div(top_k_words, vocab_size, rounding_mode='floor') # (k) which beam current word came from
            next_word_inds = top_k_words % vocab_size  # (k) actual word index

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_seq_inds], next_word_inds.unsqueeze(1)], dim=1)  # (k, step+1)

            # Which sequences are incomplete (not ending with <EOS>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds)
                               if next_word.item() != vocab.stoi["<EOS>"]]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            
            seqs = seqs[incomplete_inds]
            h = h[:, prev_seq_inds[incomplete_inds]] # (num_layers, k_new, decoder_dim)
            c = c[:, prev_seq_inds[incomplete_inds]] # (num_layers, k_new, decoder_dim)
            encoder_out_k = encoder_out_k[prev_seq_inds[incomplete_inds]] # (k_new, num_pixels, encoder_dim)
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1) # (k_new, 1)
            if step > max_sample_length:
                if len(seqs) > 0: 
                    complete_seqs.extend(seqs.tolist())
                    squeezed_scores = top_k_scores.squeeze() # top_k_scores is (k_remaining, 1) here
                    if squeezed_scores.ndim == 0: 
                        complete_seqs_scores.append(squeezed_scores.item()) 
                    else: 
                        complete_seqs_scores.extend(squeezed_scores.tolist())
                break
            step += 1

        if not complete_seqs: # Handle case where no sequence reaches EOS
            # This can happen if max_sample_length is too short or model is stuck
            # Fallback: return the best sequence found so far (if any) or an empty list
            if len(seqs) > 0: # If beam search ended due to max_length with incomplete seqs
                 best_incomplete_seq_idx = top_k_scores.argmax().item()
                 best_seq_ids = seqs[best_incomplete_seq_idx].tolist()
                 # Remove SOS if present, and any PADs that might have been added by mistake
                 best_seq_ids = [idx for idx in best_seq_ids if idx != vocab.stoi["<SOS>"] and idx != vocab.stoi["<PAD>"]]
                 return best_seq_ids
            return []


        i = complete_seqs_scores.index(max(complete_seqs_scores))
        best_seq_ids = complete_seqs[i]
        
        # Remove SOS, EOS, PAD if they are part of the raw list
        sampled_ids = [idx for idx in best_seq_ids 
                       if idx != vocab.stoi["<SOS>"] and \
                          idx != vocab.stoi["<EOS>"] and \
                          idx != vocab.stoi["<PAD>"]]
        return sampled_ids

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, caption_lengths):
        encoder_out = self.encoder(images)
        outputs, alphas = self.decoder(encoder_out, captions, caption_lengths)
        return outputs, alphas

    def generate_caption_beam_search(self, image, vocab, beam_width=3, max_sample_length=50):
        self.eval() # Set to evaluation mode
        with torch.no_grad():
            if image.dim() == 3: # If single image (C, H, W)
                image = image.unsqueeze(0) # Add batch dimension (1, C, H, W)
            image = image.to(DEVICE)
            
            encoder_out = self.encoder(image) # (1, num_pixels, encoder_dim)
            
            # Call the new beam search sample method in the decoder
            sampled_ids = self.decoder.sample_beam_search(encoder_out, vocab, beam_width, max_sample_length)
            
            # Convert indices to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.itos.get(word_id)
                # No need to check for SOS/EOS/PAD here if sample_beam_search already filters them
                if word: 
                    sampled_caption.append(word)
            return " ".join(sampled_caption)


# Step 4: Training and Evaluation Utilities
# -----------------------------------------------------------------------------
print("\nStep 4: Defining Training and Evaluation Utilities...")

def train_one_epoch(model, train_loader, optimizer, criterion, vocab, grad_clip_norm, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", unit="batch")
    for i, (images, captions, lengths) in enumerate(progress_bar):
        images = images.to(DEVICE)
        captions = captions.to(DEVICE) # (batch_size, max_caption_length)
        # Target captions for loss: remove <SOS> token at the beginning
        # Predictions are for words from index 1 to end
        targets = captions[:, 1:] # (batch_size, max_caption_length - 1)

        optimizer.zero_grad()
        
        # For teacher forcing, input to decoder is captions[:, :-1] (excluding <EOS>)
        # The decoder's forward method already handles this by iterating up to decode_len
        outputs, _ = model(images, captions, lengths) # outputs: (batch_size, decode_len, vocab_size)
        
        # Reshape for CrossEntropyLoss:
        # Outputs: (batch_size * (max_caption_length-1), vocab_size)
        # Targets: (batch_size * (max_caption_length-1))
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
        
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
    references_corpus = [] # List of lists of reference tokens
    hypotheses_corpus = [] # List of hypothesis tokens
    
    print(f"Epoch {epoch+1}/{num_epochs} [Validation] - Generating captions and calculating BLEU...")
    with torch.no_grad():
        for images, captions_batch, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", unit="batch"):
            images = images.to(DEVICE)
            captions_batch = captions_batch.to(DEVICE)
            targets_batch = captions_batch[:, 1:]

            outputs, _ = model(images, captions_batch, None) # lengths not strictly needed for loss calc if iterating fixed len
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets_batch.reshape(-1))
            total_loss += loss.item()

            # For BLEU score calculation (on a subset or all, be mindful of time)
            # Here we process each image in the batch individually for caption generation
            for i in range(images.size(0)):
                image_single = images[i].unsqueeze(0) # (1, C, H, W)
                
                # Ground truth captions for this image
                # For Flickr8k, each image might have multiple reference captions.
                # Here, val_loader gives one caption per image. For proper BLEU, you'd need all.
                # This example will use the single caption from batch as reference.
                # A better approach for val_loader might be to group all captions for an image.
                # For simplicity, we use the current caption as the single reference.
                
                ref_caption_indices = captions_batch[i].tolist()
                ref_tokens = [vocab.itos[idx] for idx in ref_caption_indices 
                              if vocab.itos[idx] not in ["<SOS>", "<EOS>", "<PAD>"]]
                references_corpus.append([ref_tokens]) # NLTK expects list of lists of tokens

                # Generated caption
                # In evaluate_model
                generated_caption_text = model.module.generate_caption_beam_search(image_single, vocab, beam_width=3, max_sample_length=max_caption_length) \
                                         if isinstance(model, nn.DataParallel) else \
                                         model.generate_caption_beam_search(image_single, vocab, beam_width=3, max_sample_length=max_caption_length)
                # Choose a beam_width, e.g., 3 or 5. max_sample_length is fine.
                hyp_tokens = Vocabulary.tokenize_gujarati(generated_caption_text)
                hypotheses_corpus.append(hyp_tokens)
                # Add this for debugging a few examples per validation epoch:
                if i < 3: # Print first 3 examples from a batch
                    print(f"  Sample Eval {i}:")
                    print(f"    Ref (indices): {captions_batch[i].tolist()}")
                    print(f"    Ref (tokens): {' '.join(ref_tokens)}")
                    print(f"    Hyp (tokens): {' '.join(hyp_tokens)}")
                    print(f"    Hyp (raw): {generated_caption_text}")
    avg_val_loss = total_loss / len(val_loader)
    
    bleu_scores = {}
    for i in range(1, 5):
        bleu_scores[f'BLEU-{i}'] = corpus_bleu(references_corpus, hypotheses_corpus, weights=tuple(1/i for _ in range(i)))
    
    print(f"Validation Results - Epoch {epoch+1}: Avg Loss: {avg_val_loss:.4f}")
    for name, score in bleu_scores.items():
        print(f"{name}: {score:.4f}")
        
    return avg_val_loss, bleu_scores


# Step 5: Plotting Utilities
# -----------------------------------------------------------------------------
print("\nStep 5: Defining Plotting Utilities...")

def plot_metrics(train_losses, val_losses, bleu_scores_history):
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot BLEU Scores
    plt.subplot(1, 2, 2)
    for i in range(1, 5):
        bleu_i_scores = [scores[f'BLEU-{i}'] for scores in bleu_scores_history]
        plt.plot(epochs_range, bleu_i_scores, marker='o', linestyle='-', label=f'BLEU-{i}')
    plt.title('Validation BLEU Scores')
    plt.xlabel('Epochs')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()
    print("Metrics plot saved as training_metrics.png")

# Step 6: Main Execution Block
# -----------------------------------------------------------------------------
def main():
    print("\nStep 6: Starting Main Execution...")

    # --- 6.1 Load Data and Build Vocabulary ---
    print("\n--- 6.1 Loading Data and Building Vocabulary ---")
    captions_data, all_captions_for_vocab = load_captions(CAPTION_FILE)
    
    vocab = Vocabulary(vocab_threshold)
    vocab.build_vocabulary(all_captions_for_vocab)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Stoi for <UNK>: {vocab.stoi['<UNK>']}")
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Or (224,224) if ResNet expects that
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 6.2 Split Data ---
    print("\n--- 6.2 Splitting Data ---")
    unique_image_names = list(captions_data.keys())
    random.shuffle(unique_image_names) # Shuffle for splitting
    
    # Check if IMAGE_DIR exists and contains images
    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: IMAGE_DIR '{IMAGE_DIR}' not found or not a directory.")
        print("Please ensure IMAGE_DIR is correctly set and contains your image files.")
        return
    
    # Filter unique_image_names to only include those present in IMAGE_DIR
    print(f"Verifying images in IMAGE_DIR: {IMAGE_DIR}...")
    verified_image_names = []
    for img_name in tqdm(unique_image_names, desc="Verifying images"):
        if os.path.exists(os.path.join(IMAGE_DIR, img_name)):
            verified_image_names.append(img_name)
        else:
            print(f"Warning: Image file {img_name} listed in captions but not found in {IMAGE_DIR}. Skipping.")
    
    if not verified_image_names:
        print("Error: No valid image files found based on captions and IMAGE_DIR. Aborting.")
        return
    
    print(f"Found {len(verified_image_names)} images common to captions and directory.")
    unique_image_names = verified_image_names


    # Split: 80% train, 20% validation (adjust as needed)
    # For Flickr8k, standard splits are often used (train, dev, test files).
    # If you have those, load them instead of this random split.
    # This is a simple random split of unique image names.
    split_idx = int(0.8 * len(unique_image_names))
    train_image_names = unique_image_names[:split_idx]
    val_image_names = unique_image_names[split_idx:]
    
    print(f"Number of unique images for training: {len(train_image_names)}")
    print(f"Number of unique images for validation: {len(val_image_names)}")

    train_dataset = FlickrDataset(IMAGE_DIR, captions_data, train_image_names, vocab, transform, max_caption_length)
    val_dataset = FlickrDataset(IMAGE_DIR, captions_data, val_image_names, vocab, transform, max_caption_length)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Training or validation dataset is empty. Check data paths and content.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train loader: {len(train_loader)} batches. Validation loader: {len(val_loader)} batches.")

    # --- 6.3 Initialize Model, Optimizer, Criterion, Scheduler ---
    print("\n--- 6.3 Initializing Model, Optimizer, Criterion, Scheduler ---")
    encoder = EncoderCNN().to(DEVICE)
    decoder = DecoderRNN(embed_size, decoder_dim, len(vocab), encoder_dim, num_layers_decoder, dropout_p).to(DEVICE)
    model = ImageCaptioningModel(encoder, decoder).to(DEVICE)

    # If multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Dynamic learning rate: reduce on plateau of validation BLEU-4 score
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=lr_patience, verbose=True)
    
    print("Model, optimizer, criterion, and scheduler initialized.")
    print(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")


    # --- 6.4 Training Loop ---
    print("\n--- 6.4 Starting Training Loop ---")
    train_losses = []
    val_losses = []
    bleu_scores_history = []
    
    best_bleu4 = 0.0
    epochs_no_improve = 0 # For early stopping

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, vocab, grad_clip_norm, epoch, num_epochs)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1} - Average Training Loss: {train_loss:.4f}")
        
        val_loss, current_bleu_scores = evaluate_model(model, val_loader, criterion, vocab, epoch, num_epochs)
        val_losses.append(val_loss)
        bleu_scores_history.append(current_bleu_scores)
        
        current_bleu4 = current_bleu_scores['BLEU-4']
        lr_scheduler.step(current_bleu4) # Step scheduler based on BLEU-4

        # Save best model
        if current_bleu4 > best_bleu4:
            print(f"BLEU-4 improved from {best_bleu4:.4f} to {current_bleu4:.4f}. Saving model...")
            best_bleu4 = current_bleu4
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"BLEU-4 did not improve. Best BLEU-4: {best_bleu4:.4f}. Epochs without improvement: {epochs_no_improve}/{patience_early_stop}")

        if epochs_no_improve >= patience_early_stop:
            print(f"Early stopping triggered after {patience_early_stop} epochs without improvement.")
            break
            
    print("\nTraining finished.")

    # --- 6.5 Plot Metrics ---
    print("\n--- 6.5 Plotting Metrics ---")
    plot_metrics(train_losses, val_losses, bleu_scores_history)

    # --- 6.6 Example Usage (Optional: Load best model and generate caption for a sample image) ---
    print("\n--- 6.6 Example Usage (Generating caption for a sample validation image) ---")
    if os.path.exists(MODEL_SAVE_PATH) and len(val_dataset) > 0 :
        # Re-initialize model structure
        encoder_final = EncoderCNN().to(DEVICE)
        decoder_final = DecoderRNN(embed_size, decoder_dim, len(vocab), encoder_dim, num_layers_decoder, dropout_p).to(DEVICE)
        final_model = ImageCaptioningModel(encoder_final, decoder_final).to(DEVICE)
        
        # Handle DataParallel if it was used
        if torch.cuda.device_count() > 1 and not isinstance(final_model, nn.DataParallel):
             # If saved model was DataParallel, keys will have 'module.' prefix
            state_dict = torch.load(MODEL_SAVE_PATH)
            if all(key.startswith('module.') for key in state_dict.keys()):
                final_model = nn.DataParallel(final_model) # Wrap model first
                final_model.load_state_dict(state_dict)
            else: # Saved model was not DataParallel but current setup is
                # This case is tricky, usually one saves model.module.state_dict()
                # For simplicity, assuming if multi-GPU, saved model also was.
                final_model.load_state_dict(state_dict)

        elif not (torch.cuda.device_count() > 1) and isinstance(model, nn.DataParallel):
            # Saved model was DataParallel, but loading on single GPU/CPU
            state_dict = torch.load(MODEL_SAVE_PATH)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            final_model.load_state_dict(new_state_dict)
        else: # Single GPU or CPU for both training and loading
             final_model.load_state_dict(torch.load(MODEL_SAVE_PATH))


        final_model.eval()
        
        # Get a sample image from validation set
        sample_img, sample_caption_tensor, _ = val_dataset[random.randint(0, len(val_dataset)-1)]
        
        # Display original image (optional, might not work in all Kaggle notebook environments directly)
        # try:
        #     img_display = transforms.ToPILImage()(sample_img)
        #     plt.imshow(img_display)
        #     plt.title("Sample Image for Captioning")
        #     plt.axis('off')
        #     plt.show()
        # except Exception as e:
        #     print(f"Could not display sample image: {e}")

        # Original caption
        original_caption_tokens = [vocab.itos[idx] for idx in sample_caption_tensor.tolist() 
                                   if vocab.itos[idx] not in ["<SOS>", "<EOS>", "<PAD>"]]
        print(f"Original Caption: {' '.join(original_caption_tokens)}")

        # Generated caption
        generated_caption = final_model.module.generate_caption_beam_search(sample_img, vocab) \
                            if isinstance(final_model, nn.DataParallel) else \
                            final_model.generate_caption_beam_search(sample_img, vocab)
        print(f"Generated Caption: {generated_caption}")
    else:
        print("Skipping example usage: Best model not saved or validation set empty.")

    print("\nScript finished.")

if __name__ == "__main__":
    # Ensure NLTK data is available (run this once if needed)
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' not found. Downloading...")
        nltk.download('punkt', quiet=True)
    
    main()