Okay, this is a great set of experiments for image captioning! I'll provide detailed documentation covering each trial, highlighting the changes, the model architecture, training specifics, and interpreting the results.

---

## Project Documentation: Gujarati Image Captioning Experiments

**Overall Project Goal:** To develop and evaluate different deep learning models for generating Gujarati captions for images from the Flickr8k dataset. The experiments explore variations in vocabulary construction, model architecture (LSTM vs. Transformer), hyperparameter tuning, and data augmentation.

**Dataset:**
*   **Images:** Flickr8k dataset, `/kaggle/input/flickr8k/Images`
*   **Captions:** Custom Gujarati captions, `/kaggle/input/guj-captions/gujarati_captions.txt`
    *   Format: `image_name.jpg#id\tcaption_text`
*   **Splits:** 80% unique images for training, 20% for validation. Images are shuffled before splitting.

**Common Components Across Trials (unless specified otherwise):**

1.  **Image Encoder (`EncoderCNN`):**
    *   Uses a pre-trained ResNet-50 model (`models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)`).
    *   The final average pooling and fully connected layers are removed.
    *   Output: Spatial image features of shape `(batch_size, H'*W', encoder_dim)`, where `encoder_dim = 2048` (from ResNet-50) and `H', W'` are reduced dimensions (e.g., 7x7 or 8x8 for 224/256 input). These are flattened to `(batch_size, num_pixels, encoder_dim)`.

2.  **Image Preprocessing (`transform`):**
    *   Standard: `Resize((256, 256))`, `ToTensor()`, `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
    *   Trial 5 introduces specific augmentation for the training set.

3.  **Data Loading (`FlickrDataset`, `DataLoader`):**
    *   `FlickrDataset` loads image-caption pairs.
    *   Captions are numericalized and padded to `max_caption_length`.
    *   `DataLoader` is used with `batch_size = 32`, `num_workers = 2`, `pin_memory = True`. Training loader shuffles data.

4.  **Training Setup (General):**
    *   **Device:** GPU (`cuda`) if available, else CPU.
    *   **Loss Function:** `nn.CrossEntropyLoss` (with `ignore_index` for `<PAD>` tokens).
    *   **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_` with `grad_clip_norm = 5.0`.
    *   **Learning Rate Scheduler:** `optim.lr_scheduler.ReduceLROnPlateau` (mode='max', factor=0.1, patience=`lr_patience`=2, based on validation BLEU-4).
    *   **Early Stopping:** If validation BLEU-4 doesn't improve for `patience_early_stop`=5 epochs.
    *   **Metrics:** Training Loss, Validation Loss, BLEU-1 to BLEU-4 scores. BLEU scores are calculated using `nltk.translate.bleu_score.corpus_bleu`.
    *   **Model Saving:** Best model based on validation BLEU-4 is saved.

5.  **Plotting (`plot_metrics`):**
    *   Generates plots for Training/Validation Loss and Validation BLEU scores over epochs.

6.  **Seed:** `torch.manual_seed(42)`, `random.seed(42)`, `np.random.seed(42)` for reproducibility.

---

### Trial 1: Baseline LSTM-based Model

*   **Code File:** `trial 1 code.py`
*   **Objective:** Establish a baseline using a standard LSTM-based encoder-decoder architecture with Bahdanau attention and a custom vocabulary.

**1. Vocabulary (`Vocabulary` Class):**
    *   **Type:** Custom, word-based.
    *   **Tokenizer:** Simple space splitting (`text.lower().split(' ')`).
    *   **Special Tokens:** `<PAD>` (0), `<SOS>` (1), `<EOS>` (2), `<UNK>` (3).
    *   **`vocab_threshold = 2`:** Words appearing less than twice are mapped to `<UNK>`.
    *   **Resulting Size:** 8301 (from `mixed trial results.csv`).

**2. Model Architecture (`ImageCaptioningModel`):**
    *   **Encoder:** `EncoderCNN` (ResNet-50 backbone).
    *   **Decoder (`DecoderRNN`):**
        *   **Word Embeddings:** `embed_size = 256`.
        *   **Attention:** `BahdanauAttention` (`attention_dim = 512`).
            *   Calculates attention weights `alpha` over encoder features based on the previous decoder hidden state.
            *   `encoder_att = nn.Linear(encoder_dim, attention_dim)`
            *   `decoder_att = nn.Linear(decoder_dim, attention_dim)`
            *   `full_att = nn.Linear(attention_dim, 1)`
        *   **LSTM:**
            *   `decoder_dim = 512` (hidden state size).
            *   `num_layers_decoder = 1`.
            *   Input to LSTM cell at each step: Concatenation of current word embedding and attention-weighted encoding (`embed_size + encoder_dim`).
            *   LSTM initialized using mean encoder output projected by `init_h` and `init_c` linear layers.
        *   **Output Layer:** `nn.Linear(decoder_dim, vocab_size)`.
        *   **Dropout:** `dropout_p = 0.5` (applied after LSTM output, before FC layer).
    *   **Caption Generation (Evaluation):** Greedy decoding (`DecoderRNN.sample`).

**3. Training Specifics:**
    *   **`max_caption_length = 50`**.
    *   **Optimizer:** `optim.Adam(model.parameters(), lr=1e-4)`.
    *   **Epochs:** Max `num_epochs = 20`.

**4. Results (from `mixed trial results.csv` for Trial_ID 1):**
    *   **Trainable Parameters:** 39,073,198
    *   **Data Augmentation:** No
    *   **Epochs Run:** 14
    *   **Stopping Reason:** Early Stopping
    *   **Best BLEU-4:** 0.0484 (achieved at epoch 9)
    *   **Avg Train Loss (at epoch 14):** 2.4430
    *   **Avg Val Loss (at epoch 14):** 3.4892
    *   **Avg Train Time/Epoch:** 450s
    *   **Avg Val Time/Epoch:** 169s

**5. Observations:**
    *   The model starts to overfit, as validation loss plateaus and early stopping is triggered.
    *   The BLEU-4 score of 0.0484 is a decent starting point for a relatively simple LSTM model on a custom dataset.
    *   Greedy decoding is used for evaluation, which might not be optimal for BLEU scores compared to beam search.

---

### Trial 2: LSTM with Increased Depth & Beam Search

*   **Code File:** `trial 2 code.py`
*   **Objective:** Investigate the impact of a deeper LSTM decoder and using beam search for caption generation during evaluation. Also, minor optimizer change.

**1. Key Configuration Changes from Trial 1:**
    *   **Decoder (`DecoderRNN`):**
        *   `num_layers_decoder = 2` (increased from 1). Dropout is now applied between LSTM layers if `num_layers > 1`.
    *   **Caption Generation (Evaluation):** Beam search (`DecoderRNN.sample_beam_search` with `beam_width=3`).
    *   **Optimizer:** `optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)`. (Added `weight_decay`).

**2. Vocabulary:**
    *   Same as Trial 1 (Custom, size 8301).

**3. Model Architecture:**
    *   Same core LSTM-based encoder-decoder with Bahdanau attention.
    *   The LSTM in `DecoderRNN` now has 2 layers.
    *   `DecoderRNN.sample_beam_search` implementation:
        *   Maintains `k` (beam_width) candidate sequences.
        *   At each step, expands each of the `k` sequences with all possible next words.
        *   Calculates scores (log probabilities) for all expanded sequences.
        *   Selects the top `k` sequences from these.
        *   Handles `<EOS>` token to mark completed sequences.
        *   Returns the highest-scoring completed sequence.

**4. Training Specifics:**
    *   Same as Trial 1 (`max_caption_length = 50`, `num_epochs = 20`, Adam optimizer with LR 1e-4).

**5. Results (from `mixed trial results.csv` for Trial_ID 2):**
    *   **Trainable Parameters:** 41,174,446 (increased due to 2-layer LSTM)
    *   **Data Augmentation:** No
    *   **Epochs Run:** 20
    *   **Stopping Reason:** Finished all epochs
    *   **Best BLEU-4:** 0.0547 (achieved at epoch 20)
    *   **Avg Train Loss (at epoch 20):** 2.3488
    *   **Avg Val Loss (at epoch 20):** 3.4374
    *   **Avg Train Time/Epoch:** 472s
    *   **Avg Val Time/Epoch:** 227s (increased due to beam search during validation)

**6. Observations:**
    *   Increasing LSTM depth and using beam search slightly improved the BLEU-4 score to 0.0547.
    *   The model trained for all 20 epochs, suggesting that early stopping (based on 5 epochs patience) might not have been triggered, or the improvements were marginal but consistent enough.
    *   The increase in parameters and validation time is expected.
    *   The addition of `weight_decay` to the optimizer is a form of L2 regularization, which can help prevent overfitting.

---

### Trial 3: Transformer-based Model with SentencePiece Vocabulary

*   **Code File:** `trial 3 code.py`
*   **Objective:** Transition to a Transformer-based decoder and utilize a subword vocabulary (SentencePiece) to potentially handle out-of-vocabulary words better and capture richer semantic units.

**1. Key Configuration Changes from Trial 2:**
    *   **Vocabulary (`SentencePieceVocabulary`):**
        *   **Type:** SentencePiece (BPE model).
        *   **`SP_MODEL_PREFIX = "gujarati_caption_sp"`**
        *   **`SP_VOCAB_SIZE = 8000`** (target vocabulary size).
        *   Special Tokens: `<pad>` (0), `<unk>` (1), `<sos>` (2), `<eos>` (3), managed by SentencePiece during training (`pad_id`, `unk_id`, `bos_id`, `eos_id` and corresponding `_piece` arguments).
        *   `tokenize_for_bleu`: Uses `sp_model.encode_as_pieces()` for BLEU evaluation, aligning tokenization with the model's vocabulary.
    *   **Model Architecture (Replaced `DecoderRNN` with `TransformerDecoderModel`):**
        *   **Encoder:** `EncoderCNN` (ResNet-50 backbone). Output features are projected by `encoder_projection = nn.Linear(encoder_feature_dim, d_model)` before being fed to the decoder.
        *   **Decoder (`TransformerDecoderModel`):**
            *   `d_model = 512` (embedding dimension and main model dimension).
            *   `nhead = 8` (number of attention heads).
            *   `num_transformer_decoder_layers = 6`.
            *   `dim_feedforward = 2048` (dimension of FFN inside Transformer layers).
            *   `transformer_dropout = 0.1`.
            *   **Embeddings:** `nn.Embedding(vocab_size, d_model)` followed by `PositionalEncoding`.
            *   **Core:** `nn.TransformerDecoder` built from `nn.TransformerDecoderLayer` (which includes self-attention, multi-head cross-attention with encoder output, and FFN).
            *   **Masks:**
                *   `_generate_square_subsequent_mask`: For self-attention in the decoder to prevent attending to future tokens.
                *   `_create_padding_mask`: To ignore `<PAD>` tokens in the target sequence.
            *   **Output Layer:** `nn.Linear(d_model, vocab_size)`.
        *   **Caption Generation (Evaluation):** Beam search (`TransformerDecoderModel.sample_beam_search` with `beam_width=3` or `5` as per code, CSV shows `beam_width=3`).
    *   **Training Specifics:**
        *   **`max_caption_length = 75`**.
        *   **Optimizer:** `optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)`. (Transformer-specific Adam params).
        *   **Evaluation Frequency:** `evaluate_model` is called only every 5 epochs. This explains missing validation metrics for some epochs in the CSV.

**2. Results (from `mixed trial results.csv` for Trial_ID 3):**
    *   **Trainable Parameters:** 57,981,312
    *   **Data Augmentation:** No
    *   **Epochs Run:** 20
    *   **Stopping Reason:** Finished all epochs
    *   **Best BLEU-4:** 0.0717 (achieved at epoch 5)
    *   **Avg Train Loss (at epoch 20):** 0.8859
    *   **Avg Val Loss (at epoch 20, from epoch 20 eval):** 3.7497
    *   **Avg Train Time/Epoch:** 311s
    *   **Avg Val Time/Epoch:** 3339s (significantly higher due to Transformer complexity and less frequent but full validation runs)

**3. Observations:**
    *   The switch to SentencePiece and a Transformer decoder significantly improved BLEU-4 to 0.0717.
    *   The best BLEU-4 was achieved early (epoch 5), and subsequent training led to much lower training loss but higher validation loss, indicating overfitting despite the Transformer's regularization capabilities (dropout). The LR scheduler might not have kicked in effectively if BLEU-4 peaked early.
    *   Trainable parameters increased substantially.
    *   The much longer validation time per epoch is due to evaluating every 5 epochs (so each validation reported is actually a full pass) and the inherent complexity of the Transformer model.
    *   SentencePiece likely helps handle the Gujarati script's morphology better than simple word splitting.

---

### Trial 4: Larger Transformer Model & AdamW

*   **Code File:** `trial 4 code.py`
*   **Objective:** Explore if a larger Transformer model can yield further improvements and switch to the AdamW optimizer with label smoothing.

**1. Key Configuration Changes from Trial 3:**
    *   **Model Architecture (`TransformerDecoderModel`):**
        *   `d_model = 768` (increased from 512).
        *   `num_transformer_decoder_layers = 8` (increased from 6).
        *   `dim_feedforward = 3072` (increased from 2048).
        *   `transformer_dropout = 0.15` (increased from 0.1).
    *   **Training Specifics:**
        *   **`max_caption_length = 25`** (significantly reduced from 75). This is a crucial change.
        *   **Loss Function:** `nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, label_smoothing=0.1)`. (Added `label_smoothing`).
        *   **Optimizer:** `optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)`. (Switched to AdamW, increased `weight_decay`).

**2. Vocabulary:**
    *   Same as Trial 3 (SentencePiece, target size 8000).

**3. Results (from `mixed trial results.csv` for Trial_ID 4):**
    *   **Trainable Parameters:** 112,991,872 (significantly increased)
    *   **Data Augmentation:** No
    *   **Epochs Run:** 20
    *   **Stopping Reason:** Finished all epochs
    *   **Best BLEU-4:** 0.0713 (achieved at epoch 5)
    *   **Avg Train Loss (at epoch 20):** 2.0304
    *   **Avg Val Loss (at epoch 20, from epoch 20 eval):** 4.3514
    *   **Avg Train Time/Epoch:** 321s
    *   **Avg Val Time/Epoch:** 1738s

**4. Observations:**
    *   Despite a much larger model, the BLEU-4 score (0.0713) did not improve over Trial 3 (0.0717) and was also achieved at epoch 5.
    *   The significantly reduced `max_caption_length = 25` might be a bottleneck, preventing the model from generating longer, more descriptive captions, or it might be that Gujarati captions for Flickr8k are often short. If captions are typically longer, this truncation would hurt performance.
    *   Label smoothing and AdamW are good practices for training Transformers, but their benefits weren't apparent here, possibly overshadowed by other factors like `max_caption_length` or the dataset's characteristics.
    *   Overfitting seems more pronounced (higher validation loss compared to Trial 3).

---

### Trial 5: Transformer with Data Augmentation

*   **Code File:** `trial 5 code.py`
*   **Objective:** Evaluate the impact of image data augmentation on the Transformer model performance, using a model size similar to Trial 3 but with some hyperparameter tweaks from Trial 4.

**1. Key Configuration Changes from Trial 4 (and referencing Trial 3):**
    *   **Data Augmentation:** Yes.
        *   `train_transform`: Includes `Resize(288)`, `RandomResizedCrop(256)`, `RandomHorizontalFlip(p=0.5)`, `ColorJitter`, `RandomRotation(degrees=10)`.
        *   `val_transform`: Standard resize and normalize (no augmentation).
    *   **Model Architecture (`TransformerDecoderModel`):** (Reverted to Trial 3's size, but kept dropout from Trial 4)
        *   `d_model = 512` (like Trial 3).
        *   `num_transformer_decoder_layers = 6` (like Trial 3).
        *   `dim_feedforward = 2048` (like Trial 3).
        *   `transformer_dropout = 0.15` (kept from Trial 4, higher than Trial 3's 0.1).
    *   **Training Specifics:**
        *   **`max_caption_length = 65`** (Increased from Trial 4's 25, closer to Trial 3's 75).
        *   **Loss Function:** `nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, label_smoothing=0.1)` (kept from Trial 4).
        *   **Optimizer:** `optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.05)`. (Kept AdamW, adjusted `weight_decay`).

**2. Vocabulary:**
    *   Same as Trial 3 & 4 (SentencePiece, target size 8000).

**3. Results (from `mixed trial results.csv` for Trial_ID 5):**
    *   **Trainable Parameters:** 57,981,312 (same as Trial 3, significantly less than Trial 4)
    *   **Data Augmentation:** Yes
    *   **Epochs Run:** 20
    *   **Stopping Reason:** Finished all epochs
    *   **Best BLEU-4:** 0.0725 (achieved at epoch 5)
    *   **Avg Train Loss (at epoch 20):** 2.4171
    *   **Avg Val Loss (at epoch 20, from epoch 20 eval):** 4.1920
    *   **Avg Train Time/Epoch:** 331s
    *   **Avg Val Time/Epoch:** 2995s

**4. Observations:**
    *   This trial achieved the highest BLEU-4 score of 0.0725.
    *   Data augmentation appears to be beneficial.
    *   The model size is the same as Trial 3. The increased `transformer_dropout` (0.15 vs 0.1) and `weight_decay` (0.05 vs 0.01 in AdamW, or 1e-5 in Adam for T3) along with augmentation might have contributed to better generalization or robustness, leading to the slight BLEU score improvement.
    *   `max_caption_length = 65` seems like a reasonable compromise.
    *   Again, the best score was achieved early at epoch 5, with subsequent overfitting.

---

### Comparative Analysis & Key Findings:

1.  **LSTM vs. Transformer:** The transition from LSTM-based models (Trials 1 & 2, best BLEU-4 ~0.055) to Transformer-based models (Trials 3-5, best BLEU-4 ~0.072) yielded a significant improvement in caption quality as measured by BLEU-4. Transformers, with their multi-head attention mechanism, are generally more powerful for sequence-to-sequence tasks.

2.  **Vocabulary (Custom vs. SentencePiece):** The switch from a custom word-based vocabulary (Trials 1 & 2) to SentencePiece (Trials 3-5) coincided with the architecture change to Transformer, making it hard to isolate its impact. However, SentencePiece is generally favored for its ability to handle rare words, morphology, and reduce vocabulary size with subword units, which is particularly useful for morphologically rich languages like Gujarati. This likely contributed to the Transformer's better performance.

3.  **Model Size and Complexity:**
    *   Increasing LSTM layers from 1 (Trial 1, 39M params) to 2 (Trial 2, 41M params) gave a slight boost.
    *   The base Transformer (Trial 3, 58M params) significantly outperformed LSTMs.
    *   A much larger Transformer (Trial 4, 113M params) did not improve over the base Transformer (Trial 3) and performed slightly worse. This could be due to the limited dataset size (Flickr8k is relatively small for very large models) or the restrictive `max_caption_length=25` in Trial 4.
    *   Trial 5 (58M params, same as Trial 3) with data augmentation achieved the best result, suggesting that for this dataset size, a moderately sized Transformer with good regularization (dropout, weight decay, augmentation) is optimal.

4.  **Data Augmentation:** Introducing image augmentation in Trial 5 for the Transformer model led to the best BLEU-4 score, indicating its effectiveness in improving model generalization and robustness.

5.  **Decoding Strategy (Greedy vs. Beam Search):**
    *   Trial 1 (LSTM) used greedy decoding.
    *   Trial 2 (LSTM) used beam search and saw an improvement.
    *   Trials 3-5 (Transformer) all used beam search. Beam search generally produces better quality sequences than greedy decoding for NLG tasks.

6.  **Hyperparameters & Regularization:**
    *   **`max_caption_length`:** This proved to be an influential hyperparameter. Trial 4's very short length (25) might have limited its performance despite a larger model. Trial 3 (75) and Trial 5 (65) used more generous lengths.
    *   **Optimizers:** AdamW with appropriate weight decay and label smoothing (Trials 4 & 5) are standard for Transformers and likely beneficial, though their impact wasn't dramatically isolated here.
    *   **Overfitting:** All trials, especially Transformer-based ones, showed signs of overfitting, with the best BLEU-4 scores often achieved early in training (e.g., epoch 5 or 9). This suggests that more aggressive early stopping, a more sensitive LR scheduler, or further regularization techniques might be needed. The current `ReduceLROnPlateau` patience of 2 (for LR) and early stopping patience of 5 (for training) might be too long if performance peaks very early.

7.  **Evaluation Frequency:** Evaluating only every 5 epochs (Trials 3-5) saves computation but might miss the true peak performance if it occurs between these evaluation points. It also makes the learning rate scheduler and early stopping less responsive.

**Conclusion:**

The experiments demonstrate a clear progression in performance. The Transformer architecture combined with SentencePiece vocabulary (Trial 3) provided a substantial leap over LSTM-based models. Further gains were achieved by introducing data augmentation (Trial 5) and carefully tuning model size and other hyperparameters. The best performing model (Trial 5) used a Transformer of moderate size (58M parameters), SentencePiece vocabulary, image data augmentation, and appropriate regularization techniques, achieving a BLEU-4 score of 0.0725.

**Potential Future Work:**
*   More sophisticated learning rate scheduling (e.g., cosine annealing with warm-up).
*   More aggressive early stopping or checkpointing based on validation loss as well as BLEU.
*   Experiment with different pre-trained image encoders (e.g., EfficientNet, Vision Transformer).
*   Explore larger datasets if available.
*   Fine-tune the image encoder along with the decoder.
*   Investigate attention visualization for Transformers to understand model focus.
*   Systematic hyperparameter optimization (e.g., using Optuna or Ray Tune) for `d_model`, `num_layers`, `dropout`, `max_caption_length`, and optimizer settings.
*   More frequent validation to better capture peak performance.

---

This documentation should provide a solid technical overview of your experiments. Let me know if you have any specific sections you'd like to elaborate on further!