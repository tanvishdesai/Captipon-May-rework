Okay, this is a comprehensive set of experiments for Gujarati image captioning. I'll break down the documentation into sections: an overall introduction, common components, detailed descriptions of each model type, a comparative analysis based on your results, and finally, general observations and potential future work.

---

## Documentation: Gujarati Image Captioning Model Experiments

**Project Goal:** To develop and compare different deep learning models for generating descriptive captions in Gujarati for given images, utilizing the Flickr8k dataset with custom Gujarati annotations.

**Document Purpose:** This document details three distinct model architectures (referred to as Model Type-1, Model Type-2, and Model Type-3), their implementation specifics, training procedures, and a comparative analysis of their performance based on provided experimental results.

---

### I. Common Components and Preprocessing

Across all three experiments, several components and preprocessing steps are shared or conceptually similar:

1.  **Dataset:**
    *   **Images:** Flickr8k dataset (typically `/kaggle/input/flickr8k/Images`).
    *   **Captions:** Custom Gujarati captions provided in a text file (e.g., `/kaggle/input/guj-captions/gujarati_captions.txt`). The format appears to be `image_filename#caption_index\tGujarati_caption_text`. Each image typically has multiple associated captions.

2.  **Vocabulary and Tokenization (SentencePiece):**
    *   All models leverage **SentencePiece** for subword tokenization of Gujarati captions. This is crucial for handling a morphologically rich language like Gujarati, managing vocabulary size, and dealing with out-of-vocabulary words.
    *   A vocabulary is built from the corpus of all Gujarati captions.
    *   **Special Tokens:** Standard special tokens are used:
        *   `<pad>`: For padding sequences to a uniform length.
        *   `<sos>`: Start-of-sentence token, prepended to captions.
        *   `<eos>`: End-of-sentence token, appended to captions.
        *   `<unk>`: Unknown token, for subwords not in the learned vocabulary.
    *   The SentencePiece model type varies (BPE in Type-1 & Type-3, Unigram in Type-2), which can affect tokenization granularity and performance.
    *   Vocabulary size is a key hyperparameter (e.g., `SP_VOCAB_SIZE = 8000`).

3.  **Image Transformations:**
    *   Standard `torchvision.transforms` are used for preprocessing images:
        *   `Resize`: To a fixed size (e.g., 256x256, 224x224).
        *   `ToTensor`: Convert PIL Image to PyTorch tensor.
        *   `Normalize`: Normalize tensor values using ImageNet mean and standard deviation.
    *   **Data Augmentation** (for training sets):
        *   Type-1 & Type-2 include more aggressive augmentation (RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation/Affine).
        *   Type-3 uses a simpler common transform without extensive augmentation during training.

4.  **Data Splitting:**
    *   Unique image names are typically split into training and validation sets (e.g., 80/20 split). All captions associated with an image go to the respective set.

5.  **Evaluation Metrics:**
    *   **BLEU (Bilingual Evaluation Understudy):** Primary metric used across all models (BLEU-1 to BLEU-4). Calculated using `nltk.translate.bleu_score.corpus_bleu`. References and hypotheses are tokenized by SentencePiece into subwords for this calculation.
    *   **METEOR & ROUGE-L (Model Type-3):** Model Type-3 additionally reports METEOR (`nltk.translate.meteor_score.meteor_score`) and ROUGE-L (`rouge_score.rouge_scorer`). For these, decoded (textual) captions are used, typically tokenized by NLTK's `word_tokenize` for METEOR.

6.  **Caption Generation:**
    *   All models employ **Beam Search** for generating captions during evaluation and inference to produce more coherent and higher-quality outputs than greedy decoding. The beam width is a configurable parameter.

7.  **Hardware & Environment:**
    *   Experiments are run on a `DEVICE` (typically `cuda` if available, else `cpu`).
    *   PyTorch is the primary deep learning framework.

---

### II. Model Architectures and Experiment Details

#### A. Model Type-1: ResNet50 Encoder + Transformer Decoder

*   **Core Idea:** A standard and powerful encoder-decoder architecture using a pre-trained CNN for image feature extraction and a Transformer decoder for sequence generation.
*   **Encoder (`EncoderCNN`):**
    *   Uses a **ResNet50** model pre-trained on ImageNet.
    *   The last two layers (average pooling and fully connected layer) of ResNet50 are removed.
    *   Output features are `(batch_size, H', W', 2048)`.
    *   These features are permuted and reshaped to `(batch_size, num_pixels, encoder_dim=2048)`, effectively treating different spatial locations as a sequence of features.
*   **Decoder (`TransformerDecoderModel`):**
    *   **Embedding Layer:** Gujarati subword tokens are mapped to `d_model` dimensional embeddings.
    *   **Positional Encoding:** Added to embeddings to provide sequence order information.
    *   **Encoder Feature Projection:** A `nn.Linear` layer projects the `encoder_dim` (2048) image features to `d_model` to match the decoder's dimensionality. This projected output serves as the `memory` for the Transformer decoder.
    *   **Transformer Decoder Stack:** Composed of `num_transformer_decoder_layers` (e.g., 6) `nn.TransformerDecoderLayer` instances. Each layer contains:
        *   Self-attention mechanism (masked to prevent attending to future tokens).
        *   Cross-attention mechanism (attending to the projected image features/`memory`).
        *   Feed-forward network.
    *   **Output Layer:** A final `nn.Linear` layer maps the decoder's output to vocabulary size logits.
*   **Vocabulary:**
    *   `SentencePieceVocabulary` class.
    *   SentencePiece model type: `bpe`.
    *   Vocabulary size: `SP_VOCAB_SIZE = 8000`.
    *   Special token IDs: `<pad>=0`, `<unk>=1`, `<sos>=2`, `<eos>=3`.
*   **Training Details:**
    *   Optimizer: `AdamW` (lr=`1e-4`, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.05).
    *   Loss Function: `nn.CrossEntropyLoss` with `label_smoothing=0.1` and `ignore_index=vocab.pad_idx`.
    *   Learning Rate Scheduler: `ReduceLROnPlateau` (mode='max', factor=0.1, patience=`lr_patience=2`), monitors BLEU-4.
    *   Gradient Clipping: `grad_clip_norm = 5.0`.
    *   Batch Size: 32.
    *   Data Augmentation: `RandomResizedCrop(256, scale=(0.8, 1.0))`, `RandomHorizontalFlip(p=0.5)`, `ColorJitter`, `RandomRotation(degrees=10)`.
    *   Key Hyperparameters: `d_model=512`, `nhead=8`, `num_transformer_decoder_layers=6`, `dim_feedforward=2048`, `transformer_dropout=0.15`.
*   **Caption Generation:**
    *   `sample_beam_search` method implemented within `TransformerDecoderModel`. Default `beam_width=5` in code, max length `max_caption_length`.
*   **Code Highlights:**
    *   `EncoderCNN` class for ResNet50 feature extraction.
    *   `PositionalEncoding` class.
    *   `TransformerDecoderModel` class encapsulating the decoder logic.
    *   `ImageCaptioningModel` combining encoder and decoder.
    *   Explicit data augmentation in `train_transform`.
    *   Validation frequency: `eval_frequency = 5` epochs.
    *   Early stopping based on BLEU-4 not improving for `patience_early_stop = 5` *evaluation cycles* (i.e., `5 * eval_frequency` epochs).

#### B. Model Type-2: EfficientNet-B0 Encoder + Transformer Decoder

*   **Core Idea:** Similar to Model Type-1 (CNN Encoder + Transformer Decoder) but utilizes a more modern and efficient CNN (EfficientNet-B0) and has a slightly different structure for integrating image features into the decoder.
*   **Encoder (`ImageEncoder`):**
    *   Uses an **EfficientNet-B0** model pre-trained on ImageNet (default weights).
    *   Allows fine-tuning of the last `fine_tune_last_n_blocks` (e.g., 2) of EfficientNet. Other layers are frozen.
    *   The `features` part of EfficientNet is used, followed by `avgpool`.
    *   A `nn.Linear` layer projects the flattened features from `avgpool` (1280 for B0) to `d_model`.
    *   This results in a single feature vector per image: `(batch_size, d_model)`.
*   **Decoder (`CaptioningTransformer`):**
    *   **Embedding Layer:** Gujarati subword tokens mapped to `d_model` embeddings.
    *   **Positional Encoding:** Added to embeddings.
    *   **Memory Preparation:** The single image feature vector from the encoder `(batch_size, d_model)` is `unsqueeze(1)` to become `(batch_size, 1, d_model)`. This serves as the `memory` for the Transformer decoder, meaning the decoder attends to a single, global image representation.
    *   **Transformer Decoder Stack:** Similar to Model Type-1, using `nn.TransformerDecoderLayer` and `nn.TransformerDecoder`.
    *   **Output Layer:** A final `nn.Linear` layer to vocabulary size logits.
*   **Vocabulary:**
    *   Trained via `train_sentencepiece_model` function.
    *   SentencePiece model type: `unigram`.
    *   Vocabulary size: `spm_vocab_size = 8000`.
    *   Special token IDs (`pad_token_id`, `sos_token_id`, etc.) are dynamically queried from the trained SPM and stored in `CONFIG`. `<PAD>` is a user-defined symbol.
*   **Training Details:**
    *   Optimizer: `AdamW` (lr=`1e-4`, weight_decay=`1e-5`).
    *   Loss Function: `nn.CrossEntropyLoss` with `ignore_index=CONFIG['pad_token_id']`. (Label smoothing not explicitly mentioned in loss instantiation).
    *   Learning Rate Scheduler: `ReduceLROnPlateau` (mode='max', factor=0.2, patience=`CONFIG['patience_early_stopping'] // 2`), monitors BLEU-4.
    *   Gradient Clipping: `grad_clip_value = 1.0`.
    *   Batch Size: 32.
    *   Data Augmentation: `RandomHorizontalFlip`, `ColorJitter`, `RandomAffine`.
    *   Key Hyperparameters: `d_model=512`, `n_heads=8`, `num_decoder_layers=6`, `dim_feedforward=2048`, `dropout_rate=0.1`.
*   **Caption Generation:**
    *   `generate_caption` method with beam search.
    *   `beam_width_generation=5`, `length_penalty_alpha=0.7`.
*   **Code Highlights:**
    *   `ImageEncoder` class for EfficientNet-B0 feature extraction.
    *   `CaptioningTransformer` class.
    *   `EarlyStopping` class based on BLEU-4.
    *   Dynamic handling of SPM special token IDs.
    *   Validation performed every epoch.

#### C. Model Type-3: ResNet18 Encoder + GRU Decoder

*   **Core Idea:** A more lightweight, traditional RNN-based encoder-decoder model.
*   **Encoder (within `LightweightCaptioningModel`):**
    *   Uses a **ResNet18** model pre-trained on ImageNet.
    *   The final fully connected layer of ResNet18 is removed (`*list(resnet.children())[:-1]`).
    *   Output features are `(batch_size, resnet.fc.in_features, 1, 1)`.
    *   These features are flattened (`view(batch_size, -1)`).
    *   An `nn.Linear` layer (`image_projection`) projects these flattened features (512 for ResNet18) to `hidden_dim`.
    *   This projected feature vector serves as the initial hidden state for the GRU.
*   **Decoder (within `LightweightCaptioningModel`):**
    *   **Embedding Layer:** Gujarati subword tokens mapped to `embed_dim` embeddings.
    *   **Embedding Projection:** An `nn.Linear` layer (`embed_projection`) projects embeddings from `embed_dim` to `hidden_dim` to match GRU input size.
    *   **GRU Layer:** A single-layer `nn.GRU` processes the sequence of projected embeddings.
        *   The initial hidden state is derived from the projected image features.
    *   **Output Layer:** A final `nn.Linear` layer maps the GRU's output to vocabulary size logits.
    *   **Dropout:** Applied after image projection and embedding.
*   **Vocabulary:**
    *   `SentencePieceVocabulary` class (similar to Model Type-1).
    *   SentencePiece model type: `bpe`.
    *   Vocabulary size: `sp_vocab_size = 8000`.
    *   Special token IDs: `<pad>=0`, `<unk>=1`, `<sos>=2`, `<eos>=3`.
*   **Training Details:**
    *   Optimizer: `Adam` (lr=`4e-4`).
    *   Loss Function: `nn.CrossEntropyLoss` with `ignore_index=vocab.pad_idx`.
    *   Gradient Clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.
    *   Batch Size: 32.
    *   Data Augmentation: Basic `Resize`, `ToTensor`, `Normalize` via `common_transform`. No heavy augmentation explicitly applied during training dataset creation.
    *   Key Hyperparameters: `embed_dim=256`, `hidden_dim=512`.
*   **Caption Generation:**
    *   `generate_caption_beam_search` method.
    *   `beam_width=3`.
    *   Length penalty applied as `score / (len(seq)**0.7)`.
*   **Code Highlights:**
    *   `LightweightCaptioningModel` class encapsulating both encoder and GRU decoder.
    *   `SimpleDataset` for training (image-caption pairs) and `UniqueImageDataset` for validation (image to list of captions).
    *   Evaluation includes METEOR and ROUGE-L in addition to BLEU.
    *   Validation performed every epoch.

---

### III. Results and Comparative Analysis

Based on the `final Model's results comparison intial.csv`:

| Feature                      | Model type-1 (ResNet50+TF) | Model type-2 (EffNetB0+TF) | Model type-3 (ResNet18+GRU) |
| :--------------------------- | :------------------------- | :------------------------- | :-------------------------- |
| Trainable Params (M)       | 57.98                      | 35.21                      | 19.30                       |
| Vocab Source                 | Built (SP BPE)             | Trained (SP Unigram)       | Built (SP BPE)              |
| Vocab Size (Target)        | 8000                       | 8000                       | 8000                        |
| Batch Size                   | 32                         | 32                         | 32                          |
| Learning Rate                | 1e-4                       | 1e-4                       | 4e-4 (from code)            |
| Beam Width (Gen)           | 5 (from code)              | 5                          | 3                           |
| Dataset Train Imgs         | 6472                       | 6473                       | 6472                        |
| Dataset Val Imgs           | 1619                       | 809                        | 1619                        |
| Epochs Ran                   | 10                         | 22                         | 20                          |
| Epochs Planned               | 20                         | 30                         | 20                          |
| Total Eff. Train Time (s)    | 3318.11                    | 25215.46                   | 5350.60                     |
| Avg Train Epoch Time (s)   | 331.81                     | 1146.16                    | 267.53                      |
| Early Stopping Triggered     | Yes                        | Yes                        | No                          |
| Best Metric Tracked          | BLEU-4                     | BLEU-4                     | BLEU-4                      |
| Best Epoch for Metric        | 5                          | 17                         | 5                           |
| Val Loss at Best Epoch       | 3.9647                     | 2.8309                     | N/A (not in CSV for best)   |
| **BLEU-4 at Best Epoch**     | **0.0744**                 | **0.2276**                 | **0.1742**                  |
| BLEU-3 at Best Epoch     | 0.1201                     | 0.3215                     | 0.2677                      |
| BLEU-2 at Best Epoch     | 0.2090                     | 0.4510                     | 0.4049                      |
| BLEU-1 at Best Epoch     | 0.3176                     | 0.6093                     | 0.5654                      |
| Peak GPU Memory MB         | N/A (not logged explicitly)  | 1110.95                    | N/A (not logged explicitly) |

**Discussion of Results:**

1.  **Performance (BLEU-4):**
    *   **Model Type-2 (EfficientNet-B0 + Transformer Decoder) achieved the highest BLEU-4 score (0.2276).** This suggests that the combination of a modern efficient CNN and the Transformer decoder, with its attention mechanisms, is highly effective for this task. The use of a single global image feature as memory for the Transformer decoder seems to work well.
    *   Model Type-3 (ResNet18 + GRU) is the runner-up with a BLEU-4 of 0.1742. This is a respectable score for a more lightweight RNN-based model.
    *   Model Type-1 (ResNet50 + Transformer Decoder) performed the poorest on BLEU-4 (0.0744). Despite using a powerful ResNet50 and a Transformer decoder, its performance lagged. Several factors could contribute:
        *   **Validation Frequency:** Validating only every 5 epochs might mean the best model checkpoint was missed, or learning rate adjustments were less timely.
        *   **Early Stopping:** It stopped early at epoch 10 (out of 20 planned). The early stopping patience was 5 *evaluation cycles*, which means 5*5 = 25 epochs. However, the CSV says it stopped after 1 eval cycle (5 epochs) of no improvement. This implies it stopped after epoch 10 because epoch 5 was best and epoch 10 showed no improvement.
        *   **Image Feature Handling:** Providing spatial features (`num_pixels` sequence) to the Transformer decoder is a valid approach, but perhaps less effective than the global feature approach of Model Type-2 for this dataset/setup, or it required more tuning/training.
        *   **Hyperparameters:** The specific Transformer hyperparameters might not have been optimal.

2.  **Model Size & Efficiency:**
    *   Model Type-3 is the most lightweight (19.30M params) and has the fastest average training epoch time (267.53s).
    *   Model Type-2 (35.21M params) is mid-range in size. Its average epoch time is the longest (1146.16s), likely due to the more complex EfficientNet processing (especially if fine-tuning) and potentially the Transformer operations on each validation step for generation.
    *   Model Type-1 is the largest (57.98M params). Its average training epoch time is relatively moderate (331.81s).

3.  **Training Dynamics:**
    *   Model Type-2 ran for the most epochs (22) before early stopping, indicating it found improvements for a longer duration.
    *   Model Type-1 and Type-3 both identified their best BLEU-4 performance at epoch 5, but Type-3 continued training for all 20 epochs (no early stop), while Type-1 stopped at epoch 10. This suggests Type-3 might have overfit or plateaued after epoch 5 regarding BLEU-4, even if other metrics or training loss might have continued improving slightly.
    *   The very low ROUGE-L for Model Type-3 (0.0000) is concerning and might indicate an issue with its generation or the ROUGE calculation setup for that specific run, as BLEU and METEOR are reasonable. It's possible that the generated captions were very short or had very little lexical overlap with references at the n-gram level ROUGE measures.

4.  **Vocabulary and Tokenization:**
    *   Model Type-2 used 'unigram' SPM, while Type-1 and Type-3 used 'BPE'. Unigram models often result in more semantically meaningful subwords compared to BPE's merge-frequency approach. This could contribute to Type-2's better performance.

5.  **Data Augmentation:**
    *   Models Type-1 and Type-2 employed more significant data augmentation compared to Type-3. This could have helped them generalize better, especially Model Type-2.

---

### IV. General Observations and Potential Future Work

1.  **Transformer Architectures:** The Transformer decoder (Models Type-1 and Type-2) demonstrates strong potential, with Model Type-2 showing the best results. The attention mechanisms are key for capturing long-range dependencies and relevant image-text alignments.
2.  **Encoder Choice:** EfficientNet-B0 (Model Type-2) outperformed ResNet50 (Model Type-1) and ResNet18 (Model Type-3) as an image encoder backbone in these experiments.
3.  **Image Feature Representation for Transformer:** Model Type-2's approach of using a single global image feature vector as `memory` for the Transformer decoder was more effective than Model Type-1's approach of using a sequence of spatial features in this specific comparison.
4.  **RNNs as a Baseline:** Model Type-3 (GRU-based) provides a solid, computationally cheaper baseline. Its performance is commendable for its simplicity.
5.  **Hyperparameter Sensitivity:** All these models have numerous hyperparameters. The observed differences could be due to the architectures themselves or the specific hyperparameter settings and training configurations (e.g., learning rate schedule, optimizer, dropout rates, validation frequency).
6.  **SentencePiece Model Type:** The choice between 'BPE' and 'unigram' for SentencePiece can impact tokenization and, consequently, downstream performance.
7.  **Further Experimentation & Tuning:**
    *   **Cross-validate Hyperparameters:** More rigorous hyperparameter tuning for each model type.
    *   **Consistent Validation:** Ensure validation is run every epoch for all models to enable fairer comparison of learning dynamics and more precise early stopping.
    *   **Unified Augmentation:** Apply a consistent, strong augmentation strategy across all models during training for a fairer comparison of architectural strengths.
    *   **Investigate Model Type-1:** Further tune Model Type-1, perhaps experiment with different ways of integrating spatial features or try the global feature approach like Model Type-2. Check its learning rate scheduler and early stopping logic.
    *   **ROUGE for Model Type-3:** Investigate the low ROUGE-L score for Model Type-3.
    *   **Larger Encoders:** Experiment with larger variants of ResNet (e.g., ResNet101) or EfficientNet (e.g., B3, B4) if computational resources allow, ensuring decoder capacity is also scaled.
    *   **Pre-trained Language Model Decoders:** Explore using pre-trained Gujarati language models (if available) as decoders for potentially better fluency and contextual understanding.
    *   **Attention Visualization:** For Transformer models, visualizing attention weights could provide insights into what image regions the model focuses on when generating specific words.

---

### V. Conclusion

The experiments provide valuable insights into different architectural choices for Gujarati image captioning. **Model Type-2 (EfficientNet-B0 Encoder + Transformer Decoder)** emerged as the top performer in terms of BLEU-4 score, highlighting the strength of modern CNNs combined with Transformer decoders. Model Type-3 (ResNet18 + GRU) offers a good balance of performance and computational efficiency. Model Type-1, while conceptually sound, underperformed in this setup and would require further investigation and tuning. These results form a strong basis for further development and refinement of Gujarati image captioning systems.

---