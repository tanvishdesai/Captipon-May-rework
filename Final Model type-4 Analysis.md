# Analysis of Final Model Type-4: Gujarati Image Captioning

## 1. Introduction

This document details the architecture, design rationale, and components of **Model Type-4** for Gujarati image captioning. This model aims to build upon the strengths observed in previous experiments (Model Types 1, 2, and 3), primarily leveraging the successful architecture of Model Type-2 and incorporating more comprehensive evaluation metrics as seen in Model Type-3.

**Project Goal:** To develop an optimized deep learning model for generating descriptive Gujarati captions for images from the Flickr8k dataset, using custom Gujarati annotations.

## 2. Design Rationale and Baseline

*   **Baseline:** Model Type-4 is directly derived from **Model Type-2 (EfficientNet-B0 Encoder + Transformer Decoder)**. This choice is based on Model Type-2 achieving the highest BLEU-4 score (0.2276) in comparative experiments, indicating a robust and effective architecture.
*   **Incorporated Strengths:**
    *   **High-Performing Architecture:** Retains the core EfficientNet-B0 encoder and Transformer decoder structure from Model Type-2.
    *   **Tokenization Strategy:** Continues to use 'unigram' SentencePiece model type, which was associated with the best performing Model Type-2 and may offer better subword segmentation for Gujarati.
    *   **Comprehensive Evaluation:** Integrates METEOR and ROUGE-L scores into the evaluation pipeline, a practice noted in Model Type-3's codebase, allowing for a more holistic understanding of caption quality beyond BLEU scores.
    *   **Data Augmentation:** Preserves the robust data augmentation techniques used in Model Type-2 to enhance model generalization.
    *   **Configuration and Logging:** Maintains clear configuration management, training time logging, and peak GPU memory logging.

## 3. Model Architecture

Model Type-4 consists of an image encoder and a Transformer-based caption decoder.

### 3.1. Image Encoder

*   **Type:** `ImageEncoder` (adapted from Model Type-2).
*   **Backbone:** EfficientNet-B0, pre-trained on ImageNet.
*   **Fine-tuning:** The last few blocks of EfficientNet-B0 are fine-tuned during training (configurable by `fine_tune_last_n_blocks`, typically 2), while earlier blocks remain frozen. This balances transfer learning with task-specific adaptation.
*   **Output:** The encoder processes an input image and produces a single global feature vector of dimension `d_model` (e.g., 512). This vector is then passed as `memory` to the Transformer decoder.

### 3.2. Caption Decoder

*   **Type:** `CaptioningTransformer` (adapted from Model Type-2).
*   **Components:**
    *   **Embedding Layer:** Input Gujarati subword tokens (from SentencePiece) are mapped to `d_model` dimensional embeddings. `padding_idx` is used to handle padded sequences.
    *   **Positional Encoding:** Standard sinusoidal positional encodings are added to the token embeddings to provide sequence order information.
    *   **Transformer Decoder Stack:** Composed of multiple `nn.TransformerDecoderLayer` instances (`num_decoder_layers`, e.g., 6). Each layer includes:
        *   Masked self-attention (to prevent attending to future tokens).
        *   Cross-attention (attending to the global image feature vector from the encoder).
        *   Feed-forward network.
    *   **Output Layer:** A final `nn.Linear` layer maps the decoder's output to logits over the Gujarati vocabulary.
*   **Key Hyperparameters:**
    *   `d_model`: Dimensionality of embeddings and Transformer model (e.g., 512).
    *   `n_heads`: Number of attention heads in the multi-head attention mechanisms (e.g., 8).
    *   `num_decoder_layers`: Number of decoder layers (e.g., 6).
    *   `dim_feedforward`: Dimension of the feed-forward network within each decoder layer (e.g., 2048).
    *   `dropout_rate`: Dropout applied within the decoder (e.g., 0.1).

## 4. Tokenization

*   **Method:** SentencePiece.
*   **Model Type:** 'unigram'.
*   **Vocabulary Size:** Target vocabulary size (e.g., 8000 subwords).
*   **Special Tokens:** `<pad>`, `<sos>`, `<eos>`, `<unk>` are handled, with specific IDs managed by the SentencePiece model and used by the PyTorch model.

## 5. Training Details

*   **Optimizer:** AdamW.
*   **Loss Function:** `nn.CrossEntropyLoss` (ignoring `<pad>` token).
*   **Learning Rate Scheduler:** `ReduceLROnPlateau` (monitoring BLEU-4 or another key validation metric).
*   **Gradient Clipping:** Applied to prevent exploding gradients.
*   **Data Augmentation:** Includes `RandomHorizontalFlip`, `ColorJitter`, `RandomAffine` to improve robustness.
*   **Early Stopping:** Implemented to stop training if the monitored validation metric (e.g., BLEU-4) does not improve for a set number of epochs.

## 6. Evaluation Metrics

Model Type-4 will be evaluated using a comprehensive set of metrics:

*   **BLEU (1-4):** Primary metric for n-gram precision compared to reference captions. Calculated using SentencePiece subword tokens.
*   **METEOR:** Measures unigram precision and recall, considering stemming and synonymy. Calculated on decoded text, tokenized by `nltk.word_tokenize`.
*   **ROUGE-L:** Measures longest common subsequence, focusing on recall. Calculated on decoded text.

## 7. Expected Improvements / Outcomes

*   **Performance:** By building on the strongest baseline (Model Type-2), Model Type-4 is expected to achieve high performance, comparable to or potentially slightly exceeding Model Type-2's BLEU scores.
*   **Comprehensive Insights:** The inclusion of METEOR and ROUGE-L will provide a more nuanced understanding of the generated captions' quality, beyond n-gram overlap.
*   **Well-Documented and Configurable:** The model code is intended to be well-organized, with clear configurations for paths, hyperparameters, and model identifiers.

## 8. (Future) Results and Comparative Analysis

*(This section will be populated after the model is trained and evaluated. It will include quantitative results for Model Type-4 and a comparison against Model Types 1, 2, and 3 based on all relevant metrics and training efficiency.)*
