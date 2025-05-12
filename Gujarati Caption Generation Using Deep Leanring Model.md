# Gujarati Caption Generation Using DL Models 

**Abstract:**

Image captioning is an important branch of artificial intelligence that enables the creation of natural descriptions for visual content. Regional languages like Gujarati are still completely unexplored, despite significant progress in widely spoken languages like English. The goal of this study is to create a Gujarati picture captioning model using the Flickr8k dataset's translated captions. Text is generated using a GRU-based decoder, and features are extracted using ResNet-18. The suggested method shows promise for Gujarati picture captioning, despite the fact that state-of-the-art performance levels have not been reached. Along with a discussion of the difficulties faced and possible avenues for development, BLEU ratings are presented as evaluation measures.

**1\. Introduction:**

Image captioning is a critical study topic in artificial intelligence that includes creating descriptive textual representations of visual input. In order to comprehend images and translate them into plain language, this challenge combines computer vision and natural language processing (NLP). While English-based models have demonstrated great improvement, low-resource languages including Hindi, Urdu, Tamil, Arabic, Assamese, Bengali, and Vietnamese confront hurdles due to dataset scarcity, linguistic complexities, and lack of established evaluation benchmarks \[1\]. By using deep learning architectures, attention mechanisms, and transfer learning techniques, recent research have tried to close this gap \[1\] \[2\] \[3\] \[4\] \[5\] \[6\] \[7\] \[8\].

**Developments in Capturing Language Images with Low Resources:**

ResNet50 \+ LSTM with attention dramatically raises BLEU scores, according to research on Hindi picture captioning, highlighting the importance of attention-based architectures in enhancing contextual comprehension \[1\]. CNN-GRU architectures have also been successful in Urdu picture captioning, with a BLEU score of 0.83 demonstrating the efficacy of GRUs in sequence-based tasks \[2\]. The Merge CNN-LSTM model with Inception-V3 has been investigated in Tamil captioning studies, yielding BLEU scores of up to 0.37, indicating its promise for low-resource environments \[3\].

Researchers have improved BLEU-1 and BLEU-4 scores in the Arabic picture captioning example by introducing **artificial neural networks** (ANNs) with transformers and attention mechanisms \[4\]. Despite difficulties with dataset translation, the Assamese captioning study used EfficientNetB3 with Bahdanau attention and produced competitive BLEU scores \[5\]. By using human annotated captions, the Vietnamese image captioning model, which was based on ResNet152/VGG16 \+ LSTM, achieved a BLEU-1 score of 0.71, outperforming machine-translated captions \[6\].

A generic image captioning model using Xception \+ LSTM demonstrated the effectiveness of transfer learning for feature extraction, displaying promising results on the Flickr8K dataset \[7\]. Furthermore, Bengali image captioning leveraged context-aware attention in a ResNet50-BiGRU model, significantly improving BLEU and METEOR scores, emphasizing the importance of capturing contextual relationships in images \[8\].

**Extending Image Captioning to Gujarati:**

Gujarati image captioning is still unexplored in artificial intelligence research, despite recent developments. This paper suggests a Gujarati image captioning model that uses a Gated Recurrent Unit (GRU)-based decoder for sequence generation and a ResNet-18 encoder for feature extraction. With their computational efficiency and quicker convergence, GRUs are perfect for low-resource datasets, in contrast to earlier efforts that mostly relied on LSTMs \[1\] \[3\] \[6\] \[7\] \[2\] \[8\].

Assamese, Tamil, Arabic, Hindi, Urdu, Bengali, and Vietnamese image captioning research methods are followed for training the model on translated captions from the Flickr8K dataset \[1\] \[2\] \[3\] \[4\] \[5\] \[6\] \[7\] \[8\]. BLEU scores, a commonly used statistic in image captioning, are used to assess performance \[1\] \[2\] \[3\] \[6\] \[7\]. Even though preliminary findings show encouraging progress, problems including bias in the dataset, inconsistent grammar, and limited vocabulary still exist \[4\] \[5\] \[8\].

To improve Gujarati image captioning ability, this study investigates potential enhancements such as dual attention techniques \[8\], cross-domain adaptation \[7\], and data expansion \[6\].

   
**2\. Literature Review:**

In artificial intelligence (AI), picture captioning is a crucial issue that combines natural language processing (NLP) and computer vision to produce insightful descriptions of images. Earlier techniques used retrieval-based or template-based techniques that created captions by comparing photos with pre-written text \[1\]. Nevertheless, these techniques have little flexibility in producing a variety of captions and low generality \[2\].

Encoder-decoder architectures, in which Convolutional Neural Networks (CNNs) extract visual features and Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), or Gated Recurrent Units (GRU) construct captions, have been more and more popular among academics since the introduction of deep learning \[3\] \[4\]. To improve the contextual understanding of images, attention methods have also been added. This enables models to concentrate on pertinent areas when producing captions \[5\].

Research in low-resource languages, including Hindi, Urdu, Tamil, Arabic, Assamese, Bengali, and Vietnamese, has gained impetus due to the need for linguistic variety and accessibility, even though English-based captioning models have made great strides in this area \[6\] \[7\] \[8\]. Recent developments in low-resource language image captioning are summarized in the following subsections.

**2.1 Image Captioning in Hindi**

Despite being one of the most frequently spoken languages, there is very little research on captioning images in Hindi. An LSTM-based model with an attention mechanism was presented in a study by Sethi et al., which enhanced contextual coherence and caption fluency \[1\]. In comparison to baseline LSTM models, their model performed better and used ResNet50 for feature extraction. However, issues like vocabulary restrictions, syntax inconsistencies, and a paucity of datasets continue to undermine the model's effectiveness \[1\].

**2.2 Image Captioning in Urdu**

A notable study obtained a BLEU score of 0.83, indicating the effectiveness of GRUs in low-resource language captioning \[2\]. However, the morphological complexity of Urdu, along with dataset translation challenges, hinders further improvements. Accordingly, future research is expected to focus on grammar refinement techniques and syntactic loss correction to improve accuracy. CNN-GRU architectures have been investigated for Urdu image captioning and have been found to outperform LSTMs in handling long-range dependencies \[2\].

**2.3 Image Captioning in Tamil**

CNN-LSTM architectures, namely Inception-V3 for feature extraction, have been used in Tamil image captioning research \[3\]. The potential of deep learning in Tamil captioning was demonstrated by a recent study that examined a Merge CNN-LSTM model and obtained a BLEU-1 score of 0.37 \[3\]. However, misidentifications of cultural objects, translation problems, and dataset biases continue to be major challenges. To increase caption fluency, the dataset must be expanded and model architectures must be improved.

**2.4 Image Captioning in Arabic**

Arabic poses special difficulties for image captioning because it is a morphologically complex language. By combining attention mechanisms and transformers with **artificial neural networks** (ANNs), researchers have considerably raised BLEU-1 and BLEU-4 ratings \[4\]. Models were able to increase word alignment and semantic accuracy by the use of preprocessing based on AraBERT \[4\]. The lack of high-quality Arabic datasets is still a problem, though, and multilingual pre-training strategies have been proposed as a potential avenue for further study.

**2.5 Image Captioning in Assamese**

A study on Assamese picture captioning used EfficientNetB3 with Bahdanau attention and obtained competitive BLEU scores despite dataset translation restrictions \[5\]. The significance of dataset quality in enhancing caption production was highlighted by the finding that humanly annotated datasets performed noticeably better than machine-generated captions \[5\]. Extending Assamese datasets and incorporating Transformer-based systems for more enhancements are potential future projects.

**2.6 Image Captioning in Vietnamese**

Using the UIT-ViIC dataset, Vietnamese image captioning has been thoroughly examined. The results indicate that hand-annotated captions perform noticeably better than machine-translated ones \[6\]. A BLEU-1 score of 0.71 was obtained by using ResNet152/VGG16 with LSTM decoders, indicating that human-curated datasets have a substantial impact on model performance \[6\]. However, the dataset's generalizability is limited by its emphasis on photos connected to sports. The goal of future research is to incorporate attention-based architectures and increase dataset coverage.

**2.7 General Image Captioning Models**

There have been investigations into general picture captioning models in addition to language-specific ones. In order to demonstrate how deep learning models may generalize to many languages, a study used Xception \+ LSTM architectures on Flickr8K \[7\]. According to the findings, transfer learning preserves competitive BLEU scores while drastically lowering computational costs \[7\]. Nonetheless, the lack of multilingual flexibility and the diversity of datasets continue to be major drawbacks.

**2.8 Image Captioning in Bengali**

Research on Bengali captioning has improved BLEU and METEOR scores by integrating context-aware attention mechanisms into a ResNet50-BiGRU model \[8\]. According to the study, attention-based architectures improve contextual awareness and object localization, which results in more accurate and cohesive captions \[8\]. Nevertheless, problems with cultural diversity and dataset biases persist, indicating the necessity of domain adaptation and wider dataset inclusion.

**2.9 Research Gaps and Future Directions**

Despite advancements in other low-resource languages, Gujarati image captioning has not yet been explored \[1\] \[2\] \[3\] \[4\] \[5\] \[6\] \[7\] \[8\]. Since most rely on machine translation, a major problem is the dearth of high-quality natively annotated datasets \[6\]. Research on Bengali and Urdu indicates that GRUs perform better than LSTMs, which makes them a good option \[2\] \[8\]. Furthermore, attention techniques, such as transformers and dual attention, have been shown to improve fluency in Bengali, Tamil, Hindi, and Assamese, indicating that Gujarati could benefit from their use. Transformers offer a possible approach to context-aware captioning, and cross-domain adaptation can enhance generalization even more \[7\]. Finally, for improved coherence, Gujarati's morphological complexity—like that of Arabic and Assamese—needs grammatical awareness \[4\] \[5\].

**3\. METHODOLOGY:**

The model architecture, training procedure, assessment metrics, and dataset preparation are all covered in this portion of the suggested methodology for Gujarati picture captioning. The method uses deep learning techniques, attention mechanisms, and transfer learning to build on successful image captioning models in other low-resource languages \[1\] \[2\] \[3\] \[4\] \[5\] \[6\] \[7\] \[8\].

**3.1 Dataset Preparation**

**3.1.1 Dataset Selection**

The Flickr8K dataset, which has been extensively utilized in earlier research for low-resource language image captioning \[1\] \[2\] \[3\] \[6\] \[7\], is used to train the suggested model. Each of the 8,000 photos in the dataset has five English captions that offer a variety of textual descriptions.

**3.1.2 Gujarati Caption Translation**

Gujarati image captioning has not been extensively investigated, so English captions from the Flickr8K dataset are translated into Gujarati using a combination of:

* neural machine translation (NMT) models that have already been trained, like Google Translate.  
* To guarantee linguistic authenticity, native Gujarati speakers make manual adjustments.

This semi-supervised dataset development procedure is similar to approaches employed in Hindi, Urdu, and Assamese captioning studies \[1\] \[2\] \[5\].

**3.1.3 Data Preprocessing**

Images and text are preprocessed before being fed into the model:

* Image Processing: Images are downsized to 224×224 pixels and adjusted for better efficiency.  
* Text Tokenization: Tokenized captions are padded to a predetermined sequence length and transformed into Gujarati word embeddings.  
* Start & End Tokens: To help the model identify caption borders, each caption is wrapped in and tokens \[7\].

**3.2 Model Architecture**

The proposed model employs a hybrid encoder-decoder architecture, consisting of a ResNet-18-based feature extractor and a GRU-based caption generator, as inspired by recent advances in low-resource language captioning \[2\] \[5\] \[8\].

**3.2.1 Encoder: Feature Extraction using ResNet-18**

The input dataset's picture features are extracted using a pre-trained ResNet-18 CNN. The selection of ResNet-18 is based on:

* less expensive to compute than more complex CNNs like Inception-V3 or ResNet-50.  
* Efficiency with tiny datasets, as shown in comparable low-resource captioning studies \[3\] \[6\].  
* Transfer learning for Gujarati captions is possible because it is pre-trained on ImageNet.

The output feature vector is sent into the decoder after the final fully connected layer of ResNet-18 is eliminated.

**3.2.2 Decoder: GRU-based Caption Generator**

A Gated Recurrent Unit (GRU) in the decoder creates Gujarati captions word by word. The following reasons make GRUs superior to LSTMs:

* faster training-to-training convergence \[2\].  
* They are more suited for datasets with limited resources since they consume less memory \[8\].

The GRU uses word embeddings and an attention strategy to parse the caption sequence after receiving picture feature vectors from ResNet-18.

**3.2.3 Attention Mechanism**

To improve caption accuracy and contextual relevance, Bahdanau Attention is used, which allows the model to dynamically focus on different portions of an image while generating text \[5\] \[8\]. The attention layer helps:

* Align text with pertinent visual elements.  
* By connecting visual clues with sentence structure, you can increase the coherence of Gujarati grammar.

**3.3 Training Process**

**3.3.1 Model Training Configuration**

The model is trained using:

* Batch size: 64  
* Optimizer: Adam (learning rate \= 0.0001)  
* Loss function: Categorical Cross-Entropy  
* Training-validation split: 80%-20%  
* Epochs: 50

**3.3.2 Transfer Learning**

The GRU decoder is trained from scratch; however, pre-trained ResNet-18 weights are employed to increase training efficiency. Bengali, Vietnamese, and Hindi captioning models have all used comparable transfer learning techniques \[1\] \[6\] \[8\].

**3.3.3 Data Augmentation**

Similar to the augmentation strategies employed in Assamese and Arabic image captioning research \[4\] \[5\], data augmentation techniques are applied to the small Gujarati dataset to prevent overfitting. These techniques include random rotation and flipping for images, synonym replacement, and sentence restructuring for captions.

**3.4 Evaluation Metrics**

The following metrics are used to assess the suggested model's performance:

**3.4.1 BLEU Score (Bilingual Evaluation Understudy)**

A popular metric for assessing the precision of text synthesis in picture captioning is BLEU. Reference captions at varying n-gram levels (BLEU-1 to BLEU-4) are compared to the model's output. \[1\] \[2\] \[3\] \[6\].

**3.4.2 METEOR (Metric for Evaluation of Translation with Explicit ORdering)**

METEOR is appropriate for Gujarati, which has a variable word order, since it assesses semantic soundness by taking synonyms and word alignments into account \[8\].

**3.4.3 CIDEr (Consensus-based Image Description Evaluation)**

CIDEr compares the sentence similarity of human-annotated and produced captions. It is helpful in low-resource languages where semantic correctness might not be captured by BLEU alone \[7\].

**3.5 Summary of the Methodology**

| Component | Approach | Justification |
| :---- | :---- | :---- |
| Dataset |  Flickr8K (Translated to Gujarati)   | Used in previous low-resource studies \[1\] \[6\]. |
|  Image Encoder   | ResNet-18 | Efficient for small datasets \[3\] \[6\]. |
|  Text Decoder   | GRU |  Faster and memory-efficient compared to LSTM \[2\] \[8\].   |
| Attention |  Bahdanau Attention   |  Enhances caption fluency \[5\] \[8\].   |
|  Training Optimizer   |  Adam   |  Faster convergence \[7\].   |
|  Evaluation Metrics   |  BLEU, METEOR, CIDEr   |   Used in low-resource NLP studies \[1\] \[6\] \[8\]. |

**3.6 Implementation Framework**

The model is implemented using:

* **Programming Language:** Python  
* **Deep Learning Libraries:** TensorFlow, Keras  
* **Hardware:** NVIDIA GPU for training  
* **Dataset Storage:** Google Drive / Local System

**3.7 Challenges and Future Considerations**

* There are few high-quality Gujarati datasets available.  
* Gujarati's rich morphology needs attention models that are aware of grammar.  
* As demonstrated by research conducted in Arabic and Assam, there is a need for more extensive pretraining \[4\] \[5\].  
* Investigating Transformers, which have performed better in the most current Bengali captioning models \[8\].

**4\. Results and Discussion**