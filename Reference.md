3. METHODOLOGY
The proposed methodology in this research has a parallel architecture which consists of
two pipelines. The first pipeline takes in the image data and generates features out of it.
We have utilized multiple pre – trained well known CNN models viz VGG16, Resnet50.
The second pipeline takes in processed reference captions and feeds them into an LSTM.
The outputs from both the pipelines are concatenated together to generate captions for
the image using a combination of dense layers consisting of 256 neurons. The
architecture of the model is explained in fig 2.
A. 102 Sethi et al. / Image Caption Generator in Hindi Using Attention
Fig. 2: Architecture of Proposed model
Before sending the reference captions through the pipeline, text data is pre-processed
as follows:
3.1. Text Wrapping
For the pre-processing purposes, the training dataset has been wrapped in between two
tags: <startseq> and <endseq>. This is done because the model predicts the captions
based on image features and previous words sequences. So the tags are just an indicator
for the model to stop because as soon as it predicts the <endseq>. It will stop predicting
new words.
3.2. Text Tokenization
We have done text tokenization and padding, as deep learning algorithms cannot
understand text, so we replace text by floating point values and they get processed by the
deep learning algorithm (RNN), in our case LSTM. The pre-processed captions were fed
into the LSTM model which were then concatenated with features generated from CNN
models.
To further enhance the results of the model, attention module was introduced into
the proposed architecture of the image captioning model. Various different versions of
attention model have been tried out in the past(10). created a very efficiency weightbased
model but the model started missing out on small parts of the image(11) introduced
an attention module which focused on specific actions in the image.(12) came up with
an adaptive attention module which switches between two different forms of attention -
visual attention and language model. It automatically switches between the two methods
and relies on combined result of both of them.
For feature extraction, we have used two pre-trained CNN models namely VGG16
and Resnet50 which are explained below.
3.3. VGG 16
Simonyan et al(13) built the VGG16 convolutional neural network (CNN) model in 2014.
The authors after performing experimentation realised that the CNN was performing
better as they increased the layers and they achieved appreciable results when the layer
count reached 16.
Their conclusions were based on their entry to the ImageNet competition. The model
had an appreciable test accuracy of 92.7 percent. Stride 2 MaxPool layers are used to
supplement the convolutional layers. These layers are then connected to FC (fullyconnected)
layers that process and classify the data from the convolved output, using the
SoftMax function for activation.
A. Sethi et al. / Image Caption Generator in Hindi Using Attention 103
3.4. Resnet 50
The term "residual network architecture" refers to the architecture of residual networks.
In 2012, the residual learning technique was highlighted using this model at the
ILSVRC2012 classification challenge.
Instead of connecting layers and learning features, residual learning involves the
learning of residuals, which is accomplished through the use of shortcut connections.
That is, the nth layer's input is connected to the (n+x) th layer's input.
In parallel to feature extraction, the reference captions are processed using LSTM,
explained below
3.5. LSTM (Long Short-Term Memory)
Recurrent neural networks (RNNs) are neural networks that repeat themselves. RNNs
have a difficulty with long-term dependencies, which LSTM networks were designed to
solve.
LSTMs differ from more standard feedforward neural networks in that they have
feedback connections. This property allows LSTMs to handle whole data sequences
(such as time series) without having to deal with each point individually.
3.6. Attention Mechanism
The attention mechanism was proposed to address the performance bottleneck of
conventional encoder-decoder architectures, achieving significant improvements over
the conventional approach.
The goal of this study is to create Hindi captions for images. There has been very
less research on Hindi language and no research has been done on attention model along
with encoder decoded framework. As a result, this work investigates implementing an
encoder-decoder model in which picture features are encoded using pretrained CNN and
text features are encoded using LSTM-RNN.
4. EXPERIMENTAL WORK
For the training and testing purpose, Flickr 8k Dataset has been used. It consists of 8000
different images along with 5 different captions for each images describing the events in
the image in detail. The dataset contains all the captions in the original Flickr8k dataset
but in our regional language i.e., Hindi. The Dataset has a split of 6000:1000:1000 for
training, validation and testing purposes.
For checking the efficiency of the models, we have used Bleu scores.
We have compared multiple CNNs with the presence and absence of attention
module to evaluate and compare the performance of different models constructed. The
results obtained were at par and in some models even better than the existing work in
English language of Image captioning.
BLEU or the Bilingual Evaluation Understudy(15), is a metric for checking the
similarities between the predicted text using machine learning models and the originally
classified text captions used as training labels. It is a score between 0 and 1. A perfect
score of 1.0 indicates a human like translation and a score of 0 indicates a very poor
translation. The more matches, the better the candidate translation is.
A. 104 Sethi et al. / Image Caption Generator in Hindi Using Attention
The weights are specified in a data structure where each value refers to the
contribution of that index in overall text. You can give a weight of (1,0,0,0) for 1-gram
matches and divide the weights correspondingly for 2,3,4-gram matches.
Table 1. Performance of the Proposed work on Flickr8K dataset
Model BLEU-1 BLEU-2 BLEU-3 BLEU-4
VGG16 0.425 0.183 0.207 0.246
Resnet - 50 0.461 0.192 0.212 0.298
VGG16 with
attention
0.564 0.203 0.290 0.321
Resnet50 with
attention
0.556 0.219 0.304 0.335
The weights for BLEU-1 are (1,0,0,0), for BLEU-2 are (0.5,0.5,0,0), for BLEU-3 are
(0.33,0.33,0.33,0.33) and for BLEU-4 are (0.25,0.25,0.25,0.25) respectively.
The performance of the proposed work on Flickr8K dataset is provided in Table 1.
It is observed that attention based deep features provided better BLEU scores than VGG-
16 and ResNet-50 deep features. From the experiments it is also observed that ResNet50
based captioning model with attention layer performs best with highest BLEU score. A
comparison with the existing works has also been shown in Table 2.
Table 2. Comparison with existing work
Model Dataset BLEU-1 BLEU-2 BLEU-3 BLEU-4
Vinyals et al. (9) Flickr8k 0.630 0.424 0.270 -
A. Rathi (3)
Flickr8k
in Hindi
0.584 0.4 0.27 0.12
Our Model
Flickr8k
in Hindi
0.556 0.219 0.304 0.335
Predicted captions along with actual captions are shown below.