
# NLP
# SENTIMENT.py
* this project is a playground, where i am just trying out things
* This program pics up files from imdb reviews and provides sentiment whether the sentence is positive or negative.
* While training, tried from simple to more advanced models to improve accuracy.
* In this program there is option to specify different type of models to create and train(simple,lstm,conv) etc. Also 
after each training the model is saved so in future it can be loaded and further trained.
* Reduced the sequence length to 500 words(chose top 5000 high frequency words) and padded shorter sequeunces with zeros.
* Replaced words with frequency lesser than 5000 with a value of 5000(higher frequency words have values from 0 to 4999)
* Tried one-layer SIMPLE network without batchnormlization and got 80% accuracy
* Added batchnormalization layer and improved accuracy to  83%
* Added dropout inaddition to batchnorm layer and improved accuracy to 85%
* Tried Convolution1D layer + batchnorm + maxpooling + dropout improved the result to 87% but not stable in 2 epochs.
* Tried LSTM layer. Training was more slower than other networks (slower than Simple,Convolution networks previously tried)
  accuracy went upto 86% after 4 epochs. when i added L2 regularizer it started of with 50% accuracy in 2 iterations. didn't try more
* Tried three Convolution1D parallel layers and concatenated with KERAS functional API and then added LSTM layer on top. Improved accuracy to got 87% accuracy in 2 epochs

# NEXTCHAR.py (MANY TO 1 NLP sequence prediction)
* This program predicts next character in a sequence of 8 characters(many to 1 sequence example)based on a sequence text of 600K characters.
* The model can be trained with option of different modeltypes(simpleRNN,conv,convlstm,rnn) etc.
* From the given text samples are created of length 8 characters and label (next character) is created for each sample
* Different models in increasing complexity are created to improve accuracy and the training log is as follows.
  * Simple FC model got 2.3 loss accuracy: 34%
  * ConvModel got 1.97 loss acc: 0.4250 (starts with accuracy of 0.11)
  * 100 LSTM got 1.89 loss acc: 0.4424 (betterthan conv1d) ran slow
  * with 100 SimpleRNN units got 2.11 loss and accuracy of 0.3989 ran faster
  * with 100 GRU  1.94 loss and 0.4320 accuracy
  * with Conv1D and LSTM 3 epochs. Loss:1.75 and acc:0.4716
  * with Conv1D(2) and LSTM  1 epoch: 1.97 and acc: 0.42
  * with Conv1D(2) and LSTM 12 epoch on 8 sequence to next sequence
   determination, got loss of 1.6 and accuracy of :0.52
 * Here the accuracy seem to lower, best got 52 accuracy with Conv+LSTM model. Reason is more training is needed
 * Also sample size may be small.
 
 # NEXTCHARS.py (MANY to MANY NLP sequence prediction)
 * This program predicts next character for every input character. So the network is trained with 8 character sequence to predict next 8 character sequence.
 * Used ConvLSTM network. Important is that in keras, in LSTM network need to specify return sequences to true, so that for each character in the sequence, LSTM generates one output.
 * Also important to Create for the output classification layer, TimeDistributed Layer so that network generates 8 different output softmax classifications.
 * Fixed the embedding layer to have length vocab_size of 85. Trained to 93%
    accuracy with 8 epochs
 * Also tried stateful model. In stateful model, for the next batch the previous LSTM hidden state is used.
   Also in stateful model you need to have the number of training inputs as a clean multiple of the batch size used(cannot be fraction). In embedding layer batch size must be specified. In the LSTM layer stateful parameter should be set to true. Trained to around the same accuracy(over 90%) as for the non stateful model

   

