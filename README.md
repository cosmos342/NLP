# NLP
# SENTIMENT.py
* This program pics up files from imdb reviews and provides sentiment weather the sentence is positive or negative.
* While traing tried from simple to more advanced models to improve accuracy.
* Reduced the sequence length to 500 words(chose top 5000 high frequency words) and padded shorter sequeunces with zeros.
* Replaced words with frequency lesser than 5000.
* Tried on one-layer SIMPLE network without batchnormlization and got 80% accuracy
* Added batch norm layer and improved accuracy to  83%
* Added dropout inaddition to batchnorm layer and improved accuracy to 85%
* Tried Convolution1D layer + batchnorm + maxpooling + dropout improved the result to 87% but not stablein 2 epochs.
* Tried LSTM layer. Training was more slower than other networks (like Simple,Convolution networks previously tried)
  accuracy went upto 86% after 4 epochs. when i added l2 regularizer it started of with 50% accuracy in 2 iterations. didn't try more
* Tried three Convolution1D parallel layers and concatedated with keras functional API and then added LSTM layer on top. Improved accuracy to got 87% accuracy in 2 epochs

# NEXTCHAR.py
* This program predicts next character in a sequence of 8 characters based on a sequence text of 600K characters.
* From the given text samples are created of length 8 characters and label (next character) is created for each sample
* Different models in increasing complexity are created to improve accuracy and the training log is as follows.
* simple one layer without batch norm 80%
1.with Simple FC model got 2.3 loss acc?
2. with ConvModel got 1.97 loss acc: 0.4250 (starts with accuracy of 0.11)
3. with 100 LSTM got 1.89 loss acc: 0.4424 (betterthan conv1d) ran slow
4. with 100 SimpleRNN units got 2.11 loss and accuracy of 0.3989 ran faster
5. with 100 GRU  1.94 loss and 0.4320 accuracy
6. with Conv1D and LSTM 3 epochs. Loss:1.75 and acc:0.4716
7. with Conv1D(2) and LSTM  1 epoch: 1.97 and acc: 0.42
8. with Conv1D(2) and LSTM 12 epoch on 8 sequence to next sequence
   determination, got loss of 1.6 and accuracy of :0.52

