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
