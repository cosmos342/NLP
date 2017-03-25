from keras.datasets import imdb
import pdb
import numpy as np
from keras.utils.data_utils import get_file
import pickle
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Flatten,Dense, Input, Embedding, BatchNormalization, Dropout, Convolution1D, MaxPooling1D, LSTM
from keras.engine import Merge, merge
from keras.engine import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import os
import argparse

mpath =  os.getcwd()
model_path = mpath + '/models'

pdb.set_trace()

# do a sequential model with embedding layer
MAX_SEQ_LEN=500
MAX_WORD_FREQ=5000

parser = argparse.ArgumentParser(description='Vision Model parser')
parser.add_argument('--load', type=str,default="load",
                    help='load or fresh trained features')
parser.add_argument('--bn', type=bool,default=False,
                    help='load or fresh trained features')
parser.add_argument('--modeltype', type=str,default="convlstm",
                    help='load or fresh trained features')

args = parser.parse_args()
load = args.load
bn = args.bn
modeltype = args.modeltype




# save model weights
def saveModel(model,mypath,filename):
	model.save(mypath+ filename )
	print(mypath + filename)

# load model with pretrained weights
def loadModel(mypath,filename):
	print("loading nlp model")
	
	model = load_model(mypath+ filename)
	return model


def createSimpleFCModel(batchnorm):
	model = Sequential()
	model.add(Embedding(MAX_WORD_FREQ,
                    32,
                    input_length=MAX_SEQ_LEN,))

	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(100,activation='relu'))
	if(batchnorm == True):
		model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(1,activation='sigmoid'))
	return model

def createConv1DModel(batchnorm):
	model = Sequential()
	model.add(Embedding(MAX_WORD_FREQ,
                    32,
                    input_length=MAX_SEQ_LEN))
	model.add(Convolution1D(64, 5, border_mode='same',activation='relu'))
	if(batchnorm == True):
		model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(100,activation='relu'))
	if(batchnorm == True):
		model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(1,activation='sigmoid'))
	return model

def createFunctionalConvModel(bn):
	# have  same embedding feed into 3 different conv models
	# of size 5,4,3
        # then do dropout
	# maxpooling
        # flatten and concat
        # note each of these return tensors so in the end model is called
	main_input = Input(shape=(MAX_SEQ_LEN,),dtype='int32',name='main_input')
	x = Embedding(output_dim=50, input_dim=MAX_WORD_FREQ, input_length=MAX_SEQ_LEN)(main_input)
	conv1 = Convolution1D(64, 3, border_mode='same',activation='relu',name='conv1')(x)
	if(bn == True):
		conv1 = BatchNormalization()(conv1)
	#conv1 = Dropout(0.2)(conv1)
	conv1 = MaxPooling1D()(conv1)
	
	conv2 = Convolution1D(64, 4, border_mode='same',activation='relu',name='conv2')(x)
	if(bn == True):
		conv2 = BatchNormalization()(conv2)
	#conv2 = Dropout(0.2)(conv2)
	conv2 = MaxPooling1D()(conv2)

	conv3 = Convolution1D(64, 5, border_mode='same',activation='relu',name='conv3')(x)
	if(bn == True):
		conv3 = BatchNormalization()(conv3)
	#conv3 = Dropout(0.2)(conv3)
	conv3 = MaxPooling1D()(conv3)
	
	# if you have layers you can call Merge(captial m)
        # otherwise call small merge as below
        #
	# Merge is a layer.
	# Merge takes layers as input
	# Merge is usually used with Sequential models

	# merge is a function.
	# merge takes tensors as input.
	# merge is a wrapper around Merge.
	# merge is used in Functional API
	# Using Merge:
	# left = Sequential()
	# left.add(...)
	# left.add(...)

	# right = Sequential()
	# right.ad(...)
	# right.add(...)

	# model = Sequential()
	# model.add(Merge([left, right]))
	# model.add(...)
	# using merge:
	# a = Input((10,))
	# b = Dense(10)(a)
	# c = Dense(10)(a)
	# d = merge([b, c])
	# model = Model(a, d)

	mergedL = merge([conv1,conv2,conv3],mode='concat')
	x = Flatten()(mergedL)
	x = Dense(100,activation='relu')(x)
	if(bn == True):
        	x = BatchNormalization()(x)
	#x = Dropout(0.5)(x)
	out = Dense(1,activation='sigmoid')(x)
	return Model(main_input,out)




	


def createLSTMModel(batchnorm):
	model = Sequential()
	model.add(Embedding(MAX_WORD_FREQ,
                    32,
                    input_length=MAX_SEQ_LEN,W_regularizer=l2(0.01)))
	model.add(LSTM(100,W_regularizer=l2(0.01)))

	if(batchnorm == True):
		model.add(BatchNormalization())
#	model.add(Dropout(0.2))
#	model.add(MaxPooling1D())
#	model.add(Flatten())
#	model.add(Dense(100,activation='relu'))
#	if(batchnorm == True):
#		model.add(BatchNormalization())
#	model.add(Dropout(0.5))
	model.add(Dense(1,activation='sigmoid'))
	return model

if(load == "load"):
	model = loadModel(model_path,"/" + modeltype + ("bn" if bn is True else ""))
else:
	if(modeltype == "simple"):
		model = createSimpleFCModel(bn)
		print("done creating simple model")
	elif(modeltype == "conv1d"):
		model = createConv1DModel(bn)
		print("done creating conv1d model")
	elif(modeltype == "lstm"):
		model = createLSTMModel(bn)
		print("done creating lstm model")
	elif(modeltype == "convlstm"):
		model = createFunctionalConvModel(bn)
		print("done creating functional model")

	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# downloads the pickle file and stores
# in ~/.keras/datasets
path = get_file('imdb_full.pkl',
                origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl',
                md5_hash='d091312047c43cf9e4e38fef92437263')

f = open(path, 'rb')
(x_train, labels_train), (x_test, labels_test) = pickle.load(f)

#gets the word index for the database 
# based on the word and frequency. so the is given value of 0
# meaning it is the highest frequency word and so on
word_idx = imdb.get_word_index()
print("total number of words ",len(word_idx))
# the the index to word
# note the index 0 is not used so this can be used for padding
idx2word = {i : v  for v,i in word_idx.iteritems()}
# lets print the first training sentance
print(' '.join(idx2word[i] for i in x_train[0]))
#key = sorted(word_idx,key=word_idx.get)
# lets select top 5000 frequency words and replace
# the remaining ones with the index 5000
m_xtrain = np.array([[val if val < MAX_WORD_FREQ-1 else MAX_WORD_FREQ-1 for val in x ] for x in x_train])
m_xtest = np.array([[val if val < MAX_WORD_FREQ-1 else MAX_WORD_FREQ-1 for val in x ] for x in x_test])

# lets count number of positive and negative sentiments
train_counts = np.bincount(labels_train)
print("positive, negative training samples",train_counts)
test_counts = np.bincount(labels_test)
print("positive, negative testing samples",test_counts)

# lets check the averge min max length of the samples
ourlens =  map(len,m_xtrain)
print("max element,min element and average of list ",max(ourlens),min(ourlens),np.mean(ourlens))

# let us restrict the max of each sequence to 500
m_xtrain = sequence.pad_sequences(m_xtrain,maxlen=MAX_SEQ_LEN)
m_xtest  = sequence.pad_sequences(m_xtest,maxlen=MAX_SEQ_LEN)

# now let us train the network
model.fit(m_xtrain,labels_train,batch_size=128,nb_epoch=1,validation_data=(m_xtest,labels_test))

# now do prediction
print(' '.join(idx2word[i] for i in x_train[0]))
y = model.predict(m_xtrain[0].reshape(1,500))
y = 1 if y > 0.5 else 0
print("result: ", "Positive" if y is 1 else "Negative")

saveModel(model,model_path,"/" + modeltype + ("bn" if bn is True else ""))

#TODO do a predict of case

# now let us
print("DONE")
