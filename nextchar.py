from keras.utils.data_utils import get_file
import pdb
import numpy as np
from keras.utils.np_utils import to_categorical, probas_to_classes
from keras.models import Sequential, load_model
from keras.layers import Flatten,Dense, Input, Embedding, BatchNormalization, Dropout, Convolution1D, MaxPooling1D, LSTM, SimpleRNN, GRU
from keras.optimizers import Adam
import argparse
pdb.set_trace()

#download collected works of Nietzsche
path  = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
print("path is ", path)
text = open(path).read()

print("text len is ", len(text))

parser = argparse.ArgumentParser(description='NLP model parser')
parser.add_argument('--numepoch', type=int,default=1,
                    help='load or fresh trained features')
parser.add_argument('--modeltype', type=str,default="LSTMConv",
                    help='load or fresh trained features')
			
args = parser.parse_args()

num_epoch = args.numepoch
modeltype = args.modeltype

#get the unique chars in the string
# there are 85 k unique chars
unique_chars = sorted(list(set(text)))
print("unique char length is ", len(unique_chars))
#create index to char data structure
idx2char = {i:c for i,c in enumerate(unique_chars)}
char2idx = {c:i for i,c in enumerate(unique_chars)}
print("built indx 2 cha and chr 2 indx tables ");
# create index of the text
# there are 600K chars in this text
text_idx = [char2idx[cur_char]  for cur_char in text]

# now build the sequence 
cs = 8
# if sequence cs is 3 that means we build 3 letter sequence after
# which the 4th letter is to be predicted.  This is many to 1 sequence prediction
seqlist = []

for n in range(cs):
	seq = [ text_idx[i] for i in range(0+n,len(text_idx)-cs+n)]
	seqlist.append(seq)

# now we need to stack the different list into sequences of 8.
# the np.array will be of shape 8,600683 and you stack on axis 1
# so 8 consecutive characters come next to each other in a sequence
# task is to predict 9th character.

input_seq = np.stack(np.array(seqlist,dtype='int32'),axis=1)


# now lets get output labels
train_labels = np.array([text_idx[i] for i in range(0+cs,len(text_idx))],dtype='int32')

# convert them to categorical so  we can use to train the model
train_labels_c = to_categorical(train_labels)


def simpleSeqToSeqModel():
	model = Sequential()
	model.add(Embedding(len(unique_chars),
                  50,
                  input_length=cs,))
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(100,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(len(unique_chars),activation='softmax'))
	return model

def createConv1DModel():
	model = Sequential()
	model.add(Embedding(len(unique_chars),
                    50,
                    input_length=cs,))
	model.add(Convolution1D(64, 5, border_mode='same',activation='relu'))
	model.add(BatchNormalization())
	#model.add(Dropout(0.2))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(100,activation='relu'))
	model.add(BatchNormalization())
	#model.add(Dropout(0.5))
	model.add(Dense(len(unique_chars),activation='softmax'))
	return model


def createGRUModel():
	print("creating GRU model");
	model = Sequential()
	model.add(Embedding(len(unique_chars),
                    50,
                    input_length=cs,))
	model.add(GRU(100))
	model.add(Dense(len(unique_chars),activation='softmax'))
	return model

def createRNNModel():
	print("creating SimpleRNN model");
	model = Sequential()
	model.add(Embedding(len(unique_chars),
                    50,
                    input_length=cs,))
	model.add(SimpleRNN(100))
	model.add(Dense(len(unique_chars),activation='softmax'))
	return model


def createLSTMConvModel(num_lstm, num_conv):
	print("creating LSTM CONV model");
	model = Sequential()
	model.add(Embedding(len(unique_chars),
                    50,
                    input_length=cs,))
	# note here it means 100 LSTM cells
	model.add(Convolution1D(64, 5, border_mode='same',activation='relu'))
	model.add(BatchNormalization())
	model.add(Convolution1D(128, 5, border_mode='same',activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling1D())
	#model.add(Dropout(0.2))
	#if(num_conv == 2):
	#	model.add(Convolution1D(64, 4, border_mode='same',activation='relu'))
	#	model.add(BatchNormalization())
	#model.add(Dropout(0.2))
	#	model.add(MaxPooling1D())
	model.add(LSTM(100))
	#model.add(LSTM(100,dropout_W=0.2, dropout_U=0.2))
	# facing issue
	#if(num_lstm == 2):
	#	model.add(LSTM(50))
	# next add LSTM specific dropouts
	# W is for input dropout and U is for recurrant dropout
	#model.add(LSTM(100,dropout_W=0.2, dropout_U=0.2))

#	if(batchnorm == True):
#		model.add(BatchNormalization())
#	model.add(Dropout(0.2))
#	model.add(MaxPooling1D())
#	model.add(Flatten())
#	model.add(Dense(100,activation='relu'))
#	if(batchnorm == True):
#		model.add(BatchNormalization())
#	model.add(Dropout(0.5))
	model.add(Dense(len(unique_chars),activation='softmax'))
	return model


def createLSTMModel():
	print("creating LSTM  model");
	model = Sequential()
	model.add(Embedding(len(unique_chars),
                    50,
                    input_length=cs,))
	# note here it means 100 LSTM cells
	model.add(LSTM(100))
	# next add LSTM specific dropouts
	# W is for input dropout and U is for recurrant dropout
	#model.add(LSTM(100,dropout_W=0.2, dropout_U=0.2))

#	if(batchnorm == True):
#		model.add(BatchNormalization())
#	model.add(Dropout(0.2))
#	model.add(MaxPooling1D())
#	model.add(Flatten())
#	model.add(Dense(100,activation='relu'))
#	if(batchnorm == True):
#		model.add(BatchNormalization())
#	model.add(Dropout(0.5))
	model.add(Dense(len(unique_chars),activation='softmax'))
	return model


print("modeltype is",modeltype)
if(modeltype == "simple"):
	print("creating simple model")
	mymodel = simpleSeqToSeqModel()
elif(modeltype == "conv"):
	print("creating conv model")
	mymodel = createConv1DModel()
elif(modeltype == "rnn"):
	print("creating rnn model")
	mymodel = createRNNModel()
elif(modeltype ==  "LSTMConv"):
	print("creating lstm model")
	mymodel = createLSTMConvModel(2,2)
elif(modeltype == "GRU"):
	print("creating gru model")
	mymodel = createGRUModel()
elif(modeltype == "LSTM"):
	mymodel = createLSTMModel()

mymodel.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])

mymodel.fit(input_seq,train_labels_c,batch_size=128,nb_epoch=num_epoch)

	
#question how to get from categorial to real when we do prediction?

print("DONE")
