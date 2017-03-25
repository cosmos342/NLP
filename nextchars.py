from keras.utils.data_utils import get_file
import pdb
import numpy as np
from keras.utils.np_utils import to_categorical, probas_to_classes
from keras.models import Sequential, load_model
from keras.layers import Flatten,Dense, Input, Embedding, BatchNormalization, Dropout, Convolution1D, MaxPooling1D, LSTM, SimpleRNN, GRU, TimeDistributed
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
parser.add_argument('--stateful', type=bool,default=False,
                    help='load or fresh trained features')
			
args = parser.parse_args()

num_epoch = args.numepoch
stateful = args.stateful

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
bs = 128
# if sequence cs is 3 that means we build 3 letter sequence after
# which the 4th letter is to be predicted.  This is many to 1 sequence prediction

# build the input sequeunce it is 1st 8th character like that
# then 2nd 9th so on
seq = [ [text_idx[i] for i in range(0+n,len(text_idx)-cs+n,cs)]  for n in range(cs) ]

# this is 2nd and 9th 3rd  10th etc
out_seq = [[text_idx[i] for i in range(1+n,len(text_idx)-cs+n,cs)] for n in range(cs)]

#xs = [ np.stack(c[:-2]) for c in seq]
# convert to numpy array (remove last 2 elements, just for ease
# also for  last element we don't have next label anyway
xs = [np.array(c[:-2]) for c in seq]
#out_seq = [np.stack(c[:-2]) for c in out_seq]
ys = [np.stack(c[:-2]) for c in out_seq]
# this stacks so that first sample becomes 1 to 8 elements 
# so the label sample next in sequence becomes 2 end to 9 elements
xs = np.stack(xs,axis=1)
ys = np.stack(ys,axis=1)

# convert the labels to one hot categories so we can use as labels

# NOTE: you can keep the labels as it is and give as labels
# then you use objective as "sparse_categorical_entropy" 
# in prediction it produces a tensor of integers and you pick the maximum
# just like the categorical_entropy(which produces probs). 
train_labels_c = [to_categorical(y,nb_classes=85) for y in ys] 
# convert to numpy array
train_labels_c = np.array(train_labels_c)

# now we need to stack the different list into sequences of 3.
#input_seq = np.stack((np.array(seqlist[0],dtype='int32'),np.array(seqlist[1],dtype='int32'),np.array(seqlist[2],dtype='int32')),axis=-1)

# now we need to stack the different list into sequences of 8.


# now lets get output labels
#train_labels = np.array([text_idx[i] for i in range(0+cs,len(text_idx))],dtype='int32')

# convert them to categorical so  we can use to train the model
#train_labels_c = to_categorical(train_labels)
def createLSTMConvModel():
	print("creating LSTM CONV model");
	model = Sequential()

	if(stateful is True):
		print("add stateful embedding layer")
		model.add(Embedding(len(unique_chars),
                	50,
                	input_length=cs,batch_input_shape=(bs,cs)))
	else:
		print("add non stateful embedding layer")
		model.add(Embedding(len(unique_chars),
               	     			50,
                    			input_length=cs))

	# note here it means 100 LSTM cells
	model.add(Convolution1D(64, 5, border_mode='same',activation='relu'))
	model.add(BatchNormalization())
	model.add(Convolution1D(128, 5, border_mode='same',activation='relu'))
	model.add(BatchNormalization())
	#max pooling reduced the sequence so cannot use for m sequence to m sequence
	#model.add(MaxPooling1D())

	# returning sequences is needed so that for each character returns
        # output so next characters can be determined. Not needed for 
        # for a problem where after a sequence just one output is needed
	if(stateful is True):
		print("add stateful LSTM layer")
		model.add(LSTM(100,return_sequences=True,stateful=True))
	else:
		print("add stateless LSTM layer")
		model.add(LSTM(100,return_sequences=True))
	# this is if you want to introduce dropout at the
        # the LSTM layer. W is for input dropout and U is for recurrant dropout
	#model.add(LSTM(100,dropout_W=0.2, dropout_U=0.2))

#       Because of TimeDistributed you have 8 outputs for 8 input sequences
#       each of is a softmax of 85 characters
	model.add(TimeDistributed(Dense(len(unique_chars),activation='softmax')))
	return model


model = createLSTMConvModel()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])

if(stateful is True):
	slen = len(xs)/bs*bs
	model.fit(xs[:slen],train_labels_c[:slen],batch_size=128,nb_epoch=num_epoch)
else:
	model.fit(xs,train_labels_c,batch_size=128,nb_epoch=1)

	# NOW DO THE PREDICTION. 
	# TEST THE INPUT
	sample = np.array(xs[0])
	# For purposes of predict input has to be (None,8) shape
	sample = sample.reshape(1,8)
	y = model.predict(sample)
	inp = [idx2char[c] for c in xs[0]]
	# argmax provides the index that has the max value
	# Since it is one hot encoding we can use that to get the character
	# from the index to character table
	y = [idx2char[np.argmax(c)] for c in y[0]]
	print("input", inp)
	print("output",y)
print("DONE")
