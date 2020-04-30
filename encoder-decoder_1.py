# Python version: 3.7.7
# Tensorflow-gpu version: 1.14.0
# Keras version: 2.2.4-tf


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
"""

                              Encoder-Decoder 1
                                
"""
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# =============================================================================
# =============================================================================

#                    Teacher forcing - general example

# =============================================================================
# =============================================================================

# -----
# Simulate toy data 1

from random import randint
from numpy import array
from numpy import argmax
from keras.utils import to_categorical

# Data description: 
# Input is a sequence of n_in numbers. Target is first n_out elements
# of the input sequence in the reversed order 

# generate a sequence of random integers
def gen_sequence(length, n_unique): # length of sequnce; range of integers from 0 to n_unique-1
	return [randint(1, n_unique-1) for _ in range(length)]
 
# prepare data for the LSTM
def gen_dataset(n_in, n_out, cardinality, n_samples):
	X1, X2, y = list(), list(), list()
	for _ in range(n_samples):
		# generate source sequence
		source = gen_sequence(n_in, cardinality)
        
		# define target sequence:
		target = source[:n_out] # these values will be passed to encoder inputs 
		target.reverse() # the reverse values are targets
        
		# create padded input target sequence
		target_in = [0] + target[:-1] # include the start of sequence value [i.e. 0] in the first time step
        
        # encode (turn to categorical)
		src_encoded = to_categorical([source], num_classes=cardinality)
		tar_encoded = to_categorical([target], num_classes=cardinality)
		tar2_encoded = to_categorical([target_in], num_classes=cardinality)
		
        # store (create all three inputs)
		X1.append(src_encoded)
		X2.append(tar2_encoded)
		y.append(tar_encoded)
	return array(X1), array(X2), array(y)

# decode one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

     
# configure problem
k_features = 40 + 1 
# add 1 to cardinality to ensure the one-hot encoding is large enough to include start with zero 
n_steps_in = 7 # time steps in 
n_steps_out = 3 # time steps out


# View a sample = generate ONE source and target sequence
X1, X2, y = gen_dataset(n_steps_in, n_steps_out, k_features, 1)
print(X1.shape, X2.shape, y.shape) # this must be reshaped -  update to Keras to 2.1.2, it fixes bugs with to_categorical()
X1 = array(X1).reshape(1, n_steps_in, k_features)
X2 = array(X2).reshape(1, n_steps_out, k_features)
y = array(y).reshape(1, n_steps_out, k_features)
# dataset consists of X1: input sequence, X2: output sequence which starts with 0, y: actual output sequence
print('X1=%s, X2=%s, y=%s' % (one_hot_decode(X1[0]), one_hot_decode(X2[0]), one_hot_decode(y[0])))


# generate WHOLE training dataset of sample size=100,000
X1, X2, y = gen_dataset(n_steps_in, n_steps_out, k_features, 100000)
print(X1.shape,X2.shape,y.shape)
X1 = array(X1).reshape(100000, n_steps_in, k_features)  # create input
X2 = array(X2).reshape(100000, n_steps_out, k_features) # create input with start token
y = array(y).reshape(100000, n_steps_out, k_features)   # create output 


# -----------------------------------------------------------------------------   
# -----------------------------------------------------------------------------
# train 1 -  LTSM Encoder-Decoder Model
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

from keras.models import Model
from keras.layers import Input, LSTM, Dense

# n_input:  Number of values, words, or characters possible for each time step (length of dictionary/vocabulary).
# n_output: Number of values, words, or characters possible for each time step (length of dictionary/vocabulary).
# n_units:  Number of cells to create in the encoder and decoder models, e.g. 128 or 256.

n_input=k_features 
n_output=k_features
n_units=128 # the encoder and decoder layers must have the same number of RNN cells

# RUN STEP BY STEP:
    
# -----
# Step 1 - define encoder-decoder model

# --- Define training encoder
encoder_inputs = Input(shape=(None, n_input)) # size of vocabulary to be translated, number of one hot encoded features
encoder = LSTM(n_units, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs) # save outputs and states
# encoder states
encoder_states = [state_h, state_c] # save only states as a list

# --- Define training decoder 
decoder_inputs = Input(shape=(None, n_output))  # size of vocabulary of translation, number of one hot encoded features
decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
# save outputs and states using encoder states as initial states
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states) 

decoder_dense = Dense(n_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs) 

# Define the model that will turn encoder_inputs & decoder_inputs into decoder_outputs
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print(model.summary())

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit([X1, X2], y, batch_size=32, epochs=4, validation_split=0.2)

# -----
# Step 2 - define inference encoder 

encoder_model = Model(encoder_inputs, encoder_states) 
print(encoder_model.summary())

# -----
# Step 3 - define inference decoder

decoder_state_input_h = Input(shape=(n_units,)) # define input_h with n_units 
decoder_state_input_c = Input(shape=(n_units,)) # define input_c with n_units 
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c] 

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c] # save decoder states 
decoder_outputs = decoder_dense(decoder_outputs) 
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

print(decoder_model.summary())

# --------
# prediction 

# infenc: Encoder model used when making a prediction for a new source sequence.
# infdec: Decoder model use when making a prediction for a new source sequence.
# source:Encoded source sequence.
# n_steps: Number of time steps in the target sequence.
# cardinality: The number of candidate values, words, or characters (features) for each time step

# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return array(output)

# -----
# predict and check some examples
for _ in range(10):
    X1_test, X2_test, y_test = gen_dataset(n_steps_in, n_steps_out, k_features, 1)
    X1_test = array(X1_test).reshape(1, n_steps_in, k_features)
    X2_test = array(X2_test).reshape(1, n_steps_out, k_features)
    y_test = array(y_test).reshape(1, n_steps_out, k_features)
    y_hat = predict_sequence(encoder_model, decoder_model, X1_test, n_steps_out, k_features)
    print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1_test[0]), one_hot_decode(y_test[0]), one_hot_decode(y_hat)))
    

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# train 2 -  GRU Encoder-Decoder Model
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


from keras.models import Model
from keras.layers import Input, GRU, Dense

n_input=k_features 
n_output=k_features
n_units=128

# -----
# Step 1 - define encoder-decoder 

# --- Define training encoder
encoder_inputs = Input(shape=(None, n_input))
encoder = GRU(n_units, return_state=True)
encoder_outputs, state_h = encoder(encoder_inputs) # you have one hidden state

# --- Define training decoder
decoder_inputs = Input(shape=(None, n_output))
decoder_gru = GRU(n_units, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=state_h)
decoder_dense = Dense(n_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print(model.summary())

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit([X1, X2], y, batch_size=32, epochs=4, validation_split=0.2)

# -----
# Step 2 - define inference encoder 

encoder_model = Model(encoder_inputs, state_h) 
print(encoder_model.summary())

# -----
# Step 3 - define inference decoder

decoder_state_input_h = Input(shape=(n_units,)) # define input_h with n_units 
decoder_states_inputs = [decoder_state_input_h] # put inputs together
decoder_outputs, state_h = decoder_gru(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h] # save decoder states 
decoder_outputs = decoder_dense(decoder_outputs) 
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
print(decoder_model.summary())

# --------
# prediction 
 
# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode state = infenc.predict(source)
	state = [infenc.predict(source)]
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps_out):
		# predict next char
		yhat, h = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h]
		# update target sequence
		target_seq = yhat
	return array(output)

# -----
# predict and check some examples
for _ in range(10):
    X1_test, X2_test, y_test = gen_dataset(n_steps_in, n_steps_out, k_features, 1)
    X1_test = array(X1_test).reshape(1, n_steps_in, k_features)
    X2_test = array(X2_test).reshape(1, n_steps_out, k_features)
    y_test = array(y_test).reshape(1, n_steps_out, k_features)
    y_hat = predict_sequence(encoder_model, decoder_model, X1_test, n_steps_out, k_features)
    print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1_test[0]), one_hot_decode(y_test[0]), one_hot_decode(y_hat)))