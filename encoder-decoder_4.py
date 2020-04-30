# Python version: 3.7.7
# Tensorflow-gpu version: 1.14.0
# Keras version: 2.2.4-tf


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
"""

                              Encoder-Decoder 4
                                
"""
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# =============================================================================
# =============================================================================

#              Encoder-decoder models without teacher forcing  

#                          (based on RepeatVector)

# =============================================================================
# =============================================================================

import numpy as np
from random import randint
from numpy import array
from numpy import argmax
from keras.utils import to_categorical

# -----
# Simulate toy data

# generate a sequence of random integers
def gen_sequence(length, n_unique): # length of sequnce; range of integers from 0 to n_unique-1
	return [randint(1, n_unique-1) for _ in range(length)]

# decode one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# prepare data for the LSTM
def gen_in_out(n_in, n_out, n_unique):
	# generate random sequence
	sequence_in = gen_sequence(n_in, n_unique)
	sequence_out = sequence_in[:n_out] 
	# one hot encode
	X = one_hot_encode(sequence_in, n_unique)
	y = one_hot_encode(sequence_out, n_unique)
	# reshape as 3D
	X = X.reshape((1, X.shape[0], X.shape[1]))
	y = y.reshape((1, y.shape[0], y.shape[1]))
	return X,y

# View a sample - generate ONE source and target sequence
k_features = 45 #  
n_steps_in = 6 # time steps in 
n_steps_out = 3 # time steps out
X, y = gen_in_out(n_steps_in, n_steps_out, k_features)
print(X.shape, y.shape)
print('X=%s, y=%s' % (one_hot_decode(X[0]), one_hot_decode(y[0])))

# generate WHOLE training dataset of sample size=100,000
def gen_data(n_steps_in, n_steps_out, k_features, n_samples):
    X, y = list(), list()
    for _ in range(n_samples):
        X_tmp, y_tmp = gen_in_out(n_steps_in, n_steps_out, k_features)
        # store (create all inputs)
        X.append(X_tmp)
        y.append(y_tmp)
    return array(X), array(y)

X, y = gen_data(n_steps_in, n_steps_out, k_features, 100000)

X = array(X).reshape(100000, n_steps_in, k_features)
y = array(y).reshape(100000, n_steps_out, k_features)

# =============================================================================
# train 1 -  LSTM

from keras.models import Sequential
from keras.layers import RepeatVector, TimeDistributed, LSTM, Dense

model = Sequential()

# encoder layer
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, k_features)))

# -------
# intermediate encoder layer

# The encoder will produce a 2-dimensional matrix of hidden state - LTSM output shape: (batch_size, num_cells)
# The decoder is an LSTM layer expects a 3D input of [batch_size, time steps, features]

# RepeatVector layer simply repeats the provided 2D input multiple times to create a 3D output
model.add(RepeatVector(n_steps_out)) 

# decoder layer
model.add(LSTM(100, activation='relu', return_sequences=True)) #  select appropriate activation function
# Since the output is in the form of a time-step, which is a 3D format, the return_sequences has been set True.

model.add(TimeDistributed(Dense(k_features, activation='softmax')))
# The TimeDistributed applies the same Dense layer (same weights) to 
# the LSTMs outputs for one time step at a time.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(X, y, epochs=5, validation_split=0.2, verbose=1, batch_size=32)

# predict and check some examples
for _ in range(10):
    X_test, y_test = gen_data(n_steps_in, n_steps_out, k_features, 1)
    X_test = array(X_test).reshape(1, n_steps_in, k_features)
    y_test = array(y_test).reshape(1, n_steps_out, k_features)
    y_hat = np.around(model.predict(X_test, verbose=0))
    print('X=%s y=%s, yhat=%s' % (one_hot_decode(X_test[0]), one_hot_decode(y_test[0]), one_hot_decode(y_hat[0])))

# -------
# train 1b  LSTM - stacked

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, k_features), return_sequences=True))
model.add(LSTM(50, activation='relu'))
model.add(RepeatVector(n_steps_out)) 
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True)) 
model.add(TimeDistributed(Dense(k_features, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
from keras.utils.vis_utils import plot_model # print model
plot_model(model, to_file='C:\\Users\\david\\Desktop\\model.png', show_shapes=True)

history = model.fit(X, y, epochs=5, validation_split=0.2, verbose=1, batch_size=32)

# =============================================================================
# train 2  Bidirectional

from keras.layers import Bidirectional

model = Sequential()
model.add(Bidirectional(LSTM(100, activation='relu', input_shape=(n_steps_in, k_features))))
model.add(RepeatVector(n_steps_out))
model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True)))
model.add(TimeDistributed(Dense(1)))
model.add(TimeDistributed(Dense(k_features, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, y, epochs=5, validation_split=0.2, verbose=1, batch_size=32)

# =============================================================================
# train 3 Functional  api with short (state_h) or long term (state_c) state passed to RepeatVector 

import tensorflow.keras as keras
from keras.models import Model
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, RepeatVector, TimeDistributed

sequence_input = Input(shape=(n_steps_in, k_features))
encoder_lstm = LSTM(100, return_sequences= True,  return_state = True)
encoder_outputs, state_h, state_c = encoder_lstm(sequence_input)
repeat = RepeatVector(n_steps_out)(state_c) # so you can send only single state!!!
decoder_lstm = LSTM(100, return_sequences = True)(repeat)
output = TimeDistributed(Dense(k_features, activation='softmax')) (decoder_lstm)

model = keras.models.Model(inputs=[sequence_input], outputs=[output])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X, y, epochs=5, validation_split=0.2, verbose=1, batch_size=20)

# predict and check some examples
for _ in range(10):
    X_test, y_test = gen_data(n_steps_in, n_steps_out, k_features, 1)
    X_test = array(X_test).reshape(1, n_steps_in, k_features)
    y_test = array(y_test).reshape(1, n_steps_out, k_features)
    y_hat = np.around(model.predict(X_test, verbose=0))
    print('X=%s y=%s, yhat=%s' % (one_hot_decode(X_test[0]), one_hot_decode(y_test[0]), one_hot_decode(y_hat[0])))
