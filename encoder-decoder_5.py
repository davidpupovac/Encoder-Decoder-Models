# Python version: 3.7.7
# Tensorflow-gpu version: 1.14.0
# Keras version: 2.2.4-tf


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
"""

                              Encoder-Decoder 5
                                
"""
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# =============================================================================
# =============================================================================

#                  Stateful encoder-decoder model 

#                          (based on RepeatVector)

# =============================================================================
# =============================================================================


import numpy as np
from random import randint
from numpy import array
from numpy import argmax
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import RepeatVector, TimeDistributed, LSTM, Dense


# -----
# Simulate toy data

np.random.seed(15)

time_steps = 10000  # number of time steps 

# generate a sequence of random integers
def gen_sequence(length, n_unique): # length of sequnce; range of integers from 0 to n_unique-1
	return [randint(1, n_unique-1) for _ in range(length)]

def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

def gen_autocor_seq (size, cor_coef, scale=1):
    series = np.random.normal(scale=scale, size=size)
    for j in range(2,size):
        series[j] = cor_coef * series[j-1] + np.random.normal(scale=scale)
        return series
    
X=gen_autocor_seq(time_steps, 0.75)
X=np.around(X*10,0)
X=X.astype(int)

k_features = np.asarray(np.unique(X, return_counts=True)).shape[1]

X1 = list()
for j in range(time_steps):
    X1.append(X[j])

X=one_hot_encode(X1, k_features)

# redefine time_steps ()
n_steps_in = 5
num_samples = time_steps/n_steps_in
num_samples = np.array(num_samples).astype(int) 

X = np.array(X).reshape(num_samples, n_steps_in, k_features)

n_steps_out= n_steps_in-2
y = X[:,0:n_steps_out, ]

# =============================================================================
# train - stateful

batch_size= 10 # be sure that number of samples that can be divided by the batch size.

model = Sequential()
# default activation function is linear (this is differnence from above)
model.add(LSTM(150, batch_input_shape=[batch_size, n_steps_in, k_features], stateful=True))  
model.add(RepeatVector(n_steps_out))
model.add(LSTM(150, return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(k_features, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Option 1 to fit the model
history = model.fit(X, y, epochs=10, validation_split=0.2, verbose=1, batch_size=10)

# Option 2 to fit the model
for epoch in range(5):
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=10, verbose=2, validation_split=0.2, shuffle=False)
	model.reset_states()

