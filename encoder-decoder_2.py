# Python version: 3.7.7
# Tensorflow-gpu version: 1.14.0
# Keras version: 2.2.4-tf


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
"""

                              Encoder-Decoder 2
                                
"""
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# =============================================================================
# =============================================================================

#                            Teacher forcing 


#             Encoder-decoder models using tfa.seq2seq addon 

# =============================================================================
# =============================================================================

import numpy as np
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

# decode one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]


def gen_dataset(n_in, n_out, cardinality, n_samples):
	X1, X2, y = list(), list(), list()
	for _ in range(n_samples):
		# generate source sequence
		source = gen_sequence(n_in, cardinality)
        
		# define target sequence:
        # take first n elements of the source sequence as the target sequence and reverse them
		target = source[:n_out] # these values will be passed to encoder inputs 
		target.reverse() # the values are targets
        
		# create padded input target sequence
		target_in = [0] + target[:-1] # include the start of sequence value [i.e. 0] in the first time step
		# these values will be passed to decoder inputs)
		
        # store (create all three inputs)
		X1.append(source)
		X2.append(target_in)
		y.append(target)
	return array(X1), array(X2), array(y)

k_features = 40 
n_steps_in = 7 # time steps in 
n_steps_out = 3 # time steps out

X1, X2, y = gen_dataset(n_steps_in, n_steps_out, k_features, 10000)
print(X1.shape, X2.shape, y.shape) #

# -----
# pip install tensorflow_addons
import tensorflow_addons as tfa # requires TensorFlow version >= 2.1.0
import tensorflow as tf

tf.random.set_seed(42)

vocab_size = k_features
embed_size = 10
n_units=512
 
encoder_inputs = tf.keras.layers.Input(shape=[None], dtype=np.int32)
decoder_inputs = tf.keras.layers.Input(shape=[None], dtype=np.int32)
sequence_lengths = tf.keras.layers.Input(shape=[], dtype=np.int32) # for different lenghts

embeddings = tf.keras.layers.Embedding(vocab_size, embed_size)
encoder_embeddings = embeddings(encoder_inputs)
decoder_embeddings = embeddings(decoder_inputs)

encoder = tf.keras.layers.LSTM(n_units, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
encoder_state = [state_h, state_c]

sampler = tfa.seq2seq.sampler.TrainingSampler()

decoder_cell = tf.keras.layers.LSTMCell(n_units)
output_layer = tf.keras.layers.Dense(vocab_size)
decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler,
                                                 output_layer=output_layer)

seq_length_out = np.full([10000], n_steps_out)  # set the lenght of output (it must be vector)

final_outputs, final_state, final_sequence_lengths = decoder(
    decoder_embeddings, initial_state=encoder_state,
    sequence_length=seq_length_out) # set the lenght of output

Y_proba = tf.nn.softmax(final_outputs.rnn_output)

model = tf.keras.models.Model(
    inputs=[encoder_inputs, decoder_inputs, sequence_lengths],
    outputs=[Y_proba])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=['accuracy'])

seq_length_in = np.full([10000], n_steps_in)

history = model.fit([X1, X2, seq_length_in], y, epochs=5)        