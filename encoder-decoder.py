# Python version: 3.7.3
# Tensorflow-gpu version: 2.1
# Keras version: 2.2.4-tf

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
"""

                              Encoder-Decoder 
                                
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


# generate WHOLE training dataset of sample size =100,000
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
n_units=128 # the encoder and decoder layers must have the same number of cells

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

from numpy import array_equal
import numpy as np

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

# =============================================================================    
# =============================================================================

#              Encoder-decoder models using seq2seq addon

# =============================================================================
# =============================================================================

# Example includes embedding layer

# -----
# Simulate toy data 1

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
import tensorflow_addons as tfa # requires TensorFlow version >= 2.1.0
import keras
import tensorflow as tf

tf.random.set_seed(42)

vocab_size = k_features
embed_size = 10
 
encoder_inputs = tf.keras.layers.Input(shape=[None], dtype=np.int32)
decoder_inputs = tf.keras.layers.Input(shape=[None], dtype=np.int32)
sequence_lengths = tf.keras.layers.Input(shape=[], dtype=np.int32)

embeddings = tf.keras.layers.Embedding(vocab_size, embed_size)
encoder_embeddings = embeddings(encoder_inputs)
decoder_embeddings = embeddings(decoder_inputs)

encoder = tf.keras.layers.LSTM(512, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
encoder_state = [state_h, state_c]

sampler = tfa.seq2seq.sampler.TrainingSampler()

decoder_cell = tf.keras.layers.LSTMCell(512)
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

# =============================================================================
# =============================================================================

#          Teacher forcing - translation example with embedding layer

# =============================================================================
# =============================================================================


import tensorflow as tf
import numpy as np
import unicodedata
import re

# -----

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# -----
# Simulate toy data 2

raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),
    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"),
    ("He acted like he owned the place.", "Il s'est comporté comme s'il possédait l'endroit."),
    ("Honesty will pay in the long run.", "L'honnêteté paye à la longue."),
    ("How do we know this isn't a trap?", "Comment savez-vous qu'il ne s'agit pas d'un piège ?"),
    ("I can't believe you're giving up.", "Je n'arrive pas à croire que vous abandonniez."),
)

# -----
# Clean up the raw data

# normalizing strings, filtering unwanted tokens, adding space before punctuation
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s

# Split the data into two separate lists, each containing its own sentences.
raw_data_en, raw_data_fr = list(zip(*raw_data))
raw_data_en, raw_data_fr = list(raw_data_en), list(raw_data_fr)

# Then  apply the functions above and add two special tokens: <start> and <end>:
raw_data_en = [normalize_string(data) for data in raw_data_en]
raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]    

# -----
# English data preprocessing

# Tokenize the data, i.e. convert the raw strings into integer sequences
en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en)
print(en_tokenizer.word_index)

# Converte draw English sentences to integer sequences:
data_en = en_tokenizer.texts_to_sequences(raw_data_en)
# pad zeros so that all sequences have the same length
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en, padding='post')

# -----
# French  data preprocessing

fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)

data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in) # decoder data in
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in, padding='post')

data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out) # decoder data out
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out, padding='post')

# -----
# create an instance of tf.data.Dataset:
dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(20).batch(5)

# -----------------------------------------------------------------------------
# Define decoder and encoder

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size): # constructor - creates layers
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):  # create class method - for outputs and states
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)

        return output, state_h, state_c

    def init_states(self, batch_size): # create class methods - for inital state (zero matrix)
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))

    
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Decoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size) # add dense layee

    def call(self, sequence, state):
        embed = self.embedding(sequence)
        lstm_out, state_h, state_c = self.lstm(embed, state) 
        # the final states of the encoder will act as the initial states of the decoder. 
        logits = self.dense(lstm_out)

        return logits, state_h, state_c

# -----
# model basics

EMBEDDING_SIZE = 32
LSTM_SIZE = 64

en_vocab_size = len(en_tokenizer.word_index) + 1
encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, LSTM_SIZE) # define output of encoder

fr_vocab_size = len(fr_tokenizer.word_index) + 1
decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, LSTM_SIZE) # define output of decoder

# -----------------------------------------------------------------------------
# Define a loss function 

# Since we padded zeros into the sequences, do not take zeros into account when computing the loss:
def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0)) # masking
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss 

# -----
# Define optimizer
optimizer = tf.keras.optimizers.Adam()

# -----
# Define training step

def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_states = en_states

        de_outputs = decoder(target_seq_in, de_states)
        logits = de_outputs[0]
        loss = loss_func(target_seq_out, logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss

# -----
# Method for inference purpose

# Basically a forward pass, but instead of target sequences, we will feed in the <start> token.
def predict():
    test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

    de_input = tf.constant([[fr_tokenizer.word_index['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []

    while True:
        de_output, de_state_h, de_state_c = decoder(
            de_input, (de_state_h, de_state_c))
        de_input = tf.argmax(de_output, -1)
        out_words.append(fr_tokenizer.index_word[de_input.numpy()[0][0]])

        if out_words[-1] == '<end>' or len(out_words) >= 20:
            break

    print(' '.join(out_words))
    
# -----
# Training loop

NUM_EPOCHS = 250
BATCH_SIZE = 5

for e in range(NUM_EPOCHS):
    en_initial_states = encoder.init_states(BATCH_SIZE)

    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in,
                          target_seq_out, en_initial_states)

    print('Epoch {} Loss {:.4f}'.format(e + 1, loss.numpy()))
    
    try:
        predict()
    except Exception:
      continue

# -----
# predict 

# for demonstration  purposes select a random sentence from training data set
test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
print(test_source_text)
test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])

en_initial_states = encoder.init_states(1)
en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

de_input = tf.constant([[fr_tokenizer.word_index['<start>']]])
de_state_h, de_state_c = en_outputs[1:]
out_words = []

while True:
    de_output, de_state_h, de_state_c = decoder(de_input, (de_state_h, de_state_c))
    de_input = tf.argmax(de_output, -1)
    out_words.append(fr_tokenizer.index_word[de_input.numpy()[0][0]])
    
    if out_words[-1] == '<end>' or len(out_words) >= 20:
            break
    print(' '.join(out_words))    

# =============================================================================    
# =============================================================================

#              Encoder-decoder models without teacher forcing  

#                          (based on RepeatVector)

# =============================================================================
# =============================================================================

# -----
# Simulate toy data 3

import numpy as np
from random import randint
from numpy import array
from numpy import argmax
from keras.utils import to_categorical

# generate a sequence of random integers
def gen_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]
 
# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

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

k_features = 40 
n_steps_in = 7 # time steps in 
n_steps_out = 3 # time steps out

# generate WHOLE training dataset of sample size =100,000
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
# The repeat vector only repeats the encoder output and has no parameters to train.

# decoder layer
model.add(LSTM(100, activation='relu', return_sequences=True)) 
# Since the output is in the form of a time-step, which is a 3D format, the return_sequences has been set True.

model.add(TimeDistributed(Dense(k_features, activation='softmax')))
# The TimeDistributed applies the same Dense layer (same weights) to the LSTMs outputs for 
# one time step at a time.

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
plot_model(model, to_file='~Desktop\\model.png', show_shapes=True)

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
# train 3  Stateful

# (this model will not work well on this data)

from keras.models import Sequential
from keras.layers import RepeatVector, TimeDistributed, LSTM, Dense

batch_size = 100 

model = Sequential()
model.add(LSTM(50, activation='relu', 
               batch_input_shape=[batch_size, n_steps_in, k_features], return_sequences=True))
model.add(RepeatVector(n_steps_out)) 
model.add(LSTM(50, activation='relu', return_sequences=True, stateful=True)) 
model.add(TimeDistributed(Dense(k_features, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

for epoch in range(5):
	model.fit(X, y, epochs=1, batch_size=100, verbose=1,
                    validation_split=0.2, shuffle=False)
	model.reset_states()

# =============================================================================
# train 4  

# functional  api with short term (state_h) or long term (state_c) state passed to RepeatVector 

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

# =============================================================================
# =============================================================================

#                  Stateful encoder-decoder model 

# =============================================================================
# =============================================================================

# -----
# Simulate toy data 4

import numpy as np
np.random.seed(15)

time_steps = 10000  # number of time steps 

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
# train 1 - stateful, categorical

batch_size= 10 # be sure that number of samples that can be divided by the batch size.

model = Sequential()
model.add(LSTM(150, batch_input_shape=[batch_size, n_steps_in, k_features], stateful=True)) # batch_input_shape=(batch_size, time_steps, vocabulary) 
model.add(RepeatVector(n_steps_out))
model.add(LSTM(150, return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(k_features, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Option 1 to fit the model
history = model.fit(X, y, epochs=100, validation_split=0.2, verbose=1, batch_size=10)

# Option 2 to fit the model
for epoch in range(5):
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=10, verbose=2, validation_split=0.2, shuffle=False)
	model.reset_states()  
