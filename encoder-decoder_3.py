# Python version: 3.7.7
# Tensorflow-gpu version: 1.14.0
# Keras version: 2.2.4-tf


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
"""

                              Encoder-Decoder 3
                                
"""
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# =============================================================================
# =============================================================================

#                            Teacher forcing 


#             Ttranslation example with embedding layer

# =============================================================================
# =============================================================================

import tensorflow as tf
import numpy as np
import unicodedata
import re

# -----
# you got message:
# Fail to find the dnn implementation 
# when running:  encoder_output, en_state_h, en_state_c = encoder(source_input, initial_state)
# first run the following:
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# -----
# Simulate toy data

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

# The decoder, on the other hand, requires two versions  of destination language’s sequences:
# one for inputs (with <start> tokens) and
# one for targets (loss computation) (with <end> tokens).
 
# -----
# English data preprocessing

# Tokenize the data, i.e. convert the raw strings into integer sequences
en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

# and create vocabulary
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

# Inside the encoder, there are an embedding layer and an RNN layer (vanilla RNN or LSTM or GRU).
# At every forward pass, it takes in a batch of sequences and initial states and
# returns output sequences as well as final states:

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size): 
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):  
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)

        return output, state_h, state_c

    def init_states(self, batch_size): 
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))

# Without attention mechanism, the decoder is basically the same as the encoder, 
# except that it has a Dense layer to map RNN’s outputs into vocabulary space:
    
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

# -----
# Checks
source_input = tf.constant([[1, 3, 5, 7, 2, 0, 0, 0]]) # fake input sequence with some zero padding
initial_state = encoder.init_states(1) # initial states of zeros
encoder_output, en_state_h, en_state_c = encoder(source_input, initial_state)

target_input = tf.constant([[1, 4, 6, 9, 2, 0, 0]]) # fake target sequence with some zero padding
decoder_output, de_state_h, de_state_c = decoder(target_input, (en_state_h, en_state_c))

print('Source sequences shape', source_input.shape)
print('Encoder outputs shape', encoder_output.shape)
print('Encoder state_h shape', en_state_h.shape)
print('Encoder state_c shape', en_state_c.shape)

print('\nDestination vocab size', fr_vocab_size)
print('Destination sequences shape', target_input.shape)
print('Decoder outputs shape', decoder_output.shape)
print('Decoder state_h shape', de_state_h.shape)
print('Decoder state_c shape', de_state_c.shape)

# -----------------------------------------------------------------------------
# Define a loss function 

# Since we padded zeros into the sequences, do not take zeros into account when computing the loss:
def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
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

# For demonstration  purposes select a random sentence from training data set
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
