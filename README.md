[![dep1](https://img.shields.io/badge/Python-3.7.3-brightgreen.svg)](https://www.python.org/)
[![dep1](https://img.shields.io/badge/Tensorflow-2.1-brightgreen.svg)](https://www.tensorflow.org/)
[![dep2](https://img.shields.io/badge/Keras-2.2.4-brightgreen.svg)](https://keras.io/)


# Encoder Decoder Models

The file provide several alternative ways of specifying encoder-decoder model, typically used for neural machine translation (NMT), using Keras and Tensorflow. 
The examples include:  

#### [Teacher forcing approach - general example (without embedding layer)](https://github.com/davidpupovac/Encoder-Decoder-Models/blob/master/encoder-decoder_1.py)
- Long Short-Term Memory (LSTM) 
- Gated Recurrent Unit (GRU) 
#### [Teacher forcing approach - encoder-decoder models using seq2seq addon (with embedding layer)](https://github.com/davidpupovac/Encoder-Decoder-Models/blob/master/encoder-decoder_2.py)
- Long Short-Term Memory (LSTM) 
#### [Teacher forcing approach - short translation example (with embedding layer)](https://github.com/davidpupovac/Encoder-Decoder-Models/blob/master/encoder-decoder_3.py)
- Long Short-Term Memory (LSTM) 
#### [Encoder-decoder models without teacher forcing (based on RepeatVector)](https://github.com/davidpupovac/Encoder-Decoder-Models/blob/master/encoder-decoder_4.py)
- Long Short-Term Memory (LSTM)
- Bidirectional (LSTM)
- Functional api with short (state_h) or long term (state_c) state passed to RepeatVector
#### [Stateful encoder-decoder model (based on RepeatVector)](https://github.com/davidpupovac/Encoder-Decoder-Models/blob/master/encoder-decoder_5.py)
- Long Short-Term Memory (LSTM)

## References:

- https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
- https://www.tensorflow.org/tutorials/text/nmt_with_attention
- https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
- https://machinetalk.org/2019/03/29/neural-machine-translation-with-attention-mechanism/

