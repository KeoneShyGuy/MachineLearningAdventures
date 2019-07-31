# https://www.tensorflow.org/tutorials/keras/basic_text_classification

# from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras as krs

import numpy as np

print(tf.__version__)

imdb = krs.datasets.imdb
(train_data, train_labals), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labals)))

print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

# How the fuck? Research later
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))

train_data = krs.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"],
                                                        padding='post', maxlen=256)

test_data = krs.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"],
                                                     padding='post', maxlen=256)

print(len(test_data[0]), len(train_data[1]))
print(train_data[0])

vocab_size = 10000

model = krs.Sequential()
model.add(krs.layers.Embedding(vocab_size, 16))
model.add(krs.layers.GlobalAveragePooling1D())
model.add(krs.layers.Dense(16, activation=tf.nn.relu))
model.add(krs.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()