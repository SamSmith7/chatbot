import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from keras.models import Sequential
from keras.layers import Activation, Dense
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle

import json

with open('./intents.json') as json_data:
    intents = json.load(json_data)

# Extract lists of words, classes and documents from the intents file

words = []
classes = []
documents = []
ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# Covert bag of words and classes into tensors of numbers for tensorflow

training = []
output_empty = np.zeros(len(classes))

word_count = len(words)

for doc in documents:
    bag = np.zeros(word_count) # init bag of words
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    for i, w in enumerate(words):
        if w in pattern_words:
            bag[i] = 1

    output_row = np.array(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.asarray(training)

# Create training and test sets
train_x = np.stack(training[:, 0], axis=0)
train_y = np.stack(training[:, 1], axis=0)
print(train_x.shape)
print(train_y.shape)
# Train the model
tf.reset_default_graph()

# Implement the model using keras rather than tflearn
model = Sequential([
    Dense(8, input_dim=len(train_x[0])),
    Dense(9, activation='softmax')
])

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=1000, batch_size=8)
model.save('model.keras')

# net = tflearn.input_data(shape=[None, len(train_x[0])])
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
# net = tflearn.regression(net)
#
# model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
#
# model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
# model.save('model.tflearn')

pickle.dump( {'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open( 'training_data', 'wb' ) )
