from keras.models import Sequential
from keras.layers import Activation, Dense
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import tflearn
import tensorflow as tf
import pickle
import re
import utils

import csv


training1 = np.load('./emotion_data_pt1.npy')
training2 = np.load('./emotion_data_pt2.npy')
training3 = np.load('./emotion_data_pt3.npy')
training4 = np.load('./emotion_data_pt4.npy')
training5 = np.load('./emotion_data_pt5.npy')

training = np.vstack((training1, training2, training3))

print(training.shape)

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
    Dense(len(train_y[0]), activation='softmax')
])

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=100, batch_size=8)
model.save('emotion-model.keras')

# Evaluate Model
# model.evaluate(...) <--- TODO

pickle.dump({
    'train_x': train_x,
    'train_y': train_y
}, open('training_data', 'wb'))
