from keras.models import Sequential
from keras.layers import Activation, Dense
import numpy as np
import tflearn
import tensorflow as tf
import pickle
import utils

import json

with open('./intents.json') as json_data:
    intents = json.load(json_data)

# Extract lists of words, classes and documents from the intents file
parsed_intents = utils.parse_intents(intents)

# Covert bag of words and classes into tensors of numbers for tensorflow
training = utils.prepare_data(parsed_intents)

# Split data into training and test sets
# TODO

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

# Evaluate Model
# model.evaluate(...) <--- TODO

pickle.dump({
    'words': parsed_intents['words'],
    'classes': parsed_intents['classes'],
    'train_x': train_x,
    'train_y': train_y
}, open('training_data', 'wb'))
