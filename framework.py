import json
import pickle
from keras.models import load_model
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import tflearn
import tensorflow as tf

stemmer = LancasterStemmer()
data = pickle.load(open('training_data', 'rb'))
words = data['words']
classes = data['classes']
# train_x = data['train_x']
# train_y = data['train_y']

with open('intents.json') as json_data:
    intents = json.load(json_data)

# tflearn model load
# net = tflearn.input_data(shape=[None, len(train_x[0])])
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
# net = tflearn.regression(net)
#
# model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# model.load('./model.tflearn')

# Keras model load
model = load_model('./model.keras')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = np.zeros(len(words))

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print('found in bag: %s' % w)
    return bag

ERROR_THRESHOLD = 0.25
context = {}

def classify(sentence):

    # tflearn Predictions
    # results = model.predict([bow(sentence, words)])[0]

    # Keras Predictions
    res = np.expand_dims(bow(sentence, words), axis=0)

    results = model.predict(res)[0]
    results = [[i, r] for i, r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))

    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)

    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:

                    if 'context_set' in i:
                        if show_details: print('context:', i['context_set'])
                        context[userID] = i['context_set']

                    if not 'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                            if show_details: print('tag:', i['tag'])

                            return print(random.choice(i['responses']))
            results.pop(0)

print(response('we want to rent a moped'))
print(response('today'))
