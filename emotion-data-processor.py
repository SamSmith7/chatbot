import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import pickle
import re
import utils

import csv

words = []
classes = []
documents = []
ignore = re.compile('[0-9]+|[,\.!?&@#]')

# ignore_words = ['?']

with open('./data/train_data.csv') as training_csv:
    csv_reader = csv.reader(training_csv, delimiter=',')

    for row in csv_reader:

        phrase = re.sub('[0-9]+|[,\.!?&@#]', '', row[1])

        w = nltk.word_tokenize(phrase)
        words.extend(w)
        documents.append((w, row[0]))

        if row[0] not in classes:
            classes.append(row[0])

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

training = utils.prepare_data({
    'classes': classes,
    'documents': documents[8000:10000],
    'words': words
})

np.save('emotion_data_pt5', training)
