import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import random


# Parse intents file
def parse_intents(intents):

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

    return {
        'classes': classes,
        'documents': documents,
        'words': words
    }

# Prepare data for consumption by tensorflow
def prepare_data(intents):

    data = []
    output_empty = np.zeros(len(intents['classes']))

    word_count = len(intents['words'])
    doc_count = len(intents['documents'])
    count = 0

    for doc in intents['documents']:

        print(count, '/', doc_count)
        count = count + 1
        bag = np.zeros(word_count) # init bag of words
        pattern_words = doc[0]
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

        for i, w in enumerate(intents['words']):
            if w in pattern_words:
                bag[i] = 1

        output_row = np.array(output_empty)
        output_row[intents['classes'].index(doc[1])] = 1

        data.append([bag, output_row])

    random.shuffle(data)
    return np.asarray(data)
