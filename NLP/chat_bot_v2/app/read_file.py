import os
import sys
import json
import nltk
import pickle
import numpy
from nltk.stem.lancaster import LancasterStemmer


stemmer = LancasterStemmer()

def read_file():
    with open(os.path.join(sys.path[0],'intents.json')) as file:
        data = json.load(file)
    try:
        with open(os.path.join(sys.path[0],'data.pickle'),'rb') as f:
            words, labels, training, output = pickle.load(f)

    except:

        # all patterns splitted in a single word
        words = []

        # each pattern in a list eg [[Hi], ['How','are','you']]
        doc_pattern = []

        # all tags per each pattern
        doc_tag = []

        # a list of labels eg [greeting,]
        labels = []


        for intent in data['intents']:
            for pattern in intent['patterns']:
                # Tokenization is  converting big quantity of
                # text to smaller part
                # word_tokenize() to split a sentence into words
                splitted_words = nltk.word_tokenize(pattern)

                # Extends list by appending elements from the iterable.
                words.extend(splitted_words)
                doc_pattern.append(splitted_words)
                doc_tag.append(intent['tag'])

            if intent['tag'] not in labels:
                labels.append(intent['tag'])

        # Stemming is a kind of normalization for words.
        words = [stemmer.stem(w.lower()) for w in words if w != "?" ]
        # remove duplicated
        words = sorted(list(set(words)))

        training = []
        output = []

        # create an empty list with the length equal to out tags/intents
        empty_output = [0 for _ in range(len(labels))]

        for i, pattern_in_list in enumerate(doc_pattern):
            # pattern_in_list eg i>> 0, pattern_in_list ['Hi']

            # bag of words is a way of representing text
            # data when modeling text in machine learning algorithms

            # is a representation of text that describes the
            # occurrence of words within a document
            # A measure of the presence of known words.
            bag = []

            pattern_in_list_lower_case = [stemmer.stem(char.lower()) for char in pattern_in_list]
            # output example of pattern_character >> ['hi'] lowercase
            for word in words:
                if word in pattern_in_list_lower_case:
                    bag.append(1)
                else:
                    bag.append(0)

            # create a copy of empty_output list
            output_row = empty_output[:]

            # taking the tag at index i
            # then check at what index is that tag
            # located in labels and return its index lets say index y
            # then append i at index y in output_row

            output_row[labels.index(doc_tag[i])] = 1

            training.append(bag)
            output.append(output_row)

        training = numpy.array(training)
        output = numpy.array(output)

        with open(os.path.join(sys.path[0],'data.pickle'),'wb') as f:
            pickle.dump((words, labels, training, output),f)

    return training, output, words, data, labels