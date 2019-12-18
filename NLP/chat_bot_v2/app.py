import nltk
import numpy
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tflearn
import random
import json
import os
import sys
from nltk.stem.lancaster import LancasterStemmer
import pickle

# # Config to turn on JIT compilation
# config = tf.ConfigProto()
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

# session = tf.Session(config=config)

# # how to set the TF_SetConfig in the C API

stemmer = LancasterStemmer()
with open(os.path.join(sys.path[0],'intents.json')) as file:
    data = json.load(file)
try:
    with open(os.path.join(sys.path[0],'data.pickle'),'rb') as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_pattern = []
    docs_tag = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            # stem take each word in the pattern and bring it the rule word
            # tokenize get all the word in our pattern
            wrd = nltk.word_tokenize(pattern)
            words.extend(wrd)
            docs_pattern.append(wrd)
            docs_tag.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    # Bug of words
    # we use one hot enconding to represent word in numerics

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_pattern):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_tag[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)
# import pdb; pdb.set_trace()

    with open(os.path.join(sys.path[0],'data.pickle'),'wb') as f:
        pickle.dump((words, labels, training, output),f)


tf.reset_default_graph()
net = tflearn.input_data(shape= [None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation="linear")
net = tflearn.regression(net)

model =tflearn.DNN(net)


try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("""Hey I am Chatty, how can I help you. Type
          anything and when you want to leave just type (quit)""")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        results = model.predict([bag_of_words(user_input, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg["tag"] == tag:
                response = tg['responses']
        print(random.choice(response))

chat()