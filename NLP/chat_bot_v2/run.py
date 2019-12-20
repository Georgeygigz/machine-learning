import numpy
import random
from app.train_model import train_model
from app.read_file import read_file
from app.bag_of_words import bag_of_words

a, b ,words, data, labels = read_file()
model = train_model()

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


if __name__ == '__main__':
    chat()