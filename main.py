import nltk
from nltk.stem.lancaster import LancasterStemmer   
import numpy
import tflearn
import tensorflow
import json
import random

stemmer = LancasterStemmer()

with open("intents.json") as f:
    data = json.load(f)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        doc_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_emty = [0 for _ in range(len(classes))]

for x,doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_emty[:]
    out_row[labels.index(docs_y(docs_y[x]))] = 1

    training.append(bag)
    output.append(out_row)

training = numpy.array(trainning)    
output = numpy.array(output)
#if __name__ == "__main__":

