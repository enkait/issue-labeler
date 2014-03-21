from sklearn import linear_model, preprocessing
from sklearn.decomposition import PCA
import random
import arff
import csv
import math
import pprint
import sys
import json
import re
import logging
from collections import defaultdict
import string
from sklearn.naive_bayes import GaussianNB
import numpy

logging.basicConfig(level=logging.INFO, filename="script_log", filemode="a+",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

random.seed(12312)

inp = map(json.loads, open("output", "r").readlines())
random.shuffle(inp)

learn = inp[:len(inp)-200]
test = inp[len(inp)-200:]

class Processor:
    LABELS = {"enhancement" : "enhancement",
            "feature" : "enhancement",
            "bug" : "bug",
            "question" : "question"}

    def __init__(self, inp):
        words = defaultdict(int)
        for obj in inp:
            try:
                for elem in map(lambda w: w.strip(), obj["body"].split()):
                    words[elem] += 1
            except Exception as ex:
                print "Exception"
        self.chosen = set()
        for key, value in words.items():
            if value > 20:
                self.chosen.add(key)
        print self.chosen

    def process_text(self, text):
        text = re.sub("[^" + string.printable + "]", "", text)
        text = re.sub("[^" + string.ascii_letters + "]", " ", text)
        text = text.lower()
        return text

    def process_word(self, word):
        return word

    def process_obj(self, obj):
        features = defaultdict(int)
        labels = set()
        for labelobj in obj["labels"]:
            labelname = labelobj["name"]
            if labelname in self.LABELS:
                labels.add(self.LABELS[labelname])
        if len(labels) == 0:
            logging.warn("Zero labels found: %s", labels)
            return []
        if obj["body"]:
            for word in self.process_text(obj["body"]).split():
                word = self.process_word(word)
                if (len(word) <= 2):
                    continue
                if word:
                    features["body." + word] += 1
        if obj["title"]:
            for word in self.process_text(obj["title"]).split():
                word = self.process_word(word)
                if (len(word) <= 2):
                    continue
                if word:
                    features["title." + word] += 1
        return [(features, label, obj) for label in labels]

    def process(self, obj):
        words = defaultdict(int)
        try:
            for elem in map(lambda w: w.strip(), obj["body"].split()):
                words[elem] += 1
        except Exception as ex:
            print "Exception"
        result = {}
        for word in self.chosen:
            result[word] = words[word]
        return result

def pr(L):
    for (features, result, obj) in L:
        print result
        print obj["url"]
        for key, value in features.items():
            print "    ", key, ":", value

def pr_compressed(L):
    for (features, result, obj) in L:
        print result, obj["url"], features

class Compresser:
    CHOSEN = 1000
    def learn(self, L):
        words = defaultdict(int)
        self.targets = {}
        for (features, result, obj) in L:
            for key, value in features.items():
                words[key] += 1
            if result not in self.targets:
                self.targets[result] = len(self.targets)
        occur = [(count, word) for (word, count) in words.items()]
        self.choices = [word for (count, word) in sorted(occur)[-self.CHOSEN:]]

    def compress(self, L):
        compressed = []
        for (features, result, obj) in L:
            new_features = []
            for ch in self.choices:
                new_features.append(features[ch])
            compressed.append((numpy.array(new_features), self.targets[result], obj))
        return compressed

p = Processor(learn)

procinp = sum(map(p.process_obj, learn), [])
pr(procinp)
comp = Compresser()
comp.learn(procinp)
compressedinp = comp.compress(procinp)
pr_compressed(compressedinp)

f, t, m = zip(*compressedinp)

gnb = GaussianNB()
gnb.fit(f, t)

proctest = sum(map(p.process_obj, test), [])
compressedtest = comp.compress(proctest)
ft, tt, mt = zip(*compressedtest)
tt = numpy.array(tt)

testres = gnb.predict(ft)

print numpy.bincount(tt)

print testres
print tt
print sum(testres != tt)
