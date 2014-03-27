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
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import numpy
import argparse

parser = argparse.ArgumentParser(description='Simple processing script')
parser.add_argument('-inputfile', type=str, help='Input file')

args = parser.parse_args()

logging.basicConfig(level=logging.INFO, filename="script_log", filemode="a+",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

random.seed(12312)

inp = map(json.loads, open(args.inputfile, "r").readlines())
random.shuffle(inp)

learn = inp[:len(inp)-200]
test = inp[len(inp)-200:]

class Processor:
    LABELS = {"enhancement" : "enhancement",
            "feature" : "enhancement",
            "bug" : "bug",
            "question" : "question"}

    def process_text(self, text):
        text = re.sub("[^" + string.printable + "]", "", text)
        text = re.sub("[^" + string.ascii_letters + "]", " ", text)
        text = text.lower()
        return text

    def process_word(self, word):
        return word.strip()

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
            obj["body"] = self.process_text(obj["body"])
            for word in obj["body"].split():
                word = self.process_word(word)
                if (len(word) <= 2):
                    continue
                if word:
                    features["body." + word] += 1
        if obj["title"]:
            obj["title"] = self.process_text(obj["title"])
            for word in obj["title"].split():
                word = self.process_word(word)
                if (len(word) <= 2):
                    continue
                if word:
                    features["title." + word] += 1
        return [(features, label, obj) for label in labels]

def pr(L):
    for (features, result, obj) in L:
        print result
        print obj["url"]
        for key, value in features.items():
            print "    ", key, ":", value

def pr_compressed(L):
    for (features, result, obj) in L:
        print result, obj["url"], features

class HashCompressor:
    CHOSEN = 2000
    def learn(self, L):
        self.words = defaultdict(set)
        self.targets = {}

    def compress(self, L):
        compressed = []
        for (features, result, obj) in L:
            if result not in self.targets:
                self.targets[result] = len(self.targets)
            new_features = [0 for i in range(self.CHOSEN)]
            for key, value in features.items():
                enc = hash(key) % (2 * self.CHOSEN)
                #sign = -1 if enc < self.CHOSEN else 1
                sign = 1 # CAREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEe
                enc %= self.CHOSEN
                new_features[enc] += value * 1.0 * sign
                self.words[enc].add(key)
            compressed.append((numpy.array(new_features), self.targets[result], obj))
        for key, value in self.words.items():
            print key, value
        return compressed

class Compressor:
    CHOSEN = 1000
    def learn(self, L):
        words = defaultdict(int)
        self.targets = {}
        self.key_words = defaultdict(lambda: defaultdict(int))
        counts = defaultdict(int)
        for (features, result, obj) in L:
            for key, value in features.items():
                words[key] += 1
            if result not in self.targets:
                self.targets[result] = len(self.targets)
            for key, value in features.items():
                self.key_words[result][key] += 1
            counts[result] += 1
        occur = [(count, word) for (word, count) in words.items()]
        self.choices = [word for (count, word) in sorted(occur)[-self.CHOSEN:]]
        for key, value in self.key_words.items():
            for key2, value2 in value.items():
                print "%s.%s" % (key2, key), value2 * 1.0 / counts[key]
        pprint.pprint(self.key_words)
        print self.choices

    def compress(self, L):
        compressed = []
        for (features, result, obj) in L:
            new_features = []
            for ch in self.choices:
                new_features.append(features[ch])
            compressed.append((numpy.array(new_features), self.targets[result], obj))
        return compressed

p = Processor()

procinp = sum(map(p.process_obj, learn), [])
#pr(procinp)
comp = HashCompressor()
comp.learn(procinp)
compressedinp = comp.compress(procinp)
#pr_compressed(compressedinp)

f, t, m = zip(*compressedinp)

#scaler = preprocessing.StandardScaler()
#f = scaler.fit_transform(f)

#gnb = GaussianNB()
gnb = MultinomialNB()
gnb.fit(f, t)

proctest = sum(map(p.process_obj, test), [])
compressedtest = comp.compress(proctest)
ft, tt, mt = zip(*compressedtest)
#ft = scaler.fit_transform(ft)
tt = numpy.array(tt)

testres = gnb.predict(ft)
testprobs = gnb.predict_proba(ft)

for ind, (a, b) in enumerate(zip(testres, tt)):
    print "+++++++++++++++++++++++++++++++++++++++++++++++"
    print "Should be:", b, ", but was", a
    print "title:", mt[ind]["title"]
    print "body:", mt[ind]["body"]
    print "labels:", [o["name"] for o in mt[ind]["labels"]]
    print "url:", mt[ind]["url"]
    print "html url:", mt[ind]["html_url"]
    print "+++++++++++++++++++++++++++++++++++++++++++++++"

print numpy.bincount(tt)
print testres
print tt
print sum(testres == tt)
