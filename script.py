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

logging.basicConfig(level=logging.INFO, filename="script_log", filemode="a+",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

random.seed(12312)

class Processor:
    LABELS = {"enhancement" : "enhancement",
            "feature" : "enhancement",
            "bug" : "bug",
            "question" : "question"}
    MAX_WORD_LEN = 10

    def process_text(self, text):
        text = re.sub("[^" + string.printable + "]", "", text)
        text = re.sub("[^" + string.ascii_letters + "]", " ", text)
        text = text.lower()
        return text

    def process_word(self, word):
        return word.strip()

    def make_features(self, pref, text, features):
        wordlist = map(self.process_word, text.split())
        for i in range(len(wordlist)):
            word = ""
            for j in range(i, len(wordlist)):
                if len(wordlist[j]) <= 2:
                    continue
                word += wordlist[j]
                if wordlist[j] not in ["not"]:
                    break
            i = j + 1
            if word:
                #word = word[:self.MAX_WORD_LEN]
                features[pref + "." + word] += 1
        #for ind in range(len(wordlist)-1):
        #    features[pref + "." + wordlist[ind] + "->" + wordlist[ind+1]] += 1

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
            self.make_features("body", obj["body"], features)
        if obj["title"]:
            obj["title"] = self.process_text(obj["title"])
            self.make_features("title", obj["title"], features)
        """
        result = []
        for label in labels:
            nfeatures = dict(features)
            nfeatures["ans:" + label] = 1
            result.append((nfeatures, label, obj)) #SANITY CHECK
        return result
        """
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
    def __init__(self):
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

def feed(gnb, objs):
    f, t, m = zip(*objs)
    gnb.partial_fit(f, t, [0, 1, 2], 1)

def main():
    args = parser.parse_args()

    test_objs = []
    p = Processor()
    comp = HashCompressor()
    gnb = MultinomialNB()

    with open(args.inputfile, "r") as inp:
        todo_objs = []
        while True:
            line = inp.readline()
            if not line:
                feed(gnb, todo_objs)
                break
            obj = json.loads(inp.readline().strip())
            proc_objs = p.process_obj(obj)
            comp_objs = comp.compress(proc_objs)

            if random.random() < 0.1:
                test_objs += comp_objs
            else:
                todo_objs += comp_objs

            if len(todo_objs) >= 1000:
                feed(gnb, todo_objs)
                todo_objs = []

    for key, value in comp.words.items():
        print key, value

    ft, tt, mt = zip(*test_objs)
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

    for clsname, clsid in comp.targets.items():
        for ind, value in enumerate(gnb.feature_log_prob_[clsid]):
            print "%s[%s]=%s" % (clsname, ind, value)


    print numpy.bincount(tt)
    print testres
    print tt
    print "All:", sum(testres == tt)
    a = sum([1 if testres[i] == tt[i] and (tt[i] == comp.targets["bug"] or tt[i] == comp.targets["enhancement"]) else 0 for i in range(len(testres))])
    b = sum([1 if tt[i] == comp.targets["bug"] or tt[i] == comp.targets["enhancement"] else 0 for i in range(len(testres))])
    print "Bugs vs enhancements:", a, "/", b

if __name__ == "__main__":
    main()
