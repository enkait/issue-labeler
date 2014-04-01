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
import pickle
import os

parser = argparse.ArgumentParser(description='Simple processing script')
parser.add_argument('-data_file', type=str, help='Input file')
parser.add_argument('-test_file', type=str, help='Test file')
parser.add_argument('-num_bench', type=int, default=5, help='Number of benchmarks')
parser.add_argument('-selected', type=int, default=2000, help='Selected by hash compressor')
parser.add_argument('-model_file', type=str, help='File to read/write model to')
parser.add_argument('-generate', dest='generate', action='store_true', help='Should the model be generated')
parser.add_argument('-verbose', dest='verbose', action='store_true', help='Should we output compression details and model')
parser.add_argument('-data_limit', type=int, default=None, help='Number of issues to use for training')

logging.basicConfig(level=logging.INFO, filename="script_log", filemode="a+",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

#random.seed(12312)
random.seed(6355)

class Processor:
    LABELS = {"enhancement" : "enhancement",
            "feature" : "enhancement",
            "bug" : "bug",
            "question" : "question"}
    MAX_WORD_LEN = 10

    def process_text(self, text):
        text = re.sub("[^" + string.printable + "]", "", text)
        text = re.sub("[^" + string.ascii_letters + "?']", " ", text)
        text = text.lower()
        text = " ".join(text.split()) #convert multiple whitespaces to single whitespace
        text = self.process_negatives(text)
        text = self.question_mark(text)
        return text

    def process_word(self, word):
        return word.strip()

    def process_negatives(self, text):
        wordlist = text.split()
        processedlist = []
        ind = 0
        while ind < len(wordlist):
            if wordlist[ind] in ["not", "no"] and ind + 1 < len(wordlist):
                processedlist.append(wordlist[ind] + wordlist[ind+1])
                ind += 2
            else:
                processedlist.append(wordlist[ind])
                ind += 1
        return " ".join(processedlist)

    def question_mark(self, text):
        if "?" in text:
            text = re.sub("[^" + string.ascii_letters + "']", " ", text)
            text += " ?"
        return text

    def make_features(self, pref, text, features):
        wordlist = map(self.process_word, text.split())
        for word in wordlist:
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
    def __init__(self, chosen):
        self.words = defaultdict(set)
        self.targets = {}
        self.chosen = chosen

    def compress(self, L):
        compressed = []
        for (features, result, obj) in L:
            if result not in self.targets:
                self.targets[result] = len(self.targets)
            new_features = [0 for i in range(self.chosen)]
            for key, value in features.items():
                enc = hash(key) % (2 * self.chosen)
                #sign = -1 if enc < self.chosen else 1
                sign = 1 # CAREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEe
                enc %= self.chosen
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

def feed(gnbs, objs):
    f, t, m = zip(*objs)
    counts = numpy.array([0 for i in range(len(gnbs))])
    for ind, gnb in enumerate(gnbs):
        if random.random() < 1. * (ind+1) / len(gnbs):
            counts[ind] += len(f)
            gnb.partial_fit(f, t, [0, 1, 2], 1)
    return counts

def format_one_result(good, total):
    return "%s/%s = (%s)" % (good, total, good * 1.0 / total)

def format_results(expected, received, targets):
    result = "results: %s correct\n" % (format_one_result(sum(expected == received), len(expected)),)
    for target_name, target_id in targets.items():
        result += "    for %s: %s\n" % (target_name, format_one_result(
            sum((received == target_id) * (received == expected)), sum(expected == target_id)))
    return result

def main():
    args = parser.parse_args()

    if args.generate:
        if not args.model_file:
            print "No file to store model in"
            exit(1)
        if os.path.exists(args.model_file):
            print "Model file path exists"
            exit(1)

        processor = Processor()
        comp = HashCompressor(args.selected)
        gnbs = [MultinomialNB() for i in range(args.num_bench)]
        counts = numpy.array([0 for i in range(args.num_bench)])
        lines_read = 0

        with open(args.data_file, "r") as inp:
            todo_objs = []
            while True:
                line = inp.readline()
                if not line or lines_read == args.data_limit:
                    counts += feed(gnbs, todo_objs)
                    break
                lines_read += 1
                obj = json.loads(line.strip())
                proc_objs = processor.process_obj(obj)
                comp_objs = comp.compress(proc_objs)
                todo_objs += comp_objs

                if len(todo_objs) >= 500:
                    counts += feed(gnbs, todo_objs)
                    todo_objs = []

        with open(args.model_file, "w") as model_file:
            pickle.dump(processor, model_file)
            pickle.dump(comp, model_file)
            pickle.dump(gnbs, model_file)
            pickle.dump(counts, model_file)

    else:
        with open(args.model_file, "r") as model_file:
            processor = pickle.load(model_file)
            comp = pickle.load(model_file)
            gnbs = pickle.load(model_file)
            counts = pickle.load(model_file)
            gnb = gnbs[-1]

    if args.test_file:
        test_objs = []
        with open(args.test_file, "r") as inp:
            while True:
                line = inp.readline()
                if not line:
                    break
                obj = json.loads(line.strip())
                proc_objs = processor.process_obj(obj)
                comp_objs = comp.compress(proc_objs)
                test_objs += comp_objs

        print "Number of test objects:", len(test_objs)

        ft, tt, mt = zip(*test_objs)
        tt = numpy.array(tt)

        if args.verbose:
            for key, value in comp.words.items():
                print key, value

        for ind, (gnb, count) in enumerate(zip(gnbs, counts)):
            testres = gnb.predict(ft)
            testprobs = gnb.predict_proba(ft)
            print "---------------------------------------------"
            print "Test data bin counts: ", numpy.bincount(tt)
            print "for %d learning examples" % (count,)
            print format_results(tt, testres, comp.targets)
            print "---------------------------------------------"

        testres = gnb.predict(ft)

        for ind, (a, b) in enumerate(zip(testres, tt)):
            print "+++++++++++++++++++++++++++++++++++++++++++++++"
            print "Should be:", b, ", but was", a
            print "title:", mt[ind]["title"]
            print "body:", mt[ind]["body"]
            for ftkey, ftval in enumerate(ft[ind]):
                if ftval > 0:
                    print "(%s,%s):" % (ftkey, ftval)
                    for clsname, clsid in comp.targets.items():
                        log_prob = gnb.feature_log_prob_[clsid][ftkey]
                        print "        %s[%s]=%s" % (clsname, ftkey, log_prob)
            print "labels:", [o["name"] for o in mt[ind]["labels"]]
            print "url:", mt[ind]["url"]
            print "html url:", mt[ind]["html_url"]
            print "+++++++++++++++++++++++++++++++++++++++++++++++"

        if args.verbose:
            for clsname, clsid in comp.targets.items():
                for ind, value in enumerate(gnb.feature_log_prob_[clsid]):
                    print "%s[%s]=%s" % (clsname, ind, value)


if __name__ == "__main__":
    main()
