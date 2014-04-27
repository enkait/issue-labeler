from __future__ import print_function
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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import numpy
import argparse
import pickle
import os
import enchant
from stemming.porter2 import stem

model_map = {
    'BernoulliNB': BernoulliNB,
    'MultinomialNB': MultinomialNB,
}

parser = argparse.ArgumentParser(description='Simple processing script')
parser.add_argument('-data_file', type=str, help='Input file')
parser.add_argument('-test_file', type=str, help='Test file')
parser.add_argument('-num_bench', type=int, default=5, help='Number of benchmarks')
parser.add_argument('-selected', type=int, default=2000, help='Selected by hash compressor')
parser.add_argument('-model_file', type=str, help='File to read/write model to')
parser.add_argument('-stats_file', type=str, help='File to read/write stats to')
parser.add_argument('-generate', dest='generate', action='store_true', help='Should the model be generated')
parser.add_argument('-verbose', dest='verbose', action='store_true', help='Should we output compression details and model')
parser.add_argument('-data_limit', type=int, default=None, help='Number of issues to use for training')
parser.add_argument('-test_limit', type=int, default=None, help='Number of issues to use for testing')
parser.add_argument('-do_stemming', dest='do_stemming', action='store_true', help='Should stemming be used')
parser.add_argument('-do_dictionary', dest='do_dictionary', action='store_true', help='Should dictionary be used')
parser.add_argument('-do_multi', dest='do_multi', action='store_true', help='Should multi be used')
parser.add_argument('-model', dest='model', type=str, choices=model_map.keys(), help='Model to fit')

logging.basicConfig(level=logging.INFO, filename="script_log", filemode="a+",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

random.seed(6355)

class ProcessorConfiguration:
    def __init__(self, do_dictionary=False, do_stemming=False, do_multi=False):
        self.do_dictionary = do_dictionary
        self.do_stemming = do_stemming
        self.do_multi= do_multi

class Processor:
    LABELS = {"enhancement" : "enhancement",
            "feature" : "enhancement",
            "bug" : "bug",
            "question" : "question"}
    MAX_WORD_LEN = 10

    def __init__(self, config):
        self.init()
        self.config = config

    def init(self):
        self.e = enchant.Dict("en_EN")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['e'] #remove enchant from pickle
        return state

    def __setstate__(self, new_state):
        self.__dict__ = new_state
        self.e = enchant.Dict("en_EN")

    def merge_whitespace(self, text):
        return " ".join(text.split()) #convert multiple whitespaces to single whitespace

    def process_text(self, text):
        features = []
        features += self.question_mark(text)
        text = re.sub("[^" + string.ascii_letters + "']", " ", text)
        text = self.merge_whitespace(text)
        text = self.capitalize_text(text)
        if self.config.do_dictionary:
            text = self.spell_check(text)
        if self.config.do_stemming:
            text = self.stemming(text)
        #text = self.process_negatives(text)
        features += text.split()
        return (features, text)

    def multi(self, words):
        features = []
        if self.config.do_multi:
            for ind in range(len(words) - 1):
                features.append(words[ind] + "->" + words[ind + 1])
        return features

    def process_words(self, text):
        words = text.split()
        features = []
        features += self.multi(words)
        return features

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

    def spell_check(self, text):
        wordlist = text.split()
        processedlist = []
        for word in wordlist:
            if self.e.check(word):
                processedlist.append(word)
            else:
                processedlist.append("#")
        return " ".join(processedlist)

    def stemming(self, text):
        wordlist = text.split()
        processedlist = []
        for word in wordlist:
            processedlist.append(stem(word))
        return " ".join(processedlist)

    def capitalize_text(self, text):
        words = []
        for word in text.split():
            if word == word.capitalize():
                words.append(word)
            else:
                words.append(word.lower())
        return " ".join(words)

    def question_mark(self, text):
        if "?" in text:
            return ["?"]
        return []

    def add_features(self, pref, feature_list, out_features):
        for feature in feature_list:
            out_features[pref + "." + feature] += 1

    def extract_features(self, obj, label, out_features):
        if obj[label]:
            obj[label] = re.sub("[^" + string.printable + "']", " ", obj[label])
            (feature_list, text) = self.process_text(obj[label])
            feature_list += self.process_words(text)
            self.add_features(label, feature_list, out_features)

    def process_obj(self, obj):
        features = defaultdict(int)
        labels = set()
        for labelobj in obj["labels"]:
            labelname = labelobj["name"].lower()
            if labelname in self.LABELS:
                labels.add(self.LABELS[labelname])
        if len(labels) == 0:
            logging.warn("Zero labels found: %s", labels)
            return []
        if obj["body"]:
            self.extract_features(obj, "body", features)
        if obj["title"]:
            self.extract_features(obj, "title", features)
        return [(features, label, obj) for label in labels]

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
                #new_features[enc] += value * 1.0 * sign
                new_features[enc] = 1. # ZERO or ONE
                self.words[enc].add(key)
            compressed.append((numpy.array(new_features), self.targets[result], obj))
        return compressed

def feed(gnbs, objs):
    f, t, m = zip(*objs)
    counts = numpy.array([0 for i in range(len(gnbs))])
    for ind, gnb in enumerate(gnbs):
        if random.random() < 1. * (ind+1) / len(gnbs):
            counts[ind] += len(f)
            gnb.partial_fit(f, t, [0, 1, 2], 1)
    logging.info("fed %s objects, counts: %s", len(objs), counts)
    return counts

class Tester:
    def __init__(self, gnbs, counts, targets, stats_file):
        self.gnbs = gnbs
        self.counts = counts
        self.targets = targets
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.stats_file = stats_file

    def feed(self, objs):
        logging.info("Evaluating %s test objects", len(objs))

        ft, expected, mt = zip(*objs)
        expected = numpy.array(expected)

        for ind, gnb in enumerate(self.gnbs):
            received = gnb.predict(ft)
            for target_name, target_id in self.targets.items():
                self.stats[ind][target_name]["correct"] += sum((received == target_id) * (received == expected))
                self.stats[ind][target_name]["retrieved"] += sum(received == target_id)
                self.stats[ind][target_name]["received"] += sum(expected == target_id)
            self.stats[ind]["all"]["correct"] += sum(received == expected)
            self.stats[ind]["all"]["retrieved"] += len(received)
            self.stats[ind]["all"]["received"] += len(received)

        for ind, (a, b) in enumerate(zip(received, expected)):
            self.print("+++++++++++++++++++++++++++++++++++++++++++++++")
            self.print("Should be:", b, ", but was", a)
            self.print("title:", mt[ind]["title"])
            self.print("body:", mt[ind]["body"])
            for ftkey, ftval in enumerate(ft[ind]):
                if ftval > 0:
                    self.print("*******************************")
                    self.print("(%s,%s):" % (ftkey, ftval))
                    for clsname, clsid in self.targets.items():
                        log_prob = gnb.feature_log_prob_[clsid][ftkey]
                        self.print("        %s[%s]=%s" % (clsname, ftkey, log_prob))
            self.print("labels:", [o["name"] for o in mt[ind]["labels"]])
            self.print("url:", mt[ind]["url"])
            self.print("html url:", mt[ind]["html_url"])
            self.print("+++++++++++++++++++++++++++++++++++++++++++++++")

    def format_ratio(self, a, b):
        return "%s/%s = (%s)" % (a, b, a * 1.0 / b)

    def print(self, *args, **kwargs):
        print(*args, file=self.stats_file, **kwargs)

    def format_results(self, stats):
        for target_name, details in stats.items():
            self.print("    for %s: precision: %s, recall: %s" % (target_name,
                    self.format_ratio(details["correct"], details["retrieved"]),
                    self.format_ratio(details["correct"], details["received"])))

    def print_stats(self):
        for ind, (gnb, count) in enumerate(zip(self.gnbs, self.counts)):
            self.print("---------------------------------------------")
            self.print("For %d learning examples" % (count,))
            self.format_results(self.stats[ind])
            self.print("---------------------------------------------")

def assert_test_file(args):
    if args.test_file:
        if not args.stats_file:
            print("No file to store stats in")
            exit(1)
        if os.path.exists(args.stats_file):
            print("Stats file path exists")
            exit(1)

def main():
    args = parser.parse_args()

    assert_test_file(args)

    if args.generate:
        if not args.model_file:
            print("No file to store model in")
            exit(1)
        if os.path.exists(args.model_file):
            print("Model file path exists")
            exit(1)
        logging.info("Generating models")

        processor_config = ProcessorConfiguration(do_dictionary=args.do_dictionary,
                do_stemming=args.do_stemming, do_multi=args.do_multi)

        processor = Processor(processor_config)
        comp = HashCompressor(args.selected)
        gnbs = [model_map[args.model]() for i in range(args.num_bench)]
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
        logging.info("Loading models from file")
        with open(args.model_file, "r") as model_file:
            processor = pickle.load(model_file)
            processor.init()
            comp = pickle.load(model_file)
            gnbs = pickle.load(model_file)
            counts = pickle.load(model_file)
            gnb = gnbs[-1]

    if args.test_file:
        logging.info("Evaluating models on test data set")

        assert_test_file(args)

        with open(args.stats_file, "w") as stats_file:
            tester = Tester(gnbs, counts, comp.targets, stats_file)

            lines_read = 0

            with open(args.test_file, "r") as inp:
                test_objs = []
                while True:
                    line = inp.readline()
                    if not line or lines_read == args.test_limit:
                        tester.feed(test_objs)
                        break
                    lines_read += 1
                    obj = json.loads(line.strip())
                    proc_objs = processor.process_obj(obj)
                    comp_objs = comp.compress(proc_objs)
                    test_objs += comp_objs

                    if len(test_objs) >= 500:
                        tester.feed(test_objs)
                        test_objs = []

            tester.print_stats()

if __name__ == "__main__":
    main()
