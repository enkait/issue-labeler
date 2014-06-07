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
from sklearn.linear_model import SGDClassifier
import numpy
import argparse
import pickle
import os
import enchant
from stemming.porter2 import stem

model_map = {
    'BernoulliNB': BernoulliNB,
    'MultinomialNB': MultinomialNB,
    'LogisticRegressionSGD': lambda : SGDClassifier(loss="log"),
    'ModifiedHuberSGD': lambda : SGDClassifier(loss="modified_huber"),
    'HingeSGD': lambda : SGDClassifier(loss="hinge"),
}

random.seed(6355)

def add_parser_arguments(parser):
    parser.add_argument('-train_file', type=str, help='train file')
    parser.add_argument('-test_file', type=str, help='Test file')
    parser.add_argument('-num_bench', type=int, default=5, help='Number of benchmarks')
    parser.add_argument('-selected', type=int, default=2000, help='Selected by hash compressor')
    parser.add_argument('-model_file', type=str, help='File to read/write model to')
    parser.add_argument('-stats_file', type=str, help='File to read/write stats to')
    parser.add_argument('-generate', dest='generate', action='store_true', help='Should the model be generated')
    parser.add_argument('-evaluate_test', dest='evaluate_test', action='store_true', help='Should the test set be evaluated')
    parser.add_argument('-vectorize_train_output', type=str, help='Where should the vectorized train data be stored')
    parser.add_argument('-vectorize_test_output', type=str, help='Where should the vectorized test data be stored')
    parser.add_argument('-verbose', dest='verbose', action='store_true', help='Should we output compression details and model')
    parser.add_argument('-train_limit', type=int, default=None, help='Number of issues to use for training')
    parser.add_argument('-test_limit', type=int, default=None, help='Number of issues to use for testing')
    parser.add_argument('-do_stemming', dest='do_stemming', action='store_true', help='Should stemming be used')
    parser.add_argument('-do_dictionary', dest='do_dictionary', action='store_true', help='Should dictionary be used to filter words')
    parser.add_argument('-do_correct_spelling', dest='do_correct_spelling', action='store_true', help='Should dictionary be used to correct spelling')
    parser.add_argument('-do_filter_common', dest='do_filter_common', action='store_true', help='Should filtering of common words be used')
    parser.add_argument('-do_multi', dest='do_multi', action='store_true', help='Should multi be used')
    parser.add_argument('-model', dest='model', type=str, choices=model_map.keys(), help='Model to fit')
    parser.add_argument('-evaluate_on_train', dest='evaluate_on_train', action='store_true', help='Evaluate on train')

logging.basicConfig(level=logging.INFO, filename="script_log", filemode="a+",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class ProcessorConfiguration:
    @staticmethod
    def FromArgs(args):
        result = ProcessorConfiguration()
        result.do_dictionary = args.do_dictionary
        result.do_correct_spelling = args.do_correct_spelling
        result.do_stemming = args.do_stemming
        result.do_filter_common = args.do_filter_common
        result.do_multi= args.do_multi
        return result

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
        features += self.url(text)
        text = re.sub("[^" + string.ascii_letters + "']", " ", text)
        text = self.merge_whitespace(text)
        text = self.capitalize_text(text)
        if self.config.do_filter_common:
            text = self.filter_common(text)
        if self.config.do_correct_spelling:
            text = self.spell_fix(text)
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
            for ind in range(len(words) - 2):
                features.append(words[ind] + "->" + words[ind + 1] + "->" + words[ind + 2])
        return features

    def filter_common(self, text):
        result = []
        for word in text.split():
            if not word.lower() in ["the", "to", "a", "in", "is", "and", "i", "of", "for", "it", "this",
                    "on", "at", "be", "with", "that", "if", "from", "when", "as", "but", "you", "have", "an", "are", "file", "or", "by"]:
                result.append(word)
        return " ".join(result)

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

    def dict_suggest(self, word):
        w = "#"
        for w in self.e.suggest(word):
            if w == w.lower():
                return w
        return w

    def spell_fix(self, text):
        wordlist = text.split()
        processedlist = []
        for word in wordlist:
            processedlist.append(self.dict_suggest(word))
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

    def url(self, text):
        if "http://" in text or "https://" in text:
            return ["URL"]
        return []

    def question_mark(self, text):
        if "?" in text:
            return ["?"]
        return []

    def add_features(self, pref, feature_list, out_features):
        for feature in feature_list:
            out_features[pref + "." + feature] += 1

    def remove_unprintable(self, text):
        restext = []
        for c in text:
            if c in string.printable:
                restext.append(c)
        return "".join(restext)

    def extract_features(self, obj, label, out_features):
        if obj[label]:
            obj[label] = self.remove_unprintable(obj[label])
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

    def process_title_body(self, title, body):
        obj = {"title": title, "body": body}
        features = defaultdict(int)
        self.extract_features(obj, "body", features)
        self.extract_features(obj, "title", features)
        return features

class HashCompressor:
    def __init__(self, chosen):
        self.words = defaultdict(set)
        self.targets = {}
        self.chosen = chosen

    def compress_features(self, features):
        new_features = [0 for i in range(self.chosen)]
        for key, value in features.items():
            enc = hash(key) % (2 * self.chosen)
            #sign = -1 if enc < self.chosen else 1
            sign = 1 # CAREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEe
            enc %= self.chosen
            #new_features[enc] += value * 1.0 * sign
            new_features[enc] = 1. # ZERO or ONE
            self.words[enc].add(key)
        return numpy.array(new_features)

    def compress(self, L):
        compressed = []
        for (features, result, obj) in L:
            if result not in self.targets:
                self.targets[result] = len(self.targets)
            new_features = self.compress_features(features)
            compressed.append((new_features, self.targets[result], obj))
        return compressed

def feed(gnbs, objs):
    f, t, m = zip(*objs)
    counts = numpy.array([0 for i in range(len(gnbs))])
    for ind, gnb in enumerate(gnbs):
        if random.random() < 1. * (ind+1) / len(gnbs):
            counts[ind] += len(f)
            gnb.partial_fit(f, t, [0, 1, 2])
    logging.info("fed %s objects, counts: %s", len(objs), counts)
    return counts

class Tester:
    def __init__(self, gnbs, counts, targets, stats_file, verbose):
        self.gnbs = gnbs
        self.counts = counts
        self.targets = targets
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.stats_file = stats_file
        self.verbose = verbose

    def consume(self, objs):
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

        """
        if self.verbose:
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
        """

    def format_ratio(self, a, b):
        return "%s/%s = (%s)" % (a, b, a * 1.0 / b)

    def print(self, *args, **kwargs):
        print(*args, file=self.stats_file, **kwargs)

    def format_results(self, stats):
        for target_name, details in stats.items():
            if target_name != "all":
                self.print("    for %s: precision: %s, recall: %s" % (target_name,
                        self.format_ratio(details["correct"], details["retrieved"]),
                        self.format_ratio(details["correct"], details["received"])))
            else:
                self.print("    for %s: result: %s" % (target_name,
                        self.format_ratio(details["correct"], details["received"])))

    def print_stats(self):
        for ind, (gnb, count) in enumerate(zip(self.gnbs, self.counts)):
            self.print("---------------------------------------------")
            self.print("For %d learning examples" % (count,))
            self.format_results(self.stats[ind])
            self.print("---------------------------------------------")

    def done(self):
        self.print_stats()

def check_args_consistent(args):
    if args.test_file or (args.train_file and args.evaluate_on_train):
        if not args.stats_file:
            print("No file to store stats in")
            exit(1)
        if os.path.exists(args.stats_file):
            print("Stats file path exists")
            exit(1)
    if args.generate:
        if not args.model_file:
            print("No file to store model in")
            exit(1)
        if os.path.exists(args.model_file):
            print("Model file path exists")
            exit(1)

class Feeder:
    def __init__(self, processor, comp, source_file, source_limit=None, source_skip=None):
        self.processor = processor
        self.comp = comp
        self.source_file = source_file
        self.source_limit = source_limit
        self.source_skip = source_skip

    def feed(self, consumer):
        lines_read = 0
        lines_processed = 0
        test_objs = []
        while True:
            line = self.source_file.readline()
            if not line or lines_processed == self.source_limit:
                consumer.consume(test_objs)
                break
            lines_read += 1
            if self.source_skip == None or (self.source_skip != None and lines_read > self.source_skip):
                lines_processed += 1
                obj = json.loads(line.strip())
                proc_objs = self.processor.process_obj(obj)
                comp_objs = self.comp.compress(proc_objs)
                test_objs += comp_objs

            if len(test_objs) >= 500:
                consumer.consume(test_objs)
                test_objs = []

class GenerationConsumer:
    def __init__(self, gnbs, counts, model_file, processor, comp):
        self.gnbs = gnbs
        self.counts = counts
        self.model_file = model_file
        self.processor = processor
        self.comp = comp

    def consume(self, objs):
        self.counts += feed(self.gnbs, objs)

    def done(self):
        with open(self.model_file, "w") as model_file:
            pickle.dump(self.processor, model_file)
            pickle.dump(self.comp, model_file)
            pickle.dump(self.gnbs, model_file)
            pickle.dump(self.counts, model_file)

class OutputConsumer:
    def __init__(self, output_file, selected):
        self.output_file = output_file
        self.output_file.write("@relation github-issues\n")
        for i in range(selected):
            self.output_file.write("@attribute %s numeric\n" % (i,))
        self.output_file.write("@attribute output numeric\n")
        self.output_file.write("@data\n")
        self.selected = selected

    def process_row(self, f, t):
        self.output_file.write("{")
        for i in xrange(self.selected):
            if f[i] > 0.5:
                self.output_file.write("%s %s," % (i, f[i]))
        self.output_file.write("%s %s" % (self.selected, t))
        self.output_file.write("}\n")

    def consume(self, objs):
        for f, t, m in objs:
            self.process_row(f, t)

    def done(self):
        self.output_file.close()

class MultiConsumer:
    def __init__(self):
        self.consumers = []

    def add_consumer(self, consumer):
        self.consumers.append(consumer)

    def consume(self, objs):
        for consumer in self.consumers:
            consumer.consume(objs)

    def done(self):
        for consumer in self.consumers:
            consumer.done()

def load_model(model_file):
    processor = pickle.load(model_file)
    processor.init()
    comp = pickle.load(model_file)
    gnbs = pickle.load(model_file)
    counts = pickle.load(model_file)
    return (processor, comp, gnbs, counts)

def classify((proc, comp, gnbs, counts), title, body):
    gnb = gnbs[-1]
    feat = proc.process_title_body(title, body)
    cfeat = comp.compress_features(feat)
    pred = gnb.predict([cfeat])
    lookup = dict(map(lambda (a, b): (b, a), comp.targets.items()))[pred[0]]
    return lookup

def main():
    parser = argparse.ArgumentParser(description='Simple processing script')
    add_parser_arguments(parser)
    args = parser.parse_args()

    check_args_consistent(args)

    if args.generate or args.vectorize_train_output:
        processor_config = ProcessorConfiguration.FromArgs(args)

        processor = Processor(processor_config)
        comp = HashCompressor(args.selected)
        train_consumer = MultiConsumer()
        if args.generate:
            logging.info("Generating models")
            gnbs = [model_map[args.model]() for i in range(args.num_bench)]
            counts = numpy.array([0 for i in range(args.num_bench)])
            train_consumer.add_consumer(GenerationConsumer(gnbs, counts, args.model_file, processor, comp))
        if args.vectorize_train_output:
            logging.info("Vectorizing train")
            vectorize_train_output = open(args.vectorize_train_output, "w")
            train_consumer.add_consumer(OutputConsumer(vectorize_train_output, args.selected))
        lines_read = 0

        with open(args.train_file, "r") as inp:
            Feeder(processor, comp, inp, args.train_limit).feed(train_consumer)
        train_consumer.done()

    elif not args.generate and args.model_file:
        logging.info("Loading models from file")
        with open(args.model_file, "r") as model_file:
            processor, comp, gnbs, counts = load_model(model_file)

    with open(args.stats_file, "w") as stats_file:
        test_consumer = MultiConsumer()
        if args.evaluate_test:
            tester = Tester(gnbs, counts, comp.targets, stats_file, True)
            logging.info("Evaluating test")
            test_consumer.add_consumer(tester)
        if args.vectorize_test_output:
            logging.info("Vectorizing test")
            vectorize_test_output = open(args.vectorize_test_output, "w")
            test_consumer.add_consumer(OutputConsumer(vectorize_test_output, args.selected))

        if args.test_file:
            logging.info("Evaluating models on test data set")
            with open(args.test_file, "r") as test_file:
                Feeder(processor, comp, test_file, args.train_limit).feed(test_consumer)
                test_consumer.done()

        if args.train_file and args.evaluate_on_train:
            logging.info("Evaluating models on train input set")
            tester = Tester(gnbs, counts, comp.targets, stats_file, False)
            with open(args.train_file, "r") as train_file:
                tester = Tester(gnbs, counts, comp.targets, stats_file, False)
                Feeder(processor, comp, train_file, args.train_limit).feed(tester)
                tester.done()

if __name__ == "__main__":
    main()
