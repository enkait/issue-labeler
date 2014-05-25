from __future__ import print_function
import random
import logging
from collections import defaultdict
import argparse
import os
from script import Feeder, MultiConsumer, IdentityCompressor, Processor, ProcessorConfiguration

model_map = {
    'BernoulliNB': BernoulliNB, 'MultinomialNB': MultinomialNB,
}

parser = argparse.ArgumentParser(description='Simple processing script')
parser.add_argument('-input_file', type=str, help='Input file')
parser.add_argument('-output_file', type=str, help='Where should the vw transformed input be stored')
parser.add_argument('-do_stemming', dest='do_stemming', action='store_true', help='Should stemming be used')
parser.add_argument('-do_dictionary', dest='do_dictionary', action='store_true', help='Should dictionary be used')
parser.add_argument('-do_multi', dest='do_multi', action='store_true', help='Should multi be used')

logging.basicConfig(level=logging.INFO, filename="vw_log", filemode="a+",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

random.seed(6355)

def check_args_consistent(args):
    if not args.input_file or not args.output_file:
        print("Input/output files not provided")
    if os.path.exists(args.input_file):
        print("Input file path exists")
        exit(1)
    if os.path.exists(args.output_file):
        print("Output file path exists")
        exit(1)

class VWConsumer:
    def __init__(self, output_file):
        self.output_file = output_file
        self.target_lookup = defaultdict(int)

    def process_row(self, features, t):
        if t not in self.target_lookup:
            self.target_lookup[t] = len(self.target_lookup) + 1
        self.output_file.write(str(self.target_lookup[t]) + " |")
        for f in features:
            self.output_file.write(" " + f + ":1.0")
        self.output_file.write("\n")

    def consume(self, objs):
        for f, t, m in objs:
            self.process_row(f, t)

    def done(self):
        self.output_file.close()

def main():
    args = parser.parse_args()

    check_args_consistent(args)

    processor_config = ProcessorConfiguration(do_dictionary=args.do_dictionary,
            do_stemming=args.do_stemming, do_multi=args.do_multi)
    processor = Processor(processor_config)
    comp = IdentityCompressor()

    consumer = MultiConsumer()
    vw_output = open(args.output_file, "w")
    consumer.add_consumer(VWConsumer(vw_output))

    with open(args.input_file, "r") as inp:
        Feeder(processor, comp, inp, None).feed(consumer)
    consumer.done()

if __name__ == "__main__":
    main()
