#! /usr/bin/python

from __future__ import print_function
import random
import logging
from collections import defaultdict
import argparse
import os
import math
import subprocess
import sys
from script import Feeder, MultiConsumer, Processor, ProcessorConfiguration
from script import add_parser_arguments as script_add_parser_arguments

random.seed(6355)

def add_parser_arguments(parser):
    parser.add_argument('-input_file', type=str, help='Input file')
    parser.add_argument('-output_file', type=str, help='Where should the vw transformed input be stored')
    parser.add_argument('-split', type=int, help='Number of divisions to split the work to')
    parser.add_argument('-worker', dest='worker', action='store_true', help='Are we a worker')
    parser.add_argument('-worker_skip', type=int, help='How many lines to skip at start')
    parser.add_argument('-worker_size', type=int, help='How many lines to process')
    parser.add_argument('-worker_id', type=int, help='Worker id')

logging.basicConfig(level=logging.INFO, filename="vw_log", filemode="a+", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def check_args_consistent(args):
    if args.worker:
        if args.worker_size == 0:
            print("Missing worker size")
            exit(1)
    else:
        if not args.split:
            args.split = 1
    if not args.input_file or not args.output_file:
        print("Input/output files not provided")
        exit(1)
    if not os.path.exists(args.input_file):
        print("Input file path doesn't exist")
        exit(1)
    if os.path.exists(args.output_file):
        print("Output file path exists")
        exit(1)

class IdentityCompressor:
    def compress(self, L):
        return L

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

def count_lines(file_path):
    lines = 0
    with open(file_path, "r") as inp:
        while inp.readline():
            lines += 1
    return lines

def main():
    parser = argparse.ArgumentParser(description='Simple processing script')
    add_parser_arguments(parser)
    script_add_parser_arguments(parser)
    args = parser.parse_args()

    check_args_consistent(args)

    if args.worker:
        logging.info("Running worker: " + str(sys.argv))

        processor_config = ProcessorConfiguration.FromArgs(args)
        processor = Processor(processor_config)
        comp = IdentityCompressor()

        consumer = MultiConsumer()
        vw_output = open(args.output_file + "." + str(args.worker_id), "w")
        consumer.add_consumer(VWConsumer(vw_output))

        with open(args.input_file, "r") as inp:
            Feeder(processor, comp, inp, args.worker_size, args.worker_skip).feed(consumer)
        consumer.done()
        logging.info("Worker: " + str(sys.argv) + " finished")

    else:
        line_count = count_lines(args.input_file)
        split_size = int(math.ceil(1.0 * line_count / args.split))

        cur_skip = 0
        processes = []
        for i in range(args.split):
            args = sys.argv[:]
            args[0] = "./" + args[0]
            args.append("-worker")
            args.append("-worker_skip")
            args.append(str(cur_skip))
            args.append("-worker_size")
            args.append(str(split_size))
            args.append("-worker_id")
            args.append(str(i))
            cur_skip += split_size
            processes.append(subprocess.Popen(args))

        for process in processes:
            process.wait()

if __name__ == "__main__":
    main()
