import argparse

def add_parser_arguments(parser):
    parser.add_argument('-train_file', type=str, help='train file')
    parser.add_argument('-cv_file', type=str, help='cv file')

parser = argparse.ArgumentParser()
add_parser_arguments(parser)
args = parser.parse_known_args()

def group(gen):
    while True:
        (title, body, res) = (next(gen).strip(), next(gen).strip(), next(gen).strip())
        for result in res.split():
            result = int(result)
            yield (title + " " + body, result)

def load_data(f):
    data = []
    return zip(*(list(group(iter(f.readlines())))))

def raw_group(gen):
    while True:
        (title, body, res) = (next(gen).strip(), next(gen).strip(), next(gen).strip())
        for result in res.split():
            result = int(result)
            yield (title, body, result)

def raw_load_data(f):
    data = []
    return zip(*(list(raw_group(iter(f.readlines())))))

def get_train():
    return load_data(open(args.train_file, "r"))

def get_cv():
    return load_data(open(args.cv_file, "r"))

def print_stats(expected_in, got_in):
    good = [0, 0, 0]
    retrieved = [0, 0, 0]
    expected = [0, 0, 0]

    for a, b in zip(expected_in, got_in):
        a, b = a - 1, b - 1
        if a == b:
            good[a] += 1
        expected[a] += 1
        retrieved[b] += 1

    def format_ratio(a, b):
        if b != 0:
            return "(%s/%s = %s)" % (a, b, a * 1.0 / b)
        return "(%s/0 = inf)" % (a,)

    print "Total good:", format_ratio(sum(good), sum(expected))
    for i in range(3):
        print i, ": precission -", format_ratio(good[i], retrieved[i]), \
            "recall -", format_ratio(good[i], expected[i])

