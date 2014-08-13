import argparse
from sklearn.metrics import f1_score

def add_parser_arguments(parser):
    parser.add_argument('-train_file', type=str, help='train file')
    parser.add_argument('-cv_file', type=str, help='cv file')

parser = argparse.ArgumentParser()
add_parser_arguments(parser)
args = parser.parse_known_args()[0]

def group(gen, limit=None):
    cur = 0
    while True:
        (title, body, res) = (next(gen).strip(), next(gen).strip(), next(gen).strip())
        for result in res.split():
            result = int(result)
            yield (title + " " + body, result)
            cur += 1
            if limit != None and cur >= limit:
                return

def load_data(f, limit=None):
    data = []
    return zip(*(list(group(iter(f.readlines()), limit))))

def raw_group(gen):
    while True:
        (title, body, res) = (next(gen).strip(), next(gen).strip(), next(gen).strip())
        for result in res.split():
            result = int(result)
            yield (title, body, result)

def raw_load_data(f):
    data = []
    return zip(*(list(raw_group(iter(f.readlines())))))

def get_train(limit=None):
    return load_data(open(args.train_file, "r"), limit)

def get_cv(limit=None):
    return load_data(open(args.cv_file, "r"), limit)

def f_score(good, retrieved, expected):
    if retrieved == 0: return None
    if expected == 0: return None
    prec = good * 1.0 / retrieved
    rec = good * 1.0 / expected
    return 2 * (prec * rec) / (prec + rec)

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
            "recall -", format_ratio(good[i], expected[i]), \
            "fscore - ", f_score(good[i], retrieved[i], expected[i])
    print "Total fscore:", f1_score(expected_in, got_in)

