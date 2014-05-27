good = [0, 0, 0]
retrieved = [0, 0, 0]
expected = [0, 0, 0]
with open("expected_test", "r") as expected_in:
    with open("got_test", "r") as got_in:
        for a, b in zip(expected_in.readlines(), got_in.readlines()):
            a, b = int(a.strip()) - 1, int(b.strip()) - 1
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

