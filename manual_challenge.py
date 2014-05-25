import random
import json
import re
import string

L = []
for line in open("data_list"):
    if random.random() < 0.01:
        L.append(line)

filL = random.sample(L, 100)

points = 0
processed = 0
for line in filL:
    obj = json.loads(line.strip())
    title = re.sub("[^" + string.printable + "]", "", obj['title'])
    body = re.sub("[^" + string.printable + "]", "", obj['body'])
    print "Title: ", title
    print "Body: ", body
    print "Output bug, enhancement or question"
    res = raw_input()
    res = res.strip()
    for label in obj['labels']:
        if res == 'enhancement' \
                and (label['name'].lower() == 'enhancement' or label['name'].lower() == 'feature'):
            points += 1
            break
        elif res != 'enhancement' and label['name'].lower() == res:
            points += 1
            break
    processed += 1
    print "Result: %s out of %s" % (points, processed)

print "Final result: ", points
