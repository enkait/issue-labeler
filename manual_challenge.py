import random
import json
import re
import string

L = []
for line in open("data_list"):
    if random.random() < 0.01:
        L.append(line)

filL = random.sample(L, 50)

points = 0
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
                and (label['name'] == 'enhancement' or label['name'] == 'feature'):
            points += 1
            break
        elif res != 'enhancement' and label['name'] == res:
            points += 1
            break
    print "Result: ", points

print "Final result: ", points
