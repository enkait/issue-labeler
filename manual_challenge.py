import random
import json
import re
import string

L = []
L = open("data_list").readlines()

points = 0
processed = 0
while True:
    line = random.choice(L)
    obj = json.loads(line.strip())
    title = re.sub("[^" + string.printable + "]", "", obj['title'])
    body = re.sub("[^" + string.printable + "]", "", obj['body'])
    print "Title: ", title
    print "Body: ", body
    while True:
        print "Output b (bug), e (enhancement) or q (question)"
        res = raw_input()
        res = res.strip()
        if res in "ebq":
            break
    for label in obj['labels']:
        if res == 'e' \
                and (label['name'].lower() == 'enhancement' or label['name'].lower() == 'feature'):
            points += 1
            break
        elif res == 'b' and label['name'].lower() == 'bug':
            points += 1
            break
        elif res == 'q' and label['name'].lower() == 'question':
            points += 1
            break
    processed += 1
    print "Result: %s out of %s" % (points, processed)
