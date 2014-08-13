""" 
    Lemmatizes body and title of preprocessed issues.
"""
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from loaddata import raw_load_data
from itertools import chain
import sys
import string

if (len(sys.argv) < 3):
    print "Supply source and destination arguments"
    exit()

src = sys.argv[1]
dst = sys.argv[2]

wnl = WordNetLemmatizer()

def process(text):
    for p in string.punctuation:
        if p != "'" and p != "-":
            text = text.replace(p, " ")
        else:
            text = text.replace(p, "")
    return " ".join(map(wnl.lemmatize, text.split()))

with open(src, "r") as src_file:
    with open(dst, "w") as dst_file:
        data = zip(*raw_load_data(src_file))
        for (title, body, result) in data:
            ntitle = process(title)
            dst_file.write(ntitle + "\n")
            nbody = process(body)
            dst_file.write(nbody + "\n")
            dst_file.write(str(result) + "\n")
