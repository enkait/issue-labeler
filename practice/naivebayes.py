from loaddata import get_train, get_cv, print_stats
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def main():
    vectorizer = HashingVectorizer(stop_words="english", binary=True,
            tokenizer=LemmaTokenizer(),
            token_pattern=r"(?u)\b(?:\w|\?)(?:\w|\?)+\b", ngram_range=(1,2))
    (X, Y) = get_train()
    (Xcv, Ycv) = get_cv()
    Xt = vectorizer.fit_transform(X)
    Xcvt = vectorizer.transform(Xcv)
    model = LogisticRegression(C=2.0, penalty='l1')
    model.fit(Xt, Y)
    Pcv = model.predict(Xcvt)
    print_stats(Ycv, Pcv)

if __name__ == "__main__":
    main()

