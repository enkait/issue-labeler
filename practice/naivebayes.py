from loaddata import get_train, get_cv, print_stats
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

def main():
    vectorizer = CountVectorizer(stop_words="english",
            token_pattern=r"(?u)\b(?:\w|\?)(?:\w|\?)+\b", ngram_range=(1,2))
    (X, Y) = get_train()
    (Xcv, Ycv) = get_cv()
    Xt = vectorizer.fit_transform(X)
    Xcvt = vectorizer.transform(Xcv)
    model = MultinomialNB(fit_prior=False)
    model.fit(Xt, Y)
    Pcv = model.predict(Xcvt)
    print_stats(Ycv, Pcv)

if __name__ == "__main__":
    main()

