from loaddata import get_train, get_cv, print_stats
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

def calc_weights(Y):
    Ya = np.array(Y)
    res = np.array([0, 0, 0])
    for i in range(3):
        res[i] = sum(Ya == (i + 1))
    res = res.astype(np.float64)
    res = sum(res) / res
    print res
    W = map(lambda y: res[y-1], Y)
    return W

def main():
    vectorizer = CountVectorizer(stop_words="english",
            token_pattern=r"(?u)\b(?:\w|\?)(?:\w|\?)+\b", ngram_range=(1,2))
    (X, Y) = get_train()
    (Xcv, Ycv) = get_cv()
    Xt = vectorizer.fit_transform(X)
    Xcvt = vectorizer.transform(Xcv)
    model = MultinomialNB(fit_prior=False)
    #model.fit(Xt, Y, calc_weights(Y))
    model.fit(Xt, Y)
    Pcv = model.predict(Xcvt)
    print_stats(Ycv, Pcv)

if __name__ == "__main__":
    main()

