from loaddata import get_train, get_cv, print_stats
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif

def main():
    vectorizer = HashingVectorizer(stop_words="english", binary=True,
            tokenizer=lambda text: text.split(),
            token_pattern=r"(?u)\b(?:\w|\?)(?:\w|\?)+\b",
            ngram_range=(1,2))
    (X, Y) = get_train()
    (Xcv, Ycv) = get_cv()
    Xt = vectorizer.fit_transform(X)
    #selector = SelectPercentile(f_classif, percentile=40)
    #Xtt = selector.fit_transform(Xt, Y)
    Xtt = Xt
    Xcvt = vectorizer.transform(Xcv)
    #Xcvtt = selector.transform(Xcvt)
    Xcvtt = Xcvt
    #model = LinearSVC()
    #model = SVC(kernel='rbf', gamma=1.0, cache_size=1000)
    model = MultinomialNB(fit_prior=False)
    model.fit(Xtt, Y)
    Pcv = model.predict(Xcvtt)
    print_stats(Ycv, Pcv)

if __name__ == "__main__":
    main()

