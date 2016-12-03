import re

import numpy
from nltk import word_tokenize
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, average_precision_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
import math

cachedStopWords = stopwords.words("english")


def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens));
    return filtered_tokens


def get_svm_classifier():
    svm_classifier = Pipeline([
        ('vectorizer',
         TfidfVectorizer(tokenizer=tokenize, min_df=0, max_df=0.90, max_features=3000, use_idf=True,
                         sublinear_tf=True)),
        ('clf', OneVsRestClassifier(LinearSVC()))])
    return svm_classifier


def main():
    class_binarizer = MultiLabelBinarizer(classes=reuters.categories())

    input_train = list()
    output_train = list()
    input_test = list()
    output_test = list()

    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            input_train.append(reuters.raw(doc_id))
            output_train.append(reuters.categories(doc_id))
        else:
            input_test.append(reuters.raw(doc_id))
            output_test.append(reuters.categories(doc_id))

    input_train = numpy.array(input_train)
    output_train = numpy.array(output_train)
    input_test = numpy.array(input_test)
    output_test = numpy.array(output_test)

    classifier = get_svm_classifier()

    classifier.fit(input_train, class_binarizer.fit_transform(output_train))

    res = classifier.predict(input_test)

    hard_precision = classifier.score(input_test, class_binarizer.transform(output_test))

    precision_scores = average_precision_score(res, class_binarizer.fit_transform(output_test), average=None)

    recall_scores = recall_score(res, class_binarizer.fit_transform(output_test), average=None)

    f1_scores = f1_score(res, class_binarizer.fit_transform(output_test), average=None)

    print("Hard precision: " + str(hard_precision))

    print("Precision scores: " + str(precision_scores))

    print("Recall scores: " + str(recall_scores))

    print("F1 scores: " + str(f1_scores))

    pr_tot = 0
    pr_cnt = 0
    for pr in precision_scores:
        if not math.isnan(pr):
            pr_tot += pr
            pr_cnt += 1
    print("Average Precision: " + str(pr_tot / pr_cnt))

    rc_tot = 0
    rc_cnt = 0
    for rc in recall_scores:
        if not rc == 0:
            rc_tot += rc
            rc_cnt += 1
    print("Average Recall: " + str(rc_tot / rc_cnt))


if __name__ == '__main__':
    main()
