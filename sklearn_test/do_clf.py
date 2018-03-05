#! /usr/bin/env python
#encoding: utf-8

import sys
import os
import re
import logging
import numpy as np
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from collections import defaultdict

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn.pipeline import Pipeline
from sklearn import metrics
import doc_reader
import cPickle as pkl

def do_train(X_train, y_train, clf, model_file):
    '''
    train classification model
    '''
    clf.fit(X_train, y_train)
    with open(model_file, 'wb') as f:
        pkl.dump(clf, f)

def do_predict(corpus, model_file):
    '''
    load model to predict
    '''
    clf = None
    with open(model_file, 'rb') as f:
        clf = pkl.load(f)
    if not clf:
        return
    pred = clf.predict(corpus)
    #x = clf.predict_proba(corpus)
    #for i in x:
    #    print i
    return pred

def train_and_validate():
    '''
    train the best model
    '''
    # load corpus
    tfidf_model_file = './model/model.tfidf.dat'
    train_file = "./raw_data/train_val.txt"
    test_file = "./raw_data/test.txt" # replace as test.txt to see the final performance
    X_train, y_train, feature_list1, train_docs = doc_reader.read_corpus(train_file, 25, 0, tfidf_model_file)
    X_test, y_test, feature_list2, test_docs = doc_reader.read_corpus(test_file, 25, 0, tfidf_model_file)

    # train model
    tag = "final"
    clf_model_file = "./model/clf.%s.pkl" % tag
    #clf = MultinomialNB(alpha=0.01)
    #clf = KNeighborsClassifier(n_neighbors=10)
    #clf = RandomForestClassifier(n_estimators=100)
    clf = LinearSVC(penalty="l2", dual=False, tol=1e-4, C=1.0) # the best
    do_train(X_train, y_train, clf, clf_model_file)

    # predict train data
    pred = do_predict(X_train, clf_model_file)
    score = metrics.accuracy_score(y_train, pred)
    print("train_accuracy: %0.3f" % score)

    # predict test data
    pred = do_predict(X_test, clf_model_file)
    score = metrics.accuracy_score(y_test, pred)
    print("test_accuracy: %0.3f" % score)
    print_report = 1
    print_cm = 0
    if print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred))
    if print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    #for label, line in zip(pred, test_docs):
    #    print "%s\t%s" % (label, line)

def predict_file(tfidf_model_file, clf_model_file, test_file, seg_field, label_field, out_file):
    '''
    '''
    X_test, y_test, feature_list2, test_docs = doc_reader.read_corpus(test_file, seg_field, label_field, tfidf_model_file)
    pred = do_predict(X_test, clf_model_file)

    with open(out_file, 'w') as f:
        for i,cat in enumerate(pred):
            f.write("%s\t%s\n" % (test_docs[i], cat))
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "usage: %s <input_file> <seg_field> <out_file>" % __file__
        sys.exit(-1)

    test_file = sys.argv[1]
    seg_field = int(sys.argv[2])
    out_file = sys.argv[3]

    tfidf_model_file = './model/model.tfidf.dat'
    clf_model_file = './model/clf.ver1.pkl'
    label_field = -1
    #label_field = 0
    predict_file(tfidf_model_file, clf_model_file, test_file, seg_field, label_field, out_file)
