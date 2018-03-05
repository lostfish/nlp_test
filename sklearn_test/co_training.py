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

def two_clf_cotraining():
    '''
    co-training by two classifiers
    '''
    #load corpus
    tfidf_model_file = './model/model.tfidf.dat' #train before use here
    train_file = "./raw_data/train.txt"
    test_file = "./raw_data/unlabel.txt"
    seg_field = 10
    label_field = 0

    X_train, y_train, feature_list1, train_lines = doc_reader.read_corpus(train_file, seg_field, label_field, tfidf_model_file)
    X_test, y_test, feature_list2, test_lines = doc_reader.read_corpus(test_file, seg_field, label_field, tfidf_model_file)
    X_test_cp = X_test

    #raw data
    train_docs = list(doc_reader.get_corpus(train_file, seg_field))
    test_docs = list(doc_reader.get_corpus(test_file, seg_field))

    #co-training
    clf1 = MultinomialNB(alpha=.01)
    clf2 = KNeighborsClassifier(n_neighbors=5)
    count = 0
    while True:
        count += 1
        print "iter %d: %d %d" % (count, len(test_docs), len(train_docs))

        clf1.fit(X_train, y_train)
        y1 = clf1.predict(X_test)

        clf2.fit(X_train, y_train)
        y2 = clf2.predict(X_test)

        a =set()
        n = len(test_docs)
        for i in range(n):
            if y1[i] == y2[i]:
                train_docs.append(test_docs[i])
                y_train.append(y1[i])
                a.add(i)

        if len(a) == 0: #no new docs
            break

        remain_docs = []
        for i in range(n):
            if i not in a:
                remain_docs.append(test_docs[i])
        test_docs = remain_docs

        if len(test_docs) == 0:
            break

        X_train,feat_train = doc_reader.get_tfidf_corpus(train_docs, tfidf_model_file)
        train_num = X_train.shape[0]
        X_test,feat_test = doc_reader.get_tfidf_corpus(test_docs, tfidf_model_file)
        print X_train.shape
        print X_test.shape


    #final train and predict
    clf_list = [(MultinomialNB(alpha=.01), 'co_nb'),]
    #clf_list = [(KNeighborsClassifier(n_neighbors=5), 'co_knn')]
    for clf,name in clf_list:
        clf_model_file = "./model/clf.%s.pkl" % name
        do_train(X_train, y_train, clf, clf_model_file)
        #predict
        print X_test_cp.shape
        pred = do_predict(X_test_cp, clf_model_file)
        if len(pred) != len(test_lines):
            print "wrong: len(pred) != len(test_lines)"
            sys.exit(-1)
        for label, line in zip(pred, test_lines):
            print "%s\t%s" % (label, line)

if __name__ == '__main__':
    two_clf_cotraining()
