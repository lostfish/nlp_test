#! /usr/bin/env python
#encoding: utf-8

import sys
import os
import re
import logging
import numpy as np
from time import time

import xgboost as xgb
from sklearn import metrics
import doc_reader
import cPickle as pkl

def do_predict(corpus, model_file):
    '''
    load model to predict
    '''
    clf = None
    with open(model_file, 'rb') as f:
        clf = pkl.load(f)
    if not clf:
        return
    labels = []
    with open(model_file+".label", 'rb') as f:
        labels = pkl.load(f)
    xg_matrix = xgb.DMatrix(corpus)
    pred = clf.predict(xg_matrix)
    pred = [labels[int(t)] for t in pred]
    return pred

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
    clf_model_file = './model/clf.xgb.pkl'
    label_field = -1
    #label_field = 0
    predict_file(tfidf_model_file, clf_model_file, test_file, seg_field, label_field, out_file)
