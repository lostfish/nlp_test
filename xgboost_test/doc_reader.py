#! /usr/bin/env python
#encoding: utf-8

import sys
import os
import re
import logging
import time
import math
import scipy.sparse as sp
import numpy as np
from six import iteritems
from gensim import corpora, models, matutils
from pprint import pprint

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
        filename='1.log',
        filemode='w',
        level=logging.INFO)

def get_valid_vocab(dictionary, charset='utf-8'):
    '''
    '''
    vocab = range(len(dictionary))
    for k,v in dictionary.iteritems():
        vocab[k] = v.encode(charset)
    return vocab

def get_doc_repr(doc_info, vocab):
    '''
    get doc feature representation
    '''
    a = ["%s:%.3f" % (vocab[ix],v) for ix, v in doc_info]
    return ' '.join(a)

def load_words(infile):
    word_set = set()
    if os.path.isfile(infile):
        with open(infile) as f:
            for line in f:
                word = line.strip().split('\t')[0]
                word_set.add(word)
    return word_set

def extract_feature(word_str, stop_set, is_title = True):
    '''
    extract feature from seg word field
    '''
    g_core_set = load_words("./conf/short_core_words.txt") # TODO
    bad_pos = set(["c", "t", "r", "ad", "d", "f", "w", "o", "y", "p", "u", "q", "m"])

    words = word_str.lower().split("||") #format: hello@nx||world@nx
    n = len(words)/2
    if not is_title:
        words = words[:n]
    word_set = set()
    results = []
    for x in words:
        i = x.rfind('@')
        if i == -1:
            continue
        word = x[:i]
        pos = x[i+1:]
        if pos in bad_pos:
            continue
        if word in stop_set:
            continue
        if len(word) < 2:
            continue
        if len(word) < 4:
            if not word.isalpha() and word not in g_core_set:
                continue
        if word.find("http") != -1:
            continue
        #if word in word_set: #uniq
        #    continue
        #word_set.add(word)
        results.append(word)
    return results

def convert_csr(m, shape):
    '''
    convert gensim corpus object to scipy.sparse.csr.csr_matrix
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    '''
    indptr = [0]
    indices = []
    data = []
    for vec in m:
        for k,v in vec:
            index = k
            indices.append(index)
            data.append(v)
        indptr.append(len(indices))
    return sp.csr_matrix((data, indices, indptr), shape = shape, dtype=np.float64)

def choose_feature(X_train, y_train, k):
    '''
    choose feature by chi2
    '''
    from sklearn.feature_selection import SelectKBest, chi2
    ch2 = SelectKBest(chi2, k)
    X_train = ch2.fit_transform(X_train, y_train)
    return ch2.get_support(indices=True)

def build_new_tfidf_model(infile, seg_field, label_field, stop_file, model_file, choose_num, is_title = True, least_df = 2, charset = 'utf-8'):
    '''
    build tfidf model by choosing feature
    '''
    stop_set = load_words(stop_file)

    class MyCorpus(object):
        def __init__(self, fname):
            self.fname = fname
        def __iter__(self):
            for i,line in enumerate(open(self.fname)):
                s = line.rstrip('\n').split('\t')
                yield extract_feature(s[seg_field], stop_set, is_title)

    corp = MyCorpus(infile)
    dictionary = corpora.Dictionary(corp)

    filter_ids = []
    if least_df > 1:
        filter_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq < least_df]
    print "filter_ids: ",len(filter_ids)

    dictionary.filter_tokens(filter_ids)
    dictionary.compactify()
    print "feature num:", len(dictionary)

    # train firstly
    corpus = [dictionary.doc2bow(text) for text in corp]
    tfidf = models.TfidfModel(corpus)

    # choose feature
    X_train = tfidf[corpus]
    shape = (len(X_train), len(dictionary))
    X_train = convert_csr(X_train, shape)

    y_train = get_labels(infile, label_field)
    choose_ids = choose_feature(X_train, y_train, choose_num) # numpy.ndarray
    stop_ids = set(dictionary.keys()) - set(choose_ids)
    print len(stop_ids)
    dictionary.filter_tokens(stop_ids)
    dictionary.compactify()
    print "feature num:", len(dictionary)

    # train secondly
    corpus = [dictionary.doc2bow(text) for text in corp]
    tfidf = models.TfidfModel(corpus)

    # save
    tfidf.save(model_file)
    dict_file = model_file + ".dict"
    dictionary.save(dict_file)

    dict_file2 = model_file + ".dict.2" #readable
    vocab = get_valid_vocab(dictionary)
    fout = open(dict_file2, 'w')
    for i,v in enumerate(vocab):
        fout.write("%d\t%s\n" % (i,v))
    fout.close()

def build_tfidf_model(infile, seg_field, stop_file, model_file, is_title = True, least_df = 2, charset = 'utf-8'):
    '''
    build tfidf model
    '''
    stop_set = load_words(stop_file)

    class MyCorpus(object):
        def __init__(self, fname):
            self.fname = fname
        def __iter__(self):
            for i,line in enumerate(open(self.fname)):
                s = line.rstrip('\n').split('\t')
                yield extract_feature(s[seg_field], stop_set, is_title)

    corp = MyCorpus(infile)
    dictionary = corpora.Dictionary(corp)

    #no need, have been filtered in extract_feature()
    #stoplist = [w.decode(charset) for w in stop_set]
    #stop_ids = [dictionary.token2id[w] for w in stoplist if w in dictionary.token2id]
    #print "stop_ids: ",len(stop_ids)

    filter_ids = []
    if least_df > 1:
        filter_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq < least_df]
    print "filter_ids: ",len(filter_ids)

    dictionary.filter_tokens(filter_ids)
    dictionary.compactify()
    print "feature num:", len(dictionary)

    corpus = [dictionary.doc2bow(text) for text in corp]
    tfidf = models.TfidfModel(corpus)
    tfidf.save(model_file)
    dict_file = model_file + ".dict"
    dictionary.save(dict_file)

    dict_file2 = model_file + ".dict.2" #readable
    vocab = get_valid_vocab(dictionary)
    fout = open(dict_file2, 'w')
    for i,v in enumerate(vocab):
        fout.write("%d\t%s\n" % (i,v))
    fout.close()

def get_corpus(infile, seg_field, is_title=True):
    '''
    '''
    class MyCorpus(object):
        def __init__(self, fname):
            self.fname = fname
        def __iter__(self):
            for i,line in enumerate(open(self.fname)):
                s = line.rstrip('\n').split('\t')
                yield extract_feature(s[seg_field], set(), is_title)
    return  MyCorpus(infile)

def get_labels(infile, label_field):
    '''
    read labels
    '''
    y = []
    with open(infile) as f:
        if label_field == -1:
            for line in f:
                y.append("")
        else:
            for line in f:
                line = line.rstrip('\n')
                s = line.split('\t')
                y.append(s[label_field])
    return y

def get_tfidf_corpus(corp, model_file):
    '''
    convert corpus to tf-idf vector

    '''
    #load model & dictionary
    tfidf = models.TfidfModel.load(model_file)
    dict_file = model_file + ".dict"
    dictionary = corpora.Dictionary.load(dict_file)

    #convert
    corpus = [dictionary.doc2bow(text) for text in corp]
    final_corpus = tfidf[corpus]

    vocab = get_valid_vocab(dictionary)
    feature_list = [get_doc_repr(doc_info, vocab) for doc_info in final_corpus]

    shape = (len(final_corpus), len(dictionary))
    final_corpus = convert_csr(final_corpus, shape)
    return final_corpus, feature_list

def read_corpus(infile, seg_field, label_field, model_file):
    '''
    read and represent documents by tf-idf model
    '''
    corpus = get_corpus(infile, seg_field)
    labels = get_labels(infile, label_field)
    final_corpus, feature_list = get_tfidf_corpus(corpus, model_file)
    docs = []
    for line in file(infile):
        line = line.rstrip('\n')
        docs.append(line)
    return final_corpus, labels, feature_list, docs

def test():
    '''
    build tf-idf model
    '''
    big_file = "./raw_data/day7.uniq"
    stop_file = './conf/stopwords.u8'
    model_file = './model/model.tfidf.dat'
    seg_field = 25
    label_field = 0

    # use all feature, not need labels
    build_tfidf_model(big_file, seg_field, stop_file, model_file)

    # use choosed features, need labels
    #build_new_tfidf_model(big_file, seg_field, label_field, stop_file, model_file, 2000)

    # test read
    #read_corpus(infile, seg_field, 0, model_file)

if __name__ == '__main__':
    test()
