#!/usr/bin/python

from __future__ import division

import numpy as np
import xgboost as xgb
import doc_reader
import cPickle as pkl

def get_label_map(labels):
    a = set(labels)
    n = len(a)
    labels = list(a)
    m = dict(zip(labels, range(n)))
    return m, labels

def train_xgboost():
    '''
    train the best model
    '''
    # load corpus
    model_file = './model/clf.xgb.pkl'
    tfidf_model_file = './model/model.tfidf.dat'
    #train_file = "./raw_data/train.txt"
    #test_file = "./raw_data/val.txt" # replace as test.txt to see the final performance

    train_file = "./raw_data/train_val.txt"
    test_file = "./raw_data/test.txt"

    num_class = 7
    X_train, train_Y, feature_list1, train_docs = doc_reader.read_corpus(train_file, 25, 0, tfidf_model_file)
    X_test, test_Y, feature_list2, test_docs = doc_reader.read_corpus(test_file, 25, 0, tfidf_model_file)

    label_map, labels = get_label_map(train_Y)
    label_file = model_file + ".label"
    with open(label_file, 'wb') as f:
        pkl.dump(labels, f)

    train_Y = [label_map[k] for k in train_Y]
    test_Y = [label_map[k] for k in test_Y]

    xg_train = xgb.DMatrix(X_train, label=train_Y)
    xg_test = xgb.DMatrix(X_test, label=test_Y)

    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.07 # learning rate
    num_round = 8

    param['gamma'] = 0.2 # min_split_loss
    param['max_depth'] = 4
    param['min_child_weight'] = 2

    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.7
    param['alpha'] = 0.005
    param['silent'] = 1
    param['nthread'] = 8
    param['num_class'] = num_class

    watchlist = [(xg_train, 'train'), (xg_test, 'test')] # output results on the two datasets per round

    ## do cross validation

    # test max_depth, min_child_weight
    #for i in range(3,6,1):
    #    for j in range(2,5,1):
    #        param['max_depth'] = i
    #        param['min_child_weight'] = j
    #        res = xgb.cv(param, xg_train, num_round, nfold=5, metrics = 'merror', early_stopping_rounds=3, show_stdv=False)
    #        #print res.columns
    #        ix = np.argmin(res['test-merror-mean'].values)
    #        v = res['test-merror-mean'].values[ix]
    #        print "para:",i,j,ix,v

    # test gamma
    #for i in range(10):
    #    param['gamma'] = i/10.0
    #    res = xgb.cv(param, xg_train, num_round, nfold=5, metrics = 'merror', early_stopping_rounds=3, show_stdv=False)
    #    #print res.columns
    #    ix = np.argmin(res['test-merror-mean'].values)
    #    v = res['test-merror-mean'].values[ix]
    #    print "para:",i/10.0,ix,v

    ## test subsample, colsample_bytree
    #for i in range(6,11):
    #    for j in range(6,11):
    #        param['subsample'] = i/10.0
    #        param['colsample_bytree'] = j/10.0
    #        res = xgb.cv(param, xg_train, num_round, nfold=5, metrics = 'merror', early_stopping_rounds=3, show_stdv=False)
    #        #print res.columns
    #        ix = np.argmin(res['test-merror-mean'].values)
    #        v = res['test-merror-mean'].values[ix]
    #        print "para:",i,j,ix,v

    ## test alpha (L1 regularizaiton)
    ##for i in [1e-5, 1e-2, 0.1, 1, 100]:
    #for i in [0, 0.001, 0.005, 0.01, 0.02]:
    #    param['alpha'] = i
    #    res = xgb.cv(param, xg_train, num_round, nfold=5, metrics = 'merror', early_stopping_rounds=3, show_stdv=False)
    #    #print res.columns
    #    ix = np.argmin(res['test-merror-mean'].values)
    #    v = res['test-merror-mean'].values[ix]
    #    print "para:",i,ix,v

    # train xgboost
    bst = xgb.train(param, xg_train, num_round, watchlist)

    # save model
    with open(model_file, 'wb') as f:
        pkl.dump(bst, f)
    #bst.save_model("./model/clf.xgb.pkl")
    bst.dump_model('./model/dump.raw.txt')
    #bst.dump_model('./model/dump.nice.txt', './model/model.tfidf.dat.dict.2') #not work

    # get prediction
    pred = bst.predict(xg_test)
    acc = np.sum(pred == test_Y) / len(test_Y)
    print("test_accuracy = {}".format(acc))

    ## do the same thing again, but output probabilities
    #param['objective'] = 'multi:softprob'
    #bst = xgb.train(param, xg_train, num_round, watchlist)
    ## Note: this convention has been changed since xgboost-unity
    ## get prediction, this is in 1D array, need reshape to (ndata, nclass)
    #pred_prob = bst.predict(xg_test).reshape(len(test_Y), num_class)
    #pred_label = np.argmax(pred_prob, axis=1) #axis=1: index of the max value per row
    #error_rate = np.sum(pred_label != test_Y) / len(test_Y)
    #print('Test error using softprob = {}'.format(error_rate))

if __name__ == '__main__':
    train_xgboost()
