import cPickle as pkl
import gzip
import os
import sys
import time
import json
import numpy
import scipy
import re
import string

def parseWords(worddict,n_words,sent):
    toks = []
    #Pretty terrible, but it seems to work
    myRestring = r"[A-Za-z0-9\_]+|[" + string.punctuation +"]+"
    filteredsent = re.findall(myRestring, sent)
    for w in filteredsent:
        if w in worddict:
            if worddict[w] < n_words:
                toks.append(worddict[w])
            else:
                toks.append(1)
        else:
            print "whoops, key not in dictionary"
            print "sent:", sent
            print "word", w
    return toks



def prepare_data(caps, features, worddict, maxlen=None, n_words=10000, zero_pad=False):
    """ Formats the features/data
    """
    seqs = []
    feat_list = []
    #print "shape is:"
    #print features.shape
    for cc in caps:
        seqs.append(parseWords(worddict,n_words,cc[0]))
        #seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in cc[0].split()])
        #print cc[1]
        feat_list.append(features[cc[1]])

    lengths = [len(s) for s in seqs]

    if maxlen != None:
        new_seqs = []
        new_feat_list = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, feat_list):
            if l < maxlen:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        feat_list = new_feat_list
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    y = numpy.zeros((len(feat_list), feat_list[0].shape[1])).astype('float32')
    for idx, ff in enumerate(feat_list):
        y[idx,:] = numpy.array(ff.todense())
        #y[idx,:] = numpy.array(ff)
    y = y.reshape([y.shape[0], 14*14, 512])
    if zero_pad:
        y_pad = numpy.zeros((y.shape[0], y.shape[1]+1, y.shape[2])).astype('float32')
        y_pad[:,:-1,:] = y
        y = y_pad

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    return x, x_mask, y

def load_data(load_train=True, load_dev=True, load_test=True, path='./data/'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here IMDB)
    '''

    #############
    # LOAD DATA #
    #############

    print '... loading data'

    train = None
    valid = None
    test = None

    num_trains = 12
    num_devs = 1
    num_tests = 1

    if load_train:

        with open('./data/coco_align.train.pkl', 'rb') as f:
            train_cap = pkl.load(f)
            #train_feat = pkl.load(f)

        with open('./data/coco_align.train0.pkl', 'rb') as f:
            train_feat = pkl.load(f)
            train_feat = train_feat.tocsr()
        print "trainfeat", train_feat.shape
        for t in range(1,num_trains):

            with open('./data/coco_align.train' + str(t) + '.pkl', 'rb') as f:
                tempfeat = pkl.load(f)
            tempfeat = tempfeat.tocsr()
            print "tempfeat", tempfeat.shape
            train_feat = scipy.sparse.vstack([train_feat, tempfeat],format="csr")
        train_feat = train_feat.astype(numpy.float32)
        train = (train_cap, train_feat)

    if load_dev:
        with open('./data/coco_align.val.pkl', 'rb') as f:
            dev_cap = pkl.load(f)
            #dev_feat = pkl.load(f)

        with open('./data/coco_align.val0.pkl', 'rb') as f:
            dev_feat = pkl.load(f)
        dev_feat = dev_feat.tocsr()
        for t in range(1,num_devs):
            with open('./data/coco_align.val' + str(t) + '.pkl', 'rb') as f:
                tempfeat = pkl.load(f)
            tempfeat = tempfeat.tocsr()
            dev_feat = scipy.sparse.vstack([dev_feat, tempfeat],format="csr")
        dev_feat = dev_feat.astype(numpy.float32)
        valid = (dev_cap, dev_feat)

    if load_test:
        with open('./data/coco_align.test.pkl', 'rb') as f:
            test_cap = pkl.load(f)
            #test_feat = pkl.load(f)
        
        with open('./data/coco_align.test0.pkl', 'rb') as f:
            test_feat = pkl.load(f)
        test_feat = test_feat.tocsr()

        for t in range(1,num_tests):
            with open('./data/coco_align.test' + str(t) + '.pkl', 'rb') as f:
                tempfeat = pkl.load(f)
            tempfeat = tempfeat.tocsr()
            test_feat = scipy.sparse.vstack([test_feat, tempfeat],format="csr")
        test_feat = test_feat.astype(numpy.float32)
        test = (test_cap, test_feat)

    with open(path+'coco_dictionary.pkl', 'rb') as f:
        worddict = pkl.load(f)


    sys.stdout.flush()
    return train, valid, test, worddict
