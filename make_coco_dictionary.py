# Setup 

originalImagesPath = '../'
preprocessedImagesPath = '../processedImages/'

caffe_root = '/opt/caffe/python/'

vgg_ilsvrc_19_layoutFileName = '../vgg_ilsvrc_19/VGG_ILSVRC_19_layers_deploy.prototxt'
vgg_ilsvrc_19_modelFileName = '../vgg_ilsvrc_19/VGG_ILSVRC_19_layers.caffemodel'

dataPath = '../'
annotation_path = dataPath + 'captions_train2014.json'
splitFileName = dataPath + 'dataset_coco.json'

experimentPrefix = '.exp1'

# Import

import pdb
from sys import stdout
import scipy
import  cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt


import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

import os

import pandas as pd
import nltk


# Create file list 
# coco.devImages.txt 
# coco.trainImages.txt 
# coco.testImages.txt

import json
from pprint import pprint


with open(splitFileName) as f:
    data = json.load(f)

df = pd.DataFrame(data['images'])

files = [ 'dev','test','train']

dataDict = {}

dataDict['dev'] = df[df.split == 'val']
dataDict['test'] = df[df.split == 'test']
dataDict['restval'] = df[df.split == 'restval']
dataDict['train'] = df[df.split == 'train']

for f in files:
    dataDict[f]['filename'].to_csv(dataPath + 'coco.' + f + 'Images.txt',index=False)
    

def buildCapDict(sentences):
    return [s[u'raw'] for s in sentences ]

df['captions'] = df.apply(lambda row: buildCapDict(row['sentences']), axis=1)

capDict = df.loc[:,['filename', 'captions']].set_index('filename').to_dict()

capDict = capDict['captions']

# Let's make dictionary

corpus = df['captions'].values
corpus2 = [' '.join(c) for c in corpus]
corpus3 = ' '.join(corpus2)

words = nltk.FreqDist(corpus3.split()).most_common()

wordsDict = {words[i][0]:i+2 for i in range(len(words))}

with open(dataPath + 'dictionary.pkl', 'wb') as f:
    pickle.dump(wordsDict, f)
