from caffe_cnn import *
import numpy as np
import os
import scipy
import json
import cPickle
from sklearn.feature_extraction.text import CountVectorizer
import pdb
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import requests

#Create the CNN, using the 19 layers CNN
vgg_deploy_path = '../VGG_ILSVRC_19_layers_deploy.prototxt'
vgg_model_path  = '../VGG_ILSVRC_19_layers.caffemodel'
cnn = CNN(deploy=vgg_deploy_path,
          model=vgg_model_path,
          batch_size=20,
          width=224,
          height=224)



#Get filenames for training/testing. Put your own filenames here
coco_image_path = '../'
tpath = '../train2014/'
vpath = '../val2014/'

#Get train data from the training file. Put your own filenames here
t_annFile = '../captions_train2014.json'
v_annFile = '../captions_val2014.json'

with open('./splits/coco_train.txt','r') as f:
    trainids = [x for x in f.read().splitlines()]
with open('./splits/coco_restval.txt','r') as f:
    trainids += [x for x in f.read().splitlines()]
with open('./splits/coco_val.txt','r') as f:
    valids = [x for x in f.read().splitlines()]
with open('./splits/coco_test.txt','r') as f:
    testids = [x for x in f.read().splitlines()]

#Another fast representation: by dictionary
whatType = {}
for t in trainids:
    whatType[t] = "train"
for t in valids:
    whatType[t] = "val"
for t in testids:
    whatType[t] = "test"


#Extract from json
val = json.load(open(v_annFile, 'r'))
train = json.load(open(t_annFile, 'r'))
imgs = val['images'] + train['images']
annots = val['annotations'] + train['annotations']

trainImgs = []
valImgs = []
testImgs = []


#Maps image ID to the index that the image is in
#in our later giant array of features
train_id2idx = {}
val_id2idx = {}
test_id2idx = {}
trainidx = 0
validx = 0
testidx = 0
for img in imgs:
    thetype = whatType[img['file_name']]
    if thetype == "train":
        trainImgs.append(img)
        train_id2idx[img['id']] = trainidx
        trainidx += 1
    elif thetype == "val":
        valImgs.append(img)
        val_id2idx[img['id']] = validx
        validx += 1
    elif thetype == "test":
        testImgs.append(img)
        #print img.keys()
        test_id2idx[img['id']] = testidx
        testidx += 1

#max_size = 10000
#trainImgs = trainImgs[:max_size]
#valImgs = valImgs[:max_size]
#testImgs = testImgs[:max_size]


#Go through annotations. Itoa is a dictionary
#taking in an image ID and returning 5 annotations
itoa = {}
for a in annots:
    imgid = a['image_id']
    if not imgid in itoa: itoa[imgid] = []
    itoa[imgid].append(a['caption'])

#imgList is a list of image files from the JSONs
#ind_dict maps image IDs to the index that the image will appear in the
#features matrix
def makeCaps(imgList,ind_dict):
    newCaps = []
    for timg in imgList:
        myid = timg['id']
        myidx = ind_dict[myid]
        for annot in itoa[myid]:
            newCaps.append((annot,myidx))
    return newCaps

cap_train = makeCaps(trainImgs,train_id2idx)
cap_val = makeCaps(valImgs,val_id2idx)
cap_test = makeCaps(testImgs,test_id2idx)
print "done with linking caps"


def getFilename(imgobj):
    fn = imgobj['file_name']
    if fn.startswith('COCO_val'):
        return vpath + fn
    return tpath + fn

#Processes the CNN features
def processImgList(theList,basefn):
    batch_size = 100
    numPics = 0
    batchNum = 0

    for start, end in zip(range(0, len(theList)+batch_size, batch_size), range(batch_size, len(theList)+batch_size, batch_size)):
        print("processing images %d to %d" % (start, end))
        image_files = [getFilename(x) for x in theList[start:end]]
        feat = cnn.get_features(image_list=image_files, layers='conv5_4', layer_sizes=[512,14,14])
        if numPics % batch_size == 0: #reset!
            featStacks = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
        else:
            featStacks = scipy.sparse.vstack([featStacks, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))],format="csr")
        
        numPics += 1

        if numPics % batch_size == 0:
            newfn = basefn + str(batchNum) + '.pkl'
            #newfn = basefn + '.pkl'
            with open(newfn,'wb') as f:
                cPickle.dump(featStacks, f,protocol=cPickle.HIGHEST_PROTOCOL)
                print("Success!")
            batchNum += 1

    if numPics % batch_size != 0:
        newfn = basefn + str(batchNum) + '.pkl'
        #newfn = basefn + '.pkl'
        with open(newfn,'wb') as f:
            cPickle.dump(featStacks, f,protocol=cPickle.HIGHEST_PROTOCOL)
    return featStacks


print('train now')
train_feats = processImgList(trainImgs,'./data/coco_align.train')

with open('./data/coco_align.train.pkl', 'wb') as f:
    cPickle.dump(cap_train, f,protocol=cPickle.HIGHEST_PROTOCOL)

print('val now')
val_feats = processImgList(valImgs,'./data/coco_align.val')
with open('./data/coco_align.val.pkl', 'wb') as f:
    cPickle.dump(cap_val, f,protocol=cPickle.HIGHEST_PROTOCOL)

print('test now')
test_feats = processImgList(testImgs,'./data/coco_align.test')
with open('./data/coco_align.test.pkl', 'wb') as f:
    cPickle.dump(cap_test, f,protocol=cPickle.HIGHEST_PROTOCOL)
