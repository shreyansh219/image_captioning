#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import time
import glob
from PIL import Image
from sklearn import svm

# Make sure that caffe is on the python path:
caffe_root = '../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

#--------------------------------------------------------------------------------------
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#---------------------------------------------------------------------------------------
# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)



if __name__=='__main__':
    caffe.set_mode_gpu()
    net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                    caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


    all_feat = []
    y = []
    count = 0
    img_path = '/home/shreyansh/bird_workspace/mml_project/coco/images/test2014/'


    start_time = time.time()
    for files in sorted(os.listdir(img_path)):

        # Check Limits
        count = count + 1
        if(count>10000):
        	break

    	print("Image Number {}".format(count))

        # Read Image
        net.blobs['data'].reshape(1,3,227,227)
        filename = img_path + files #'COCO_test2014_%012d.jpg'%(i,)
        print("Reading Image {}".format(filename))
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(filename))


        # Forward Propoagate
        out = net.forward()

        # Exctract Features
        #layer_list = ['data', 'conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2', 'conv3', 'conv4', 'conv5', 'pool5']       
        # layer_list = ['conv3'];
        # feat_list = []
        # for k in layer_list:
        #     v = net.blobs[k]
        #     for t in range(v.data.shape[1]):
        #         #feat = np.array((Image.fromarray(v.data[0,t,:,:])).resize((227, 227), Image.NEAREST), copy=True)
        #         feat = np.array(v.data[0,t,:,:], copy=True)
        #         feat = feat.flatten()
        #         feat_list.append(feat)

        # image_feat = np.vstack(feat_list)
        # image_feat = image_feat.flatten()
        # all_feat.append(image_feat)
        
        # # Extract fc7 Features of the image
        layer_list = ['fc7'];
        start_time = time.time()
        feat_list = []
        for k in layer_list:
            v = net.blobs[k]
            feat = np.array(v.data[0,:])
            feat_list.append(feat)

        image_feat = np.vstack(feat_list)
        all_feat.append(image_feat)

        

    X = np.vstack(all_feat)
    print("Feature Space is {}.".format(X.shape))

    elapsed_time = time.time() - start_time
    print("Feature Computation took {} time.".format(elapsed_time))

    np.savetxt('feat_fc7_10k.txt', X, delimiter='\t')

    #pdb.set_trace()




