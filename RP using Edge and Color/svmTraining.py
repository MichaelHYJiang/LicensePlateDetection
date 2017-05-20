#!/usr/bin/python
# -*- coding: uft-8 -*-

# Using sampled pictures to train a SVM
# based on file svm.py, which came from the book 'Machine Learning in Action'

import svm
import cv2
import os
import numpy as np
import time

# Number of all positive samples
POS = 503
# Number of all negative samples
NEG = 11453

# function 'loadimgdata' reads all positive and negative samples into a variable named 'data'
# along with a variable named 'label' to denote the category of each sample vector
def loadimgdata(quick = True,   # when enabled, the number of negative samples will be contorled in 2 to 3 times of the positive ones
                width = 20,
                height = 5,
                channel = 3):   # width, height and channel of the region that samples will be resized to, their multiplication is the size of each data vector
    
    n_pic_pos = POS
    n_pic_neg = NEG
    label = []
    data = []
    t = time.time()
    n0 = 0
    for i in range( n_pic_pos):
        filepath = os.path.join( '.', 'Feature Training', 'POS', str( i) + '.jpg')
        img = cv2.imread( filepath)
        img = cv2.resize( img, ( width, height), interpolation = cv2.INTER_CUBIC)
        if channel != 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        s = img.reshape(( width * height * channel,))
        data.append( s)
        label.append( 1.0)
        n0 += 1
    print 'load %d positive samples' % n0
    if quick:
        p = n_pic_pos * 2.5 / float(n_pic_neg)
    else:
        p = 1.0
    n0 = 0
    for i in range( n_pic_neg):
        if np.random.rand() > p:
            continue
        filepath = os.path.join( '.', 'Feature Training', 'negative', str( i) + '.jpg')
        img = cv2.imread( filepath)
        img = cv2.resize( img, ( width, height), interpolation = cv2.INTER_CUBIC)
        if channel != 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        s = img.reshape((width * height * channel,))
        data.append( s)
        label.append( -1.0)
        n0 += 1
    print 'load %d negative samples' % n0
    print 'total elapsed time: %.2f s' % (time.time() - t)
        
    return data, label

def training(d, l, C = 0.6, tol = 0.01, max_iter = 20, p = True):
    b, alphas = svm.smoP(d, l, C, tol, max_iter)
    if p == True:
        print calcrate(d, l, b, alphas)
    return b, alphas

def calcrate(d, l, b, alphas, n_pos = 503):
    ws = svm.calcWs(alphas, d, l)
    x = []
    n = len(d)
    datmat = np.mat(d)
    for i in range(n):
        s = (datmat[i] * np.mat(ws) + b) * l[i]
        if s > 0:
            s = 1.0
        else:
            s = -1.0
        x.append(s)
    x0 = np.array(x) + 1
    r1 = sum(x0) / 2.0 / n
    r2 = sum(x0[0:n_pos]) / 2.0 / n_pos
    r3 = sum(x0[n_pos:]) / 2.0 / (n - n_pos)
    return r1,r2,r3
    print 'total:%f\npositive:%f\nnegative:%f\n' % (r1, r2, r3)

def quicktraining(C0 = 0.6):
    n_pic_pos = 503
    n_pic_neg = 11453
    label = []
    data = []
    width = 36
    height = 12
    [data, label] = loadimgdata(width = width, height = height, channel = 1)
    print len(label)
    print 'strat training'
    t0 = time.time()
    b, alphas = training(data, label, C = C0)
    print 'time:%f min'%((time.time() - t0) / 60)
    return b, alphas, data, label

# begin
if __name__ == '__main__':
    C0 = [0.6]
    x =[]
    for C in C0:
        b, alphas, d, l = quicktraining(C)
        x.append([b, alphas, d, l])
