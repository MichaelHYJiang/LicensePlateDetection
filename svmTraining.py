# -*- coding: cp936 -*-
import svm
import cv2
import numpy as np
import time

def loadimgdata():
    n_pic_pos = 503
    n_pic_neg = 11453
    label = []
    data = []
    for i in range(n_pic_pos):
        img = cv2.imread('H:\\车牌图像\\Feature Training\\POS\\'+str(i)+'.jpg')
        img = cv2.resize(img, (20,5),interpolation = cv2.INTER_CUBIC)
        s = []
        for i0 in range(5):
            for i1 in range(20):
                for i2 in range(3):
                    s.append(img[i0,i1,i2])
        data.append(s)
        label.append(1.0)
    for i in range(n_pic_neg):
        img = cv2.imread('H:\\车牌图像\\Feature Training\\negative\\'+str(i)+'.jpg')
        img = cv2.resize(img, (20,5),interpolation = cv2.INTER_CUBIC)
        s = []
        for i0 in range(5):
            for i1 in range(20):
                for i2 in range(3):
                    s.append(img[i0,i1,i2])
        data.append(s)
        label.append(-1.0)
    return data, label

def loadimg():
    n_pic_pos = 503
    n_pic_neg = 11453
    label = []
    data = []
    wid = 30
    hei = 10
    for i in range(n_pic_pos):
        img = cv2.imread('H:\\车牌图像\\Feature Training\\POS\\'+str(i)+'.jpg')
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (wid, hei),interpolation = cv2.INTER_CUBIC)
        img = img.reshape(1, wid * hei)
        data.append(img)
        label.append(1.0)
    for i in range(n_pic_neg):
        if np.random.rand() < 0.95:
            continue
        img = cv2.imread('H:\\车牌图像\\Feature Training\\negative\\'+str(i)+'.jpg')
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (wid, hei),interpolation = cv2.INTER_CUBIC)
        img = img.reshape(1, wid * hei)
        data.append(img)
        label.append(-1.0)
    return data, label
def training(d, l, C = 0.6, tol = 0.01, max_iter = 50, p = True):
    # d, l = loadimgdata()
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
    #print 'total:%f\npositive:%f\nnegative:%f\n' % (r1, r2, r3)

def quicktraining(C0 = 0.6):
    n_pic_pos = 503
    n_pic_neg = 11453
    label = []
    data = []
    width = 36
    height = 12
    for i in range(n_pic_pos):
        img = cv2.imread('H:\\车牌图像\\Feature Training\\POS\\'+str(i)+'.jpg',0)
        img = cv2.resize(img, (width, height),interpolation = cv2.INTER_CUBIC)
        s = []
        for i0 in range(height):
            for i1 in range(width):
                s.append(img[i0,i1])
        data.append(s)
        label.append(1.0)
    for i in range(n_pic_neg):
        if np.random.rand() < 0.97:
            continue
        img = cv2.imread('H:\\车牌图像\\Feature Training\\negative\\'+str(i)+'.jpg',0)
        img = cv2.resize(img, (width, height),interpolation = cv2.INTER_CUBIC)
        s = []
        for i0 in range(height):
            for i1 in range(width):
                s.append(img[i0,i1])
        data.append(s)
        label.append(-1.0)
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
