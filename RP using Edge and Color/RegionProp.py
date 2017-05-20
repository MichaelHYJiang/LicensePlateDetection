# !/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import os


def wait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contour(src, dst, color = (0,0,255), linewidth = 3):
    contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dst, contours, -1, color, linewidth)
    return contours

# function 'roundcontour' is used to dialate contours
def roundcontour(n0, src, sz, dtype = 'uint8'):
    for n in range(n0):
        dst = np.zeros(sz, dtype)
        contours = contour(src, dst, color = (255, 255, 255), linewidth = 3)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        ret, dst = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
        src = dst - src
    return src

def sobel(img, size = 7, thresh = 250, dx = 2):
    dst = cv2.Sobel(img, -1, dx, 0, ksize = size)
    dst1 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ret, dst1 = cv2.threshold(dst1, thresh, 255, cv2.THRESH_BINARY)
    return dst1

# get an envelope/a bounding box of contours
def drawenvlp(contours, imgnew):
    n = len(contours)
    envlp = []
    for k in range(0 , n):
        x = contours[k]
        left = tuple(x[:,0][x[:,:,0].argmin()])
        right = tuple(x[:,0][x[:,:,0].argmax()])
        up = tuple(x[:,0][x[:,:,1].argmin()])
        down =tuple(x[:,0][x[:,:,1].argmax()])
        l = left[0]
        u = up[1]
        r = right[0]
        d = down[1]
        envlp.append([l, u, r, d])
        cv2.rectangle(imgnew,(l, u),(r, d),(0,0,255),3)
    return imgnew, envlp

def mean_and_deviation(l):# to get mean value and standard deviation for list l
    m = len(l)
    mean = 0
    for i in range(m):
        mean += i * l[i, 0]
    mean /= l.sum()
    deviation = 0
    for i in range(m):
        deviation += ((i - mean) ** 2 * l[i, 0])
    deviation /= l.sum()
    deviation = deviation ** 0.5
    return mean, deviation

def colorSeg(img, method = 'sample', yita = 1.5):
    # First pair:[78,43,46]~[110,255,255]
    
    # Second pair:(calculated from first 100 pictures)
    #   mean value:         h--109  s--101  v--78
    #   standard deviation: h--45   s--42   v--114
    #[64,59,0]~[154,255,255]
    
    # Third pair:(calculated from try.jpg)
    #   mean value:         h--109  s--217  v--150
    #   standard deviation: h--7    s--41   v--57
    # method = 'regression'#('regression','mean','sample')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if method == 'sample':
        h = 109
        s = 217    
        v = 150
        sh = 7    
        ss = 41
        sv = 57
    elif method == 'mean':
        h = 99  #np.loadtxt('.\\New\\HSV\\mhgt.txt').mean()
        s = 129 #np.loadtxt('.\\New\\HSV\\msgt.txt').mean()
        v = 141 #np.loadtxt('.\\New\\HSV\\mvgt.txt').mean()
        sh = 27 #np.loadtxt('.\\New\\HSV\\dhgt.txt').mean()
        ss = 81 #np.loadtxt('.\\New\\HSV\\dsgt.txt').mean()
        sv = 40 #np.loadtxt('.\\New\\HSV\\dvgt.txt').mean()
    elif method == 'regression':
        cof = np.array([[  4.35361316e-02,   9.69973484e+01],
                     [  3.53137366e-01,   1.11931786e+01],
                     [ -2.29437014e-01,   1.39930583e+02],
                     [  1.98145885e-01,   7.35032801e+01],
                     [  5.78410131e-01,   6.86876872e+01],
                     [  1.23054287e-01,   3.32777673e+01]])
        
        
        hh = cv2.calcHist([hsv], [0], None, [180], [0.0, 180.0]) + 1
        hs = cv2.calcHist([hsv], [1], None, [256], [0.0, 256.0]) + 1
        hv = cv2.calcHist([hsv], [2], None, [256], [0.0, 256.0]) + 1
        
        [mean, deviation] = mean_and_deviation(hh)
        p = np.poly1d(cof[0])
        h = p(mean)
        p = np.poly1d(cof[1])
        sh = p(deviation)
        [mean, deviation] = mean_and_deviation(hs)
        p = np.poly1d(cof[2])
        s = p(mean)
        p = np.poly1d(cof[3])
        ss = p(deviation)
        [mean, deviation] = mean_and_deviation(hv)
        p = np.poly1d(cof[4])
        v = p(mean)
        p = np.poly1d(cof[5])
        sv = p(deviation)
    
    lower_blue=np.array([h - sh * yita, s - ss * yita, v - sv * yita])
    upper_blue=np.array([h + sh * yita, s + ss * yita, v + sv * yita])
    
    img1 = cv2.inRange(hsv, lower_blue, upper_blue)
    return img1

def regionprop(imgsrc, sz, sobelsize = 7, thresh = 250, dx = 2, hsv = 'sample', yita = 1.5):
    imgdst = imgsrc.copy()

    # Sobel
    img1 = sobel(imgdst, sobelsize, thresh, dx)
    
    # Color Segmentation
    img2 = colorSeg(imgdst, hsv, yita)
    
    img1 = roundcontour(3, img1, sz)
    img2 = roundcontour(3, img2, sz)
    img3 = img1 * img2

    img3 = roundcontour(3, img3, sz)

    imgnew = imgsrc.copy()
    contours = contour(img3, imgnew)
    imgnew = imgsrc.copy()
    imgdst, envlp = drawenvlp(contours, imgnew)
    return imgdst, envlp

def suffix(num):
    if num <= 5:
        return '.bmp'
    else:
        return '.jpg'

def demo():
    g = os.listdir(os.path.join('.', '0_Good'))
    g = [x.split('.') for x in g if x.split('.')[-1].lower() in ('jpg', 'bmp')]

    k0 = 0
    
    write = False

    if write:
        f = open(os.path.join('.', 'rp.txt'), 'w')
    
    while k0 < len(g):
        x = str(k0 + 1) + suffix(k0 + 1)
        filepath = os.path.join('.', '0_Good', x)
        img = cv2.imread(filepath)

        sz = img.shape
        
        ratio = max(sz) / 800.0        
        
        img400 = img.copy()
        img400 = cv2.resize(img400,(int(sz[1] / ratio / 2.0), int(sz[0] / ratio / 2.0)),
                         interpolation = cv2.INTER_CUBIC)
        sz400 = img400.shape
        imgnew, envlp400 = regionprop(img400, sz400)
        cv2.imshow(x+'-400', imgnew)

        img800 = img.copy()
        img800 = cv2.resize(img800,(int(sz[1] / ratio), int(sz[0] / ratio)),
                         interpolation = cv2.INTER_CUBIC)
        sz800 = img800.shape
        rate = 2
        envlp800 = []
        envlp400 = [list(a) for a in list(np.array(envlp400) * rate)]
        for e in envlp400:#envelpe = [[l, u, r, d]]            
            imgnew = img800[e[1] : ( e[3] + 1),\
                            e[0] : ( e[2] + 1),\
                            :]
            imgnew, envlp = regionprop(imgnew, imgnew.shape)
            envlp = [[env[0] + e[0], env[1] + e[1], \
                      env[2] + e[0], env[3] + e[1]] \
                     for env in envlp]
            for env in envlp:
                envlp800.append(env)
            if imgnew.shape[0] < 5 or imgnew.shape[1] < 5:
                linewidth = min(imgnew.shape[0],imgnew.shape[1]) - 1
            else:
                linewidth = 5
            cv2.rectangle(imgnew, (0,0), (imgnew.shape[1]-1,imgnew.shape[0]-1), (0, 255, 0), linewidth)
            img800[e[1] : ( e[3] + 1),\
                   e[0] : ( e[2] + 1),\
                   :] = imgnew
        cv2.imshow(x+'-800', img800)        
        
        img0 = img.copy()
        sz0 = img0.shape
        rate = ratio
        envlp0 = []
        envlp400 = [list(a) for a in list(np.array(envlp400) * rate)]
        envlp800 = [list(a) for a in list(np.array(envlp800) * rate)]
        for e in envlp800:#envelpe = [[l, u, r, d]]            
            imgnew = img0[int(e[1]) : int( e[3] + 1),\
                          int(e[0]) : int( e[2] + 1),\
                          :]
            imgnew, envlp = regionprop(imgnew, imgnew.shape)
            envlp = [[env[0] + int(e[0]), env[1] + int(e[1]), \
                      env[2] + int(e[0]), env[3] + int(e[1])] \
                     for env in envlp]
            for env in envlp:
                envlp0.append(env)
            if imgnew.shape[0] < 5 or imgnew.shape[1] < 5:
                linewidth = min(imgnew.shape[0],imgnew.shape[1]) - 1
            else:
                linewidth = 5
            cv2.rectangle(imgnew, (0,0), (imgnew.shape[1]-1,imgnew.shape[0]-1), (0, 255, 0), linewidth)
            img0[int(e[1]) : int( e[3] + 1),\
                 int(e[0]) : int( e[2] + 1),\
                 :] = imgnew
        for e in envlp400:#envelpe = [[l, u, r, d]]            
            imgnew = img0[int(e[1]) : int( e[3] + 1),\
                          int(e[0]) : int( e[2] + 1),\
                          :]
            
            if imgnew.shape[0] < 5 or imgnew.shape[1] < 5:
                linewidth = min(imgnew.shape[0],imgnew.shape[1]) - 1
            else:
                linewidth = 5
            cv2.rectangle(imgnew, (0,0), (imgnew.shape[1]-1,imgnew.shape[0]-1), (255, 0, 0), linewidth)
            img0[int(e[1]) : int( e[3] + 1),\
                 int(e[0]) : int( e[2] + 1),\
                 :] = imgnew
        imgnew = cv2.resize(img0,(int(sz[1] / ratio), int(sz[0] / ratio)),
                         interpolation = cv2.INTER_CUBIC)
        cv2.imshow(x+'-original', imgnew)

        envlp = envlp0 + envlp400 + envlp800
        if write:
            writefile = x + '\t' + str(len(envlp)) + '\t'
            for e in envlp:
                for i in range(4):
                    writefile += str(int(e[i])) + '\t'
            writefile += '\n'
            f.write(writefile)
        
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 32:
            break
        elif key == 2490368:
            k0 = k0 - 2
        k0 = k0 + 1

    cv2.destroyAllWindows()
    if write:
        f.close()
#===============================================================================
# begin:
if __name__ == "__main__":
    demo()
