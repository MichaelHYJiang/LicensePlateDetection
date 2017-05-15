# -*- coding: cp936 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
#import itchat

def wait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contour(src, dst, color = (0,0,255), linewidth = 3):
    contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dst, contours, -1, color, linewidth)
    return contours
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
        #if (r - l) * (d - u) > 10000:
        cv2.rectangle(imgnew,(l, u),(r, d),(0,0,255),3)
    return imgnew, envlp
def NaiveBayes(hsv, img1):
    # Histogram of samples used for naive Bayes method
    from_hist_files = False
    if from_hist_files == True:
        hist_h0 = np.loadtxt('hist_h0_100.txt')
        hist_s0 = np.loadtxt('hist_s0_100.txt')
        hist_v0 = np.loadtxt('hist_v0_100.txt')
        hist_h = np.loadtxt('hist_h_100.txt')
        hist_s = np.loadtxt('hist_s_100.txt')
        hist_v = np.loadtxt('hist_v_100.txt')
        hist_h1 = hist_h - hist_h0
        hist_s1 = hist_s - hist_s0
        hist_v1 = hist_v - hist_v0
        py0 = (sum(hist_h0) / sum(hist_h))**0.5
        py1 = 1 - py0
    #else:
        image = cv2.imread('.\\0_Good\\颜色采样\\try.jpg')
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.ones([image.shape[0],image.shape[1]], 'uint8')
        mask[-1, 5:] = 0
        hh = cv2.calcHist([hsv_image], [0], mask, [180], [0.0, 180.0]) + 1
        hs = cv2.calcHist([hsv_image], [1], mask, [256], [0.0, 256.0]) + 1
        hv = cv2.calcHist([hsv_image], [2], mask, [256], [0.0, 256.0]) + 1

    # Naive Bayes
    Bayes = False
    if Bayes == True:
        for i in range(sz[0]):
            for j in range(sz[1]):
                if img1[i,j] != 0:
                    h = hsv[i,j,0]
                    s = hsv[i,j,1]
                    v = hsv[i,j,2]
                    p0 = hh[h] / sum(hh) #(hist_h0[h] / sum(hist_h0)) #* py0
                    p0 *= hs[s] / sum(hs) #(hist_s0[s] / sum(hist_s0))# * py0
                    p0 *= hv[v] / sum(hv) #(hist_v0[v] / sum(hist_v0))# * py0
                    p1 = (hist_h1[h] / sum(hist_h1)) #* py1
                    p1 *= (hist_s1[s] / sum(hist_s1))# * py1
                    p1 *= (hist_v1[v] / sum(hist_v1))# * py1
                    if p0 < p1:
                        img1[i,j] = 0
        cv2.imshow('img1-after', img1)

def mean_and_deviation(l):
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
def prepare():
       
    mY = []; dY = []; mYgt = []; dYgt = [];
    mCr = []; dCr = []; mCrgt = []; dCrgt = [];
    mCb = []; dCb = []; mCbgt = []; dCbgt = [];
    with open('.\\Record.txt','r') as f:
        g = f.readlines()
        g = g[1:-1]

    k0 = 0          #130    
    at = 0    
    while k0 < len(g):
        x = g[k0]
        y = x.split()
        if len(y) < 5:
            g.remove(x)
            continue
        k0 += 1
        
        t = time.time()
        
        img = cv2.imread('.\\0_Good\\'+y[0])
        x1 = int(y[1])
        y1 = int(y[2])
        x2 = x1 + int(y[3])
        y2 = y1 + int(y[4])
        
        sz = img.shape
        if max(sz) > 1000:
            x1 = x1 * 2
            x2 = x2 * 2
            y1 = y1 * 2
            y2 = y2 * 2
        
        YCrCb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        maxY = 255#max(YCrCb_image[:,:,0])
        maxCr = 255#max(YCrCb_image[:,:,1])
        maxCb = 255#max(YCrCb_image[:,:,2])
        mask = None
        hY = cv2.calcHist([YCrCb_image], [0], mask, [maxY], [0.0, maxY]) + 1
        hCr = cv2.calcHist([YCrCb_image], [1], mask, [maxCr], [0.0, maxCr]) + 1
        hCb = cv2.calcHist([YCrCb_image], [2], mask, [maxCb], [0.0, maxCb]) + 1
        
        [mean, deviation] = mean_and_deviation(hY)
        mY.append(str(mean) + '\n')
        dY.append(str(deviation) + '\n')
        [mean, deviation] = mean_and_deviation(hCr)
        mCr.append(str(mean) + '\n')
        dCr.append(str(deviation) + '\n')
        [mean, deviation] = mean_and_deviation(hCb)
        mCb.append(str(mean) + '\n')
        dCb.append(str(deviation) + '\n')

        img_gt = img[y1:y2, x1:x2, :]
        YCrCb_image = cv2.cvtColor(img_gt, cv2.COLOR_BGR2YCR_CB)
        maxY = 255#max(YCrCb_image[:,:,0])
        maxCr = 255#max(YCrCb_image[:,:,1])
        maxCb = 255#max(YCrCb_image[:,:,2])
        mask = None
        hY = cv2.calcHist([YCrCb_image], [0], mask, [maxY], [0.0, maxY]) + 1
        hCr = cv2.calcHist([YCrCb_image], [1], mask, [maxCr], [0.0, maxCr]) + 1
        hCb = cv2.calcHist([YCrCb_image], [2], mask, [maxCb], [0.0, maxCb]) + 1
        
        [mean, deviation] = mean_and_deviation(hY)
        mYgt.append(str(mean) + '\n')
        dYgt.append(str(deviation) + '\n')
        [mean, deviation] = mean_and_deviation(hCr)
        mCrgt.append(str(mean) + '\n')
        dCrgt.append(str(deviation) + '\n')
        [mean, deviation] = mean_and_deviation(hCb)
        mCbgt.append(str(mean) + '\n')
        dCbgt.append(str(deviation) + '\n')
        
        t = time.time() - t
        at += t
        print str(k0) + '\t' + str(t) + '\ts\trest:%fmin'%((504 - k0) * (at / k0)/60)

    var = ('mY','dY','mCr','dCr','mCb','dCb','mYgt','dYgt','mCrgt','dCrgt','mCbgt','dCbgt')
    for e in var:
        with open('.\\New\\YCrCb\\'+e+'.txt','w') as f:
            #eval(e) = [str(x) + '\n' for x in locals()[e]]
            #print locals()[e]
            f.writelines(locals()[e])
            
def regression():
    var = ('mh','dh','ms','ds','mv','dv','mhgt','dhgt','msgt','dsgt','mvgt','dvgt')
    var1 = ('mY','dY','mCr','dCr','mCb','dCb')#('mh','dh','ms','ds','mv','dv')
    
    p = []
    for e in var1:
        with open('.\\New\\YCrCb\\' + e + '.txt', 'r') as f:
            g = f.readlines()
        x = [float(a.strip()) for a in g]
        with open('.\\New\\YCrCb\\' + e + 'gt.txt', 'r') as f:
            g = f.readlines()
        y = [float(a.strip()) for a in g]
        cof = np.polyfit(x,y,1)
        p.append(np.poly1d(cof))
        #plt.plot(x,y,'x', x, p(x), lw = 2)
        #plt.show()
    return p

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
    
    
    #yita = 1.0
    lower_blue=np.array([h - sh * yita, s - ss * yita, v - sv * yita])
    upper_blue=np.array([h + sh * yita, s + ss * yita, v + sv * yita])
    
    img1 = cv2.inRange(hsv, lower_blue, upper_blue)
    return img1

def colorSeg_withregression(img):
    
    YCrCb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    maxY = 255#max(YCrCb_image[:,:,0])
    maxCr = 255#max(YCrCb_image[:,:,1])
    maxCb = 255#max(YCrCb_image[:,:,2])
    mask = None
    hY = cv2.calcHist([YCrCb_image], [0], mask, [maxY], [0.0, maxY]) + 1
    hCr = cv2.calcHist([YCrCb_image], [1], mask, [maxCr], [0.0, maxCr]) + 1
    hCb = cv2.calcHist([YCrCb_image], [2], mask, [maxCb], [0.0, maxCb]) + 1
    [mean, deviation] = mean_and_deviation(hY)
    mY = mean
    dY = deviation
    [mean, deviation] = mean_and_deviation(hCr)
    mCr = mean
    dCr = deviation
    [mean, deviation] = mean_and_deviation(hCb)
    mCb = mean
    dCb = deviation

    p = regression()
    mYgt = p[0](mY)
    dYgt = p[1](dY)
    mCrgt = p[2](mCr)
    dCrgt = p[3](dCr)
    mCbgt = p[4](mCb)
    dCbgt = p[5](dCb)

    yita = 1.5
    lower_blue=np.array([mYgt - dYgt * yita, mCrgt - dCrgt * yita, \
                         mCbgt - dCbgt * yita])
    upper_blue=np.array([mYgt + dYgt * yita, mCrgt + dCrgt * yita, \
                         mCbgt + dCbgt * yita])
    img1 = cv2.inRange(YCrCb_image, lower_blue, upper_blue)
    return img1

def draw_YCrCb():
    x = []; y = []
    at = 0
    for i in range(300):
        t = time.time()
        img = cv2.imread('.\\Feature Training\\POS\\' + str(i) + '.jpg')
        sz = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        for i1 in range(sz[0]):
            for i2 in range(sz[1]):
                [Y, Cr, Cb] = img[i1, i2, :]
                x.append(Y / float(Cr))
                y.append(Y / float(Cb))
        t = time.time() - t
        at += t
        print str(i) + '\t' + str(t) +'\ts\trest:%f\tmin' % ((300 - i) * (at / (i + 1)))

    
    np.savetxt('.\\New\\YCrCb-x.txt',x)
    np.savetxt('.\\New\\YCrCb-y.txt',y)        
    
    plt.plot(x, y, 'x')
    plt.xlabel('Y / Cr')
    plt.ylabel('Y / Cb')
    plt.show()

def regionprop(imgsrc, sz, sobelsize = 7, thresh = 250, dx = 2, hsv = 'sample', yita = 1.5):
    imgdst = imgsrc.copy()

    # Sobel
    img1 = sobel(imgdst, sobelsize, thresh, dx)
    
    # Color Segmentation
    img2 = colorSeg(imgdst, hsv, yita)
    #cv2.imshow('color segmentation', img2)
    
    img1 = roundcontour(3, img1, sz)
    img2 = roundcontour(3, img2, sz)
    img3 = img1 * img2

    img3 = roundcontour(3, img3, sz)

    imgnew = imgsrc.copy()
    contours = contour(img3, imgnew)
    imgnew = imgsrc.copy()
    imgdst, envlp = drawenvlp(contours, imgnew)
    return imgdst, envlp

def loadimg():
    n_pic_pos = 503#349
    n_pic_neg = 11453
    label = []
    data = []
    wid = 25
    hei = 15
    for i in range(n_pic_pos):
        img = cv2.imread('H:\\车牌图像\\Feature Training\\POS\\'+str(i)+'.jpg')
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (wid, hei),interpolation = cv2.INTER_CUBIC)
        img = img.reshape(1, wid * hei)
        data.append(img)
        label.append(1.0)
    p0 = 1 - 349 * 2.5 / 11453.0
    for i in range(n_pic_neg):
        if np.random.rand() < p0:
            continue
        img = cv2.imread('H:\\车牌图像\\Feature Training\\negative\\'+str(i)+'.jpg')
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (wid, hei),interpolation = cv2.INTER_CUBIC)
        img = img.reshape(1, wid * hei)
        data.append(img)
        label.append(-1.0)
    return data, label

def main(use_svm = False):
    with open('.\\Record.txt','r') as f:
        g = f.readlines()
        g = g[1:-1]

    k0 = 0          #130
    
    
    if use_svm:
        data,label = loadimg()
        data = np.array(data, 'float32')
        data = np.mat(data, 'float32')
        label = np.mat(label,'float32')
        svm = cv2.SVM()
        p = dict( kernel_type = cv2.SVM_RBF,
              svm_type = cv2.SVM_C_SVC,
              C = 2)
        print('svm training:')
        print(label.shape[1])
        svm.train(data, label, params = p)
        pre = svm.predict_all(data)
        pre = np.mat(pre)
        pre = pre.T
        diff = pre - label
        diff = np.array(diff)
        print(1 - sum(sum(diff * diff)) / float(label.shape[1]))
    
    while k0 < len(g):
        x = g[k0]
        y = x.split()
        if len(y) < 5:
            g.remove(x)
            continue
        img = cv2.imread('.\\0_Good\\'+y[0])
        
        sz = img.shape
        if max(sz) > 1000:
            ratio = max(sz) / 800.0
        else:
            ratio = 1
        #img = cv2.resize(img,(int(sz[1] / ratio), int(sz[0] / ratio)),
        #                 interpolation = cv2.INTER_CUBIC)

        sz = img.shape
        
        img400 = img.copy()
        img400 = cv2.resize(img400,(int(sz[1] / ratio / 2.0), int(sz[0] / ratio / 2.0)),
                         interpolation = cv2.INTER_CUBIC)
        sz400 = img400.shape
        imgnew, envlp400 = regionprop(img400, sz400)
        cv2.imshow(y[0]+'-400', imgnew)

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
            for i in range(imgnew.shape[0]):
                imgnew[i, 0, :] = (255, 0, 0)
                imgnew[i, -1, :] = (255, 0, 0)
            for i in range(imgnew.shape[1]):
                imgnew[0, i, :] = (255, 0, 0)
                imgnew[-1, i, :] = (255, 0, 0)
            img800[e[1] : ( e[3] + 1),\
                   e[0] : ( e[2] + 1),\
                   :] = imgnew        
        #imgnew, envlp = regionprop(img0, sz0)
        cv2.imshow(y[0]+'-800', img800)
        
        
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
            #print imgnew.shape
            #for i in range(imgnew.shape[0]):
            #    imgnew[i, 0, :] = (0, 255, 0)
            #    imgnew[i, 1, :] = (0, 255, 0)
            #    imgnew[i, 2, :] = (0, 255, 0)
            #    imgnew[i, 3, :] = (0, 255, 0)
            #    imgnew[i, 4, :] = (0, 255, 0)
            #    imgnew[i, -1, :] = (0, 255, 0)
            #    imgnew[i, -2, :] = (0, 255, 0)
            #    imgnew[i, -3, :] = (0, 255, 0)
            #    imgnew[i, -4, :] = (0, 255, 0)
            #    imgnew[i, -5, :] = (0, 255, 0)
            #for i in range(imgnew.shape[1]):
            #    imgnew[0, i, :] = (0, 255, 0)
            #    imgnew[1, i, :] = (0, 255, 0)
            #    imgnew[2, i, :] = (0, 255, 0)
            #    imgnew[3, i, :] = (0, 255, 0)
            #    imgnew[4, i, :] = (0, 255, 0)
            #    imgnew[-1, i, :] = (0, 255, 0)
            #    imgnew[-2, i, :] = (0, 255, 0)
            #    imgnew[-3, i, :] = (0, 255, 0)
            #    imgnew[-4, i, :] = (0, 255, 0)
            #    imgnew[-5, i, :] = (0, 255, 0)
            img0[int(e[1]) : int( e[3] + 1),\
                 int(e[0]) : int( e[2] + 1),\
                 :] = imgnew 
        #imgnew, envlp = regionprop(img0, sz0)
        imgnew = cv2.resize(img0,(int(sz[1] / ratio), int(sz[0] / ratio)),
                         interpolation = cv2.INTER_CUBIC)
        cv2.imshow(y[0]+'-original', imgnew)       

        #img0 = img.copy()
        #img0 = cv2.resize(img0,(int(sz[1] * 3), int(sz[0]* 3)),
        #                 interpolation = cv2.INTER_CUBIC)
        #sz0 = img0.shape
        #imgnew = regionprop(img0, sz0)
        #imgnew = cv2.resize(imgnew,(int(sz[1] / ratio), int(sz[0] / ratio)),
        #                 interpolation = cv2.INTER_CUBIC)
        #cv2.imshow(y[0]+'-Double', imgnew)
        #ws = np.mat(np.loadtxt('wx_0327.txt')).T
        #b = np.mat(np.loadtxt('b_0327.txt'))
        if use_svm:
            img0 = img.copy()
            rect = []
            score = []
            area0 = 10000
            height = 15
            width = 25
            for e in envlp400:
                l = int(e[0])
                u = int(e[1])
                r = int(e[2])
                d = int(e[3])
                area = (d - u + 1) * (r - l + 1)
                if area < area0:
                    continue
                imgnew = img0[u:d, l:r, :]
                imgnew = cv2.cvtColor(imgnew, cv2.COLOR_BGR2GRAY)
                imgnew = cv2.resize(imgnew, (width, height))
                data = imgnew.reshape(1, width * height)
                data = np.mat(data, 'float32')
                s = svm.predict(data)
                if s > 0:
                    rect.append([l,u,r,d])
                    score.append(s)
            for e in envlp800:
                l = int(e[0])
                u = int(e[1])
                r = int(e[2])
                d = int(e[3])
                area = (d - u + 1) * (r - l + 1)
                if area < area0:
                    continue
                imgnew = img0[u:d, l:r, :]
                imgnew = cv2.cvtColor(imgnew, cv2.COLOR_BGR2GRAY)
                imgnew = cv2.resize(imgnew, (width, height))
                data = imgnew.reshape(1, width * height)
                data = np.mat(data, 'float32')
                s = svm.predict(data)
                if s > 0:
                    rect.append([l,u,r,d])
                    score.append(s)
            for e in envlp0:
                l = int(e[0])
                u = int(e[1])
                r = int(e[2])
                d = int(e[3])
                area = (d - u + 1) * (r - l + 1)
                if area < area0:
                    continue
                imgnew = img0[u:d, l:r, :]
                imgnew = cv2.cvtColor(imgnew, cv2.COLOR_BGR2GRAY)
                imgnew = cv2.resize(imgnew, (width, height))
                data = imgnew.reshape(1, width * height)
                data = np.mat(data, 'float32')
                s = svm.predict(data)
                if s > 0:
                    rect.append([l,u,r,d])
                    score.append(s)
            #score = np.array(score)
            #if score.shape != (0,):
            #    index = score.argmax()
            print(len(score))
            for index in range(len(score)):
                [l, u, r, d] = rect[index]
                cv2.rectangle(img0, (l, u), (r, d), (255, 255, 0), 5)

                    
            img0 = cv2.resize(img0,(int(sz[1] / ratio), int(sz[0] / ratio)),
                             interpolation = cv2.INTER_CUBIC)
            cv2.imshow('new', img0)
                        
        
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 32:
            break
        elif key == 2490368:
            k0 = k0 - 2
        k0 = k0 + 1

    cv2.destroyAllWindows()
    if use_svm:
        return data, label, svm
    else:
        return None, None, None


def calcIOU(box1, box2, diff = True):
    x11 = box1[0]
    y11 = box1[1]
    x21 = box2[0]
    y21 = box2[1]
    if diff:
        x12 = x11 + box1[2]
        y12 = y11 + box1[3]
        x22 = x21 + box2[2]
        y22 = y21 + box2[3]
    else:
        x12 = box1[2]
        y12 = box1[3]
        x22 = box2[2]
        y22 = box2[3]
        
    x1 = max(x11, x21)
    y1 = max(y11, y21)
    x2 = min(x12, x22)
    y2 = min(y12, y22)
    I = max(x2 - x1, 0) * max(y2 - y1, 0);
    U = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - I;
    IOU = float(I) / U;
    return IOU

def test(sobelsize = 7, thresh = 250, dx = 2, hsv = 'sample', yita = 1.5, IOU = 0.7):
    with open('.\\Record.txt','r') as f:
        g = f.readlines()
        g = g[1:-1]

    k0 = 0

    succ_before = 0
    succ = 0
    at = 0
    nothing = 0
    t0 = time.time()
    while k0 < len(g):
        t = time.time()
        x = g[k0]
        y = x.split()
        if len(y) < 5:
            g.remove(x)
            continue
        img = cv2.imread('.\\0_Good\\'+y[0])
        x1 = int(y[1])
        y1 = int(y[2])
        x2 = x1 + int(y[3])
        y2 = y1 + int(y[4])
        
        sz = img.shape
        if max(sz) > 1000:
            x1 = x1 * 2
            x2 = x2 * 2
            y1 = y1 * 2
            y2 = y2 * 2
            ratio = max(sz) / 800.0
        else:
            ratio = 1
            
        area = (y2 - y1 + 1) * (x2 - x1 + 1)
        sz = img.shape
        
        img400 = img.copy()
        img400 = cv2.resize(img400,(int(sz[1] / ratio / 2.0), int(sz[0] / ratio / 2.0)),
                         interpolation = cv2.INTER_CUBIC)
        sz400 = img400.shape
        imgnew, envlp400 = regionprop(img400, sz400, sobelsize, thresh, dx, hsv, yita)

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
            imgnew, envlp = regionprop(imgnew, imgnew.shape, sobelsize, thresh, dx, hsv, yita)
            envlp = [[env[0] + e[0], env[1] + e[1], \
                      env[2] + e[0], env[3] + e[1]] \
                     for env in envlp]
            for env in envlp:
                envlp800.append(env)
            
        
        
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
            imgnew, envlp = regionprop(imgnew, imgnew.shape, sobelsize, thresh, dx, hsv, yita)
            envlp = [[env[0] + int(e[0]), env[1] + int(e[1]), \
                      env[2] + int(e[0]), env[3] + int(e[1])] \
                     for env in envlp]
            for env in envlp:
                envlp0.append(env)
            
        
        
        #ws = np.mat(np.loadtxt('wx_0327.txt')).T
        #b = np.mat(np.loadtxt('b_0327.txt'))
        
        img0 = img.copy()
        rect = []
        score = []
        area0 = 0000
        flag = False
        overlap = IOU #!!!!!!!!!!!
        for e in envlp400:
            l = int(e[0])
            u = int(e[1])
            r = int(e[2])
            d = int(e[3])
            area = (d - u + 1) * (r - l + 1)
            if area < area0:
                continue
            if flag == False:
                o = calcIOU([l,u,r,d], [x1,y1,x2,y2], False)
                if o > overlap:
                    succ_before += 1
                    flag = True
            
            imgnew = img0[u:d, l:r, :]                   
            imgnew = cv2.resize(imgnew, (20,5))
            #data = []
            #for i0 in range(5):
            #    for i1 in range(20):
            #        for i2 in range(3):
            #            data.append(imgnew[i0, i1, i2])
            #data = np.mat(data)
            #s = data * ws + b
            #if s > 0:
            #    rect.append([l,u,r,d])
            #    score.append(s)
                
        for e in envlp800:
            l = int(e[0])
            u = int(e[1])
            r = int(e[2])
            d = int(e[3])
            area = (d - u + 1) * (r - l + 1)
            if area < area0:
                continue
            if flag == False:
                o = calcIOU([l,u,r,d], [x1,y1,x2,y2], False)
                if o > overlap:
                    succ_before += 1
                    flag = True
                       
            #imgnew = img0[u:d, l:r, :]
            #imgnew = cv2.resize(imgnew, (20,5))
            #data = []
            #for i0 in range(5):
            #    for i1 in range(20):
            #        for i2 in range(3):
            #            data.append(imgnew[i0, i1, i2])
            #data = np.mat(data)
            #s = data * ws + b
            #if s > 0:
            #    rect.append([l,u,r,d])
            #    score.append(s)
                
        for e in envlp0:
            l = int(e[0])
            u = int(e[1])
            r = int(e[2])
            d = int(e[3])
            area = (d - u + 1) * (r - l + 1)
            if area < area0:
                continue
            if flag == False:
                o = calcIOU([l,u,r,d], [x1,y1,x2,y2], False)
                if o > overlap:
                    succ_before += 1
                    flag = True
                       
            #imgnew = img0[u:d, l:r, :]
            #imgnew = cv2.resize(imgnew, (20,5))
            #data = []
            #for i0 in range(5):
            #    for i1 in range(20):
            #        for i2 in range(3):
            #            data.append(imgnew[i0, i1, i2])
            #data = np.mat(data)
            #s = data * ws + b
            #if s > 0:
            #    rect.append([l,u,r,d])
            #    score.append(s)
        t1 = time.time() - t
        if flag == False:
            nothing += 1
        #score = np.array(score)
        #overlap = 0.6
        #if score.shape != (0,):
        #    index = score.argmax()
        #    [l, u, r, d] = rect[index]
        #    l = int(e[0])
        #    u = int(e[1])
        #    r = int(e[2])
        #    d = int(e[3])
        #    area0 = (r - l + 1) * (d - u + 1)
        #    left = max(l, x1)
        #    up = max(u, y1)
        #    right = min(r, x2)
        #    down = min(d, y2)
        #    w = right - left + 1
        #    h = down - up + 1
        #    if w > 0 and h > 0:
        #        o = w * h / float(max(area0, area))
        #        if o > overlap:
        #           succ += 1
        #t2 = time.time() - t
        at += t1
        k0 = k0 + 1

        print '%d:\t%.2f\trest:%.2fmin\t%d\t%f%%(%.2f%%)' \
              % (k0, t1,((503 - k0) * at / k0) / 60, \
                 succ_before, succ_before / float(k0) * 100, \
                 (503 - nothing) / 503.0 * 100)
        if k0 % 50 == 0:
            print k0 / float(len(g))
            print 'succ_before:%f'%(succ_before / float(k0))
            print 'succ:%f'%(succ / float(k0))
    t0 = (time.time() - t0)
    print 'succ_before_last:%f'%(succ_before / float(k0))
    print 'succ_last:%f'%(succ / float(k0))
    print 'total_time:', t0
    return succ_before / float(k0), t0
#===============================================================================
# begin:

if __name__ == "__main__":
    #d,l,svm = main()
    #itchat.auto_login()
    for sobsz in (3,5,7):
        for th in (100, 150, 200, 250):
            for dx in (1, 2):
                for hsv in ('regression','mean','sample'):
                    for yita in (0.5, 1.0, 1.5, 2.0):
                        if sobsz == 3 and th == 100 and dx == 1 and hsv == 'regression' and yita == 0.5 or \
                           sobsz == 3 and th == 100 and dx == 1 and hsv == 'regression' and yita == 1.0 or \
                           sobsz == 3 and th == 100 and dx == 1 and hsv == 'regression' and yita == 1.5 or \
                           sobsz == 3 and th == 100 and dx == 1 and hsv == 'regression' and yita == 2.0 or \
                           sobsz == 3 and th == 100 and dx == 1 and hsv == 'mean' and yita == 0.5 or \
                           sobsz == 3 and th == 100 and dx == 1 and hsv == 'mean' and yita == 1.0 or \
                           sobsz == 3 and th == 100 and dx == 1 and hsv == 'mean' and yita == 1.5 or \
                           sobsz == 3 and th == 100 and dx == 1 and hsv == 'mean' and yita == 2.0 or \
                           sobsz == 3 and th == 100 and dx == 1 and hsv == 'sample' and yita == 0.5 or \
                           sobsz == 3 and th == 100 and dx == 1 and hsv == 'sample' and yita == 1.0 or \
                           sobsz == 3 and th == 100 and dx == 1 and hsv == 'sample' and yita == 1.5 or \
                           sobsz == 3 and th == 100 and dx == 1 and hsv == 'sample' and yita == 2.0 or \
                           sobsz == 3 and th == 100 and dx == 2 and hsv == 'regression' and yita == 0.5 or \
                           sobsz == 3 and th == 100 and dx == 2 and hsv == 'regression' and yita == 1.0 or \
                           sobsz == 3 and th == 100 and dx == 2 and hsv == 'regression' and yita == 1.5 or \
                           sobsz == 3 and th == 100 and dx == 2 and hsv == 'regression' and yita == 2.0 or \
                           sobsz == 3 and th == 100 and dx == 2 and hsv == 'mean' and yita == 0.5 or \
                           sobsz == 3 and th == 100 and dx == 2 and hsv == 'mean' and yita == 1.0 or \
                           sobsz == 3 and th == 100 and dx == 2 and hsv == 'mean' and yita == 1.5 or \
                           sobsz == 3 and th == 100 and dx == 2 and hsv == 'mean' and yita == 2.0:
                            continue
                        print 'sobsz, th, dx, hsv, yita'
                        print sobsz, th, dx, hsv, yita
                        [rate, totaltime] = test(sobsz, th, dx, hsv, yita, 0.7)
                        writefile = 'sobsz == %d and th == %d and dx == %d and hsv == %s and yita == %f \t (IOU=0.7): %f%% time:%fs\n' % \
                                    (sobsz, th, dx, hsv, yita, rate * 100, totaltime)
                        with open('.\\New\\all.txt','r+') as f:
                            f.readlines()
                            f.write(writefile)
                        #itchat.send(writefile, toUserName = 'filehelper')
    #prepare()
    #p = regression()
    #draw_YCrCb()
