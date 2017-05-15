import cv2
import numpy as np


def DrawHist(hist, color, n = 256):        
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)    
    histImg = np.zeros([256,n,3], np.uint8)    
    hpt = int(0.9* 256);    
        
    for h in range(n):    
        intensity = int(hist[h]*hpt/maxVal)    
        cv2.line(histImg,(h,256), (h,256-intensity), color)    
            
    return histImg;

# begin
with open('.\\test.txt','r') as f:
    g = f.readlines()


k = 0
hist_h = np.zeros([180,1], 'float')
hist_s = np.zeros([256,1], 'float')
hist_v = np.zeros([256,1], 'float')

hist_h0 = np.zeros([180,1], 'float')
hist_s0 = np.zeros([256,1], 'float')
hist_v0 = np.zeros([256,1], 'float')

while k < len(g):
    x = g[k]
    y = x.split()
    if len(y) < 5:
        g.remove(x)
        continue
    x1 = int(y[1])
    y1 = int(y[2])
    x2 = x1 + int(y[3])
    y2 = y1 + int(y[4])
    img = cv2.imread('.\\0_Good\\'+y[0])
    
    sz = img.shape
    if max(sz) > 1000:
        n = 10
        x1 = x1 * 2
        x2 = x2 * 2
        y1 = y1 * 2
        y2 = y2 * 2
    img0 = img[y1:y2,x1:x2,:]
    
    show = False
    if show == True:
        cv2.imshow(y[0],img)
        cv2.imshow(y[0]+'->obj',img0)
        key = cv2.waitKey(0)
        #cv2.destroyAllWindows()
        if key == 32 or key == 27:#press space or esc to end
            break
        elif key == 2490368:#press up to go back
            k = k - 2
            cv2.destroyAllWindows()
        else:
            cv2.destroyAllWindows()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
    hh = cv2.calcHist([img], [0], None, [180], [0.0, 180.0])
    hs = cv2.calcHist([img], [1], None, [256], [0.0, 256.0])
    hv = cv2.calcHist([img], [2], None, [256], [0.0, 256.0])
    hh0 = cv2.calcHist([img0], [0], None, [180], [0.0, 180.0])
    hs0 = cv2.calcHist([img0], [1], None, [256], [0.0, 256.0])
    hv0 = cv2.calcHist([img0], [2], None, [256], [0.0, 256.0])
    hist_h += hh
    hist_s += hs
    hist_v += hv
    hist_h0 += hh0
    hist_s0 += hs0
    hist_v0 += hv0
    show_hist = False
    if show_hist == True:
        histImgh = DrawHist(hh, [255, 0, 0], 180)
        histImgs = DrawHist(hs, [0, 255, 0])
        histImgv = DrawHist(hv, [0, 0, 255])
        histImgh0 = DrawHist(hh0, [255, 0, 0], 180)
        histImgs0 = DrawHist(hs0, [0, 255, 0])
        histImgv0 = DrawHist(hv0, [0, 0, 255])
        cv2.imshow('histImgh', histImgh)
        cv2.imshow('histImgs', histImgs)
        cv2.imshow('histImgv', histImgv)
        cv2.imshow('histImgh0', histImgh0)
        cv2.imshow('histImgs0', histImgs0)
        cv2.imshow('histImgv0', histImgv0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    k = k + 1
    if k % 10 == 0:
        print k

cv2.destroyAllWindows()
histImghAll = DrawHist(hist_h, [255, 0, 0], 180)
histImgsAll = DrawHist(hist_s, [0, 255, 0])
histImgvAll = DrawHist(hist_v, [0, 0, 255])
cv2.imshow('h', histImghAll)
cv2.imshow('s', histImgsAll)
cv2.imshow('v', histImgvAll)
histImgh0All = DrawHist(hist_h0, [255, 0, 0], 180)
histImgs0All = DrawHist(hist_s0, [0, 255, 0])
histImgv0All = DrawHist(hist_v0, [0, 0, 255])
cv2.imshow('h0', histImgh0All)
cv2.imshow('s0', histImgs0All)
cv2.imshow('v0', histImgv0All)
cv2.waitKey(0)
cv2.destroyAllWindows()
