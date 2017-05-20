# !/usr/bin/python
# -*- coding:utf-8 -*-

# This script demonstrate a step-by-step result of the region proposal method
# based on edge features and color segmentation.

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def wait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# function 'contour' is used to draw contours found in image 'src'
# onto image 'dst'
def contour(src, dst, color = (0,0,255)):
    contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dst, contours, -1, color, 3)
    return contours

def suffix(num): # get a right file name suffix according to its number
    if num > 5:
        return '.jpg'
    else:
        return '.bmp'

if __name__ == '__main__':
    test = 56
    filepath = os.path.join('.', '0_Good', str(test) + suffix(test))
    img = cv2.imread(filepath)
    sz = img.shape

    # resize ratio
    if max(sz) > 1000:
        ratio = max(sz) / 800.0
    else:
        ratio = 1

    # resize image to a proper size
    img = cv2.resize(img,(int(sz[1] / ratio), int(sz[0] / ratio)),
                         interpolation = cv2.INTER_CUBIC)
    img0 = img.copy()
    
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.set_title('Original')
    ax.axis('off')
    ax.imshow(img0[:,:,(2,1,0)])
    
    img_sobel = cv2.Sobel(img, -1, 2, 0)
    ax = fig.add_subplot(222)
    ax.set_title('Sobel')
    ax.axis('off')
    ax.imshow(img_sobel[:,:,(2,1,0)])
    
    img_sobel1 = cv2.cvtColor(img_sobel, cv2.COLOR_BGR2GRAY)
    ret, img_sobel1 = cv2.threshold(img_sobel1, 250, 255, cv2.THRESH_BINARY)
    ax = fig.add_subplot(223)
    ax.set_title('Threshold')
    ax.axis('off')
    ax.imshow(img_sobel1, cmap = 'gray')

    contours = contour(img_sobel1, img_sobel)
    ax = fig.add_subplot(224)
    ax.set_title('Contours')
    ax.axis('off')
    ax.imshow(img_sobel[:,:,(2,1,0)])
    
    plt.show()

    # New image
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.set_title('Original')
    ax.axis('off')
    ax.imshow(img0[:,:,(2,1,0)])

    sz = img.shape

    lower_blue=np.array([78,43,46])
    upper_blue=np.array([110,255,255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_color = cv2.inRange(hsv, lower_blue, upper_blue)
    ax = fig.add_subplot(222)
    ax.set_title('ColorSeg')
    ax.axis('off')
    ax.imshow(img_color, cmap = 'gray')

    img2 = np.zeros(sz, img.dtype)
    cv2.drawContours(img2, contours, -1, (255,255,255), 1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

    img_combine = img_color * img2 

    # 'img_combine' is a binary picture, with max value = 1
    # It needs to be converted for showing
    img_show = np.array(img_combine, dtype = 'uint8')
    img_show *= 255
    ax = fig.add_subplot(223)
    ax.set_title('Combine Edge & Color')
    ax.axis('off')
    ax.imshow(img_show, cmap = 'gray')

    # Draw combined results on original image
    contours = contour(img_combine, img)

    ax = fig.add_subplot(224)
    ax.set_title('Result on Orginal Image')
    ax.axis('off')
    ax.imshow(img[:,:,(2,1,0)])
    plt.show()

    # New image
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.set_title('Original')
    ax.axis('off')
    ax.imshow(img0[:,:,(2,1,0)])

    ax = fig.add_subplot(222)
    ax.set_title('Combined Result on Orginal Image')
    ax.axis('off')
    ax.imshow(img[:,:,(2,1,0)])

    # Dialate
    for n in range(10):
        img_new = np.zeros(sz, img.dtype)
        contours = contour(img_combine, img_new, color = (255, 255, 255))
        img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
        ret, img_new = cv2.threshold(img_new, 127, 255, cv2.THRESH_BINARY)
        img_combine = img_new.copy() - img_combine

    imgnew = img0.copy()
    contours = contour(img_combine, imgnew)
    ax = fig.add_subplot(223)
    ax.set_title('Dilated')
    ax.axis('off')
    ax.imshow(imgnew[:,:,(2,1,0)])

    # Add bounding box
    n = len(contours)
    imgnew = img0.copy()
    for k in range(0 , n):
        x = contours[k]
        left = tuple(x[:,0][x[:,:,0].argmin()])
        right = tuple(x[:,0][x[:,:,0].argmax()])
        up = tuple(x[:,0][x[:,:,1].argmin()])
        down =tuple(x[:,0][x[:,:,1].argmax()])
        cv2.rectangle(imgnew,(left[0], up[1]),(right[0], down[1]),(0,0,255),3)

    ax = fig.add_subplot(224)
    ax.set_title('Bounding Box')
    ax.axis('off')
    ax.imshow(imgnew[:,:,(2,1,0)])
    plt.show()
