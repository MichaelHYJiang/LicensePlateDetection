import cv2
import numpy as np

def wait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contour(src, dst, color = (0,0,255)):
    contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dst, contours, -1, color, 3)
    return contours


img = cv2.imread('.\\0_Good\\IMG_20170122_141955.jpg')
sz = img.shape
if max(sz) > 1000:
    ratio = max(sz) / 800.0
else:
    ratio = 1


img = cv2.resize(img,(int(sz[1] / ratio), int(sz[0] / ratio)),
                     interpolation = cv2.INTER_CUBIC)
img0 = img.copy()
cv2.imshow('origin',img)

dst = cv2.Sobel(img, -1, 2, 0)
cv2.imshow('Sobel',dst)

dst1 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
ret, dst1 = cv2.threshold(dst1, 250, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold',dst1)
contours = contour(dst1, dst)

cv2.imshow('Contours',dst)
wait()

sz = img.shape

lower_blue=np.array([78,43,46])
upper_blue=np.array([110,255,255])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img1 = cv2.inRange(hsv, lower_blue, upper_blue)
#img1 = np.zeros((sz[0],sz[1]), img.dtype)
#for i in range(sz[0]):
#    for j in range(sz[1]):
#        if ((img[i][j][0]-100)**2+img[i][j][1]**2+img[i][j][2]**2)**0.5 < 60:
#            img1[i][j] = 255
#        else:
#            img1[i][j] = 0
#
#ret, img1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
#img1 = cv2.inRange(img,(0,0,255), (100,255,255))
cv2.imshow('ColorSeg', img1)

img2 = np.zeros(sz, img.dtype)
cv2.drawContours(img2, contours, -1, (255,255,255), 1)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

img3 = img1 * img2 #cv2.bitwise_and(img1, img2)

cv2.imshow('Threshold', img3)

contours = contour(img3, img)

cv2.imshow('Final', img)
wait()


for n in range(10):
    img4 = np.zeros(sz, img.dtype)
    contours = contour(img3, img4, color = (255, 255, 255))
    img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    ret, img4 = cv2.threshold(img4, 127, 255, cv2.THRESH_BINARY)
    img3 = img4.copy() - img3 #cv2.bitwise_or(img3, img4)

imgnew = img0.copy()
contours = contour(img3, imgnew)
cv2.imshow('Final', imgnew)
wait()

n = len(contours)
imgnew = img0.copy()
print n
for k in range(0 , n):
    x = contours[k]
    #print x.shape
    left = tuple(x[:,0][x[:,:,0].argmin()])
    right = tuple(x[:,0][x[:,:,0].argmax()])
    up = tuple(x[:,0][x[:,:,1].argmin()])
    down =tuple(x[:,0][x[:,:,1].argmax()])
    cv2.rectangle(imgnew,(left[0], up[1]),(right[0], down[1]),(0,0,255),3)

cv2.imshow('Final', imgnew)
wait()
