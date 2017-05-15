# -*- coding: cp936 -*-
import cv2
import os

def draw_all_boxes(n, coord, img, color_rect = (0, 0, 255), color_cornor = (0, 255, 0)):
    img0 = img.copy()
    for i0 in range(n):
        start = 2 + i0 * 4
        x1 = int(coord[start + 0])
        y1 = int(coord[start + 1])
        x2 = x1 + int(coord[start + 2])
        y2 = y1 + int(coord[start + 3])
        cv2.rectangle(img0, (x1, y1), (x2, y2), color_rect, 5)
        cv2.rectangle(img0, (x1, y1), (x1 + 8, y1 + 8), color_cornor, 5)
    return img0

def draw_box(coord, img, color_rect = (0, 0, 255), color_cornor = (0, 255, 0)):
    img0 = img.copy()
    x1 = int(coord[0])
    y1 = int(coord[1])
    x2 = x1 + int(coord[2])
    y2 = y1 + int(coord[3])
    cv2.rectangle(img0, (x1, y1), (x2, y2), color_rect, 5)
    cv2.rectangle(img0, (x1, y1), (x1 + 8, y1 + 8), color_cornor, 5)
    return img0
    

n1 = 101
n2 = 300

with open('.\\output\\10000 - 3000\\result2000_%d_%d.txt'%(n1,n2),'r') as f:
    g = f.readlines()

k = 0
f = os.listdir('.\\data\\demo\\lp\\')
f = [x for x in f if x.strip().split('.')[-1].lower() in ('jpg', 'bmp')]
n_total = len(f)
print n_total

new = [x.split('\t') for x in g if len(x.split()) > 1]
intv = 0
while k < len(new):
    x = g[k]
    y = x.split('\t')
    if len(y) < 2:
        g.remove(x)
        continue

    n_plate = int(new[k][1])
    img = cv2.imread(new[k][0])

    img0 = draw_all_boxes(n_plate, new[k], img)
    
    sz = img0.shape
    ratio = max(sz) / 800.0
    img0 = cv2.resize(img0,(int(sz[1] / ratio), int(sz[0] / ratio)),
                    interpolation = cv2.INTER_CUBIC)
    
    if k > 0:
        cv2.imshow(new[k][0].split('\\')[-1],img0)
        key = cv2.waitKey(0)
        #cv2.destroyAllWindows()
        if key == 32 or key == 27:#press space or esc to end
            break
        elif key == 2490368:#press up to go back
            k = k - 2
            cv2.destroyAllWindows()
        elif key == 3014656:#press delete to record for future deletion
            d.append(y[0])
            cv2.destroyAllWindows()
        elif key > 128:
            cv2.destroyAllWindows()
        elif chr(key).lower() == 'w':#press w to move up the left-up corner
            new[k][2] = str(int(new[k][2]) - intv)
            k = k - 1
        elif chr(key).lower() == 's':#press s to move down the left-up corner
            new[k][2] = str(int(new[k][2]) + intv)
            k = k - 1
        elif chr(key).lower() == 'a':#press a to move left the left-up corner
            new[k][1] = str(int(new[k][1]) - intv)
            k = k - 1
        elif chr(key).lower() == 'd':#press d to move right the left-up corner
            new[k][1] = str(int(new[k][1]) + intv)
            k = k - 1
        elif chr(key).lower() == 'q':#press q to shorten the width
            new[k][3] = str(int(new[k][3]) - intv)
            k = k - 1
        elif chr(key).lower() == 'e':#press e to lengthen the width
            new[k][3] = str(int(new[k][3]) + intv)
            k = k - 1
        elif chr(key).lower() == 'z':#press z to shorten the height
            new[k][4] = str(int(new[k][4]) - intv)
            k = k - 1
        elif chr(key).lower() == 'c':#press c to lengthen the height
            new[k][4] = str(int(new[k][4]) + intv)
            k = k - 1
        elif chr(key).lower() == 'x':#press x to change the interval
            if intv == 5:
                intv = 1
            else:
                intv = 5
            print 'latest intv:%d'%intv
            k = k - 1
        else:
            cv2.destroyAllWindows()
    
    k = k + 1

cv2.destroyAllWindows()

