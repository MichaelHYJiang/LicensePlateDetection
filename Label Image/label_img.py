# !/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import time
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

def draw_box(coord, img, line = 5, color_rect = (0, 0, 255), color_cornor = (0, 255, 0)):
    img0 = img.copy()
    x1 = int(coord[0])
    y1 = int(coord[1])
    x2 = x1 + int(coord[2])
    y2 = y1 + int(coord[3])
    cv2.rectangle(img0, (x1, y1), (x2, y2), color_rect, line)
    cv2.rectangle(img0, (x1, y1), (x1 + 8, y1 + 8), color_cornor, line)
    return img0

def norm_size_img(img, size = 800.0):
    sz = img.shape
    ratio = max(sz) / size
    img0 = cv2.resize(img,(int(sz[1] / ratio), int(sz[0] / ratio)),
                    interpolation = cv2.INTER_CUBIC)
    return img0, ratio

def bbox2str(bbox):
    bbox_str = ''
    for x in bbox:
        x1 = str(x[0])
        y1 = str(x[1])
        w = str(x[2])
        h = str(x[3])
        bbox_str += '\t'.join([x1, y1, w, h]) + '\t'
    return bbox_str
        
def on_mouse(event, x, y, flags, param):
    global x1, y1, w, h
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = int(x * param)
        y1 = int(y * param)
    elif event == cv2.EVENT_LBUTTONUP:
        w = int(x * param - x1)
        h = int(y * param - y1)
        
def on_mouse1(event, x, y, flags, param):
    global x1, y1, w, h
    ratio = param[0]
    n = param[1]
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 += int(x * ratio) - n
        y1 += int(y * ratio) - n
    elif event == cv2.EVENT_LBUTTONUP:
        w = int(x * ratio) - n
        h = int(y * ratio) - n
#===============================================================================
# start
#===============================================================================
try:
    f = open('new_record.txt','r')
    g = f.readlines()
    f.close()
except IOError:
    f = open('new_record.txt','w')
    f.close()
    print 'create file'
    g = []


already = set([x.split('\t')[0] for x in g])

instruction = '''press space or esc to end
press delete to record filename for future deletion
press w to move up the left-up corner
press s to move down the left-up corner
press a to move left the left-up corner
press d to move right the left-up corner
press q to shorten the width
press e to lengthen the width
press z to shorten the height
press c to lengthen the height
press x to change the interval
press v to change view
press r to append a bounding box
press f to save b-boxes for current picture
press g to delete the last b-boxes for current picture
'''

if len(g) == 0: # if record file is null, then show directions
    print instruction
    os.system('pause')


k = 0
filelist = os.listdir('.')
img_format = set(['jpg', 'bmp', 'jpeg', 'png', 'dib', 'jpe', 'jfif', 'gif',
                  'tif', 'tiff'])
allimage = set([x for x in filelist if x.split('.')[-1].lower() in img_format])
                #number of all picture files
n_total = len(allimage)
remain = list(allimage - already)  # skip files that already annotated
remain.sort()
d = []          # to record files that need to be deleted
intv = 1        # distance that one keyboard change(w,a,s,d,q,e,z,c) moves
view = True     # to denote the state of showing whole picture(True) or
                # just ROI part(False)
global x1, y1, w, h
[x1, y1, w, h] = [0, 0, 0, 0]   # current bounding box coordinates
line = 5        # linewidth when drawing boxes
bbox = []       # to record all bounding boxes
n_box = 0       # number of boxes in current picture
new = []
num_new = 0

t_start = time.time()   # timing
l_time = time.localtime()

while k < len(remain) and k >= -len(g):
    if k < 0:
        filename = g[k].strip().split('\t')[0]
    elif k < len(new):
        filename = new[k].strip().split('\t')[0]
    else:
        filename = remain[k]
        
    cv2.namedWindow(filename)
    
    img = cv2.imread(filename)

    img0 = img.copy()
    if k < 0:
        y = g[k].strip().split('\t')
        img0 = draw_all_boxes(int(y[1]), y, img0)
    elif k < len(new):
        y = new[k].strip().split('\t')
        img0 = draw_all_boxes(int(y[1]), y, img0)
    else:
        coord = [x1, y1, w, h]
        img0 = draw_box(coord, img, line = line)
        
    sz = img0.shape
    if view:
        img0,ratio = norm_size_img(img0)
        cv2.setMouseCallback(filename, on_mouse, ratio)
        cv2.imshow(filename,img0)
    else:
        margin = 10
        x2 = x1 + w
        y2 = y1 + h
        n = min(margin, x1, y1, sz[0] - y2, sz[1] - x2)
        img0 = img0[(y1 - n):(y2 + n),(x1 - n):(x2 + n),:]
        img0,ratio = norm_size_img(img0, 400.0)
        cv2.setMouseCallback(filename, on_mouse1, [ratio, n])
        cv2.imshow(filename,img0)
    key = cv2.waitKey(0)
        
    if key == 32 or key == 27:#press space or esc to end
        break
    elif key == 2490368:#press up to go back
        k = k - 1
        cv2.destroyAllWindows()
    elif key == 3014656:#press delete to record for future deletion
        d.append(y[0])
        os.remove('.\\车牌\\' + filename)
        new_filename = padzero(n_total) + '.jpg'
        os.rename('.\\车牌\\' + new_filename, '.\\车牌\\' + filename)
        n_total -= 1
        cv2.destroyAllWindows()
    elif key > 128 or key < 0:
        k += 1
        cv2.destroyAllWindows()
    elif chr(key).lower() == 'w':#press w to move up the left-up corner
        y1 -= intv
    elif chr(key).lower() == 's':#press s to move down the left-up corner
        y1 += intv
    elif chr(key).lower() == 'a':#press a to move left the left-up corner
        x1 -= intv
    elif chr(key).lower() == 'd':#press d to move right the left-up corner
        x1 += intv
    elif chr(key).lower() == 'q':#press q to shorten the width
        w -= intv
    elif chr(key).lower() == 'e':#press e to lengthen the width
        w += intv
    elif chr(key).lower() == 'z':#press z to shorten the height
        h -= intv
    elif chr(key).lower() == 'c':#press c to lengthen the height
        h += intv
    elif chr(key).lower() == 'x':#press x to change the interval
        if intv == 1:
            intv = 5
        elif intv == 5:
            intv = 20
        elif intv == 20:
            intv = 100
        elif intv == 100:
            intv = 1
        print 'latest intv:%d'%intv
    elif chr(key).lower() == 'v':#press v to change view
        view = not view
        if line == 5:
            line = 2
        else:
            line = 5
    elif chr(key).lower() == 'r':#press r to append a bounding box
        #n_box += 1
        bbox.append([x1, y1, w, h])
        print bbox
    elif chr(key).lower() == 'f':#press f to save b-boxes for current picture
        bbox_str = bbox2str(bbox)
        n_box += len(bbox)
        str_to_save = filename + '\t' + str(len(bbox)) + '\t' + bbox_str + '\n'
        print str_to_save
        num_new += 1
        new.append(str_to_save)
        bbox = []
    elif chr(key).lower() == 'g':#press g to delete the last b-boxes for current picture
        index = k - len(g) - 1
        if index + 1 > len(new):
            if len(bbox) > 0:
                trash = bbox.pop()
        else:
            y = new[index].strip().split('\t')
            y[1] = str(int(y[1]) - 1)
            if y[1] == '0':
                trash = new.pop()
            else:
                trash = y.pop()
                trash = y.pop()
                trash = y.pop()
                trash = y.pop()
                new[index] = '\t'.join(y).strip() + '\n'
    else:
        k += 1
        cv2.destroyAllWindows()

cv2.destroyAllWindows()

t_end = time.time()

writefile = False
write = raw_input('save?(y/n):')
if write[0].lower() == 'y':
    writefile = True
if writefile == True:
    with open('.\\new_record.txt','w') as f:
        f.writelines(g)
        f.writelines(new)
#print d
    total_min = (t_end - t_start) / 60.0
    total_num = len(g) + len(new)
    log_str = '\n\n\tstart: ' + str(l_time.tm_hour) + ':' + str(l_time.tm_min) + \
              ('\tduration: %.2f min\n\tlabel: %d pics\t(with %d objects)\n\tspeed: %.2f pics/min\t(%d pics labeled)'\
               % (total_min, len(new), n_box, (len(new) / total_min), total_num))
    predict_time = (n_total - total_num) / (len(new) / total_min)
    predict = 'remaining time：%.2f min( %.2f hrs or %.2f days）' % (predict_time, predict_time / 60, predict_time / 60 / 14)
              
    print log_str
    print predict
