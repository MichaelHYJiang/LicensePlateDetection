#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'licenseplate')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}

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

def vis_detections(im, class_name, dets, im_file, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    
    if len(inds) > 10:
        dets1 = dets[inds,:]
        dets1.sort()
        dets = dets1[:10, (1,2,3,4,0)]
        inds = range(10)
    
    print 'number of detection:\t', len(inds)

    imnum = int(im_file.split('.')[0][-4:])
    filepath = os.path.join(cfg.ROOT_DIR, 'data', 'demo', 'lp',
                            'GroundTruth.txt')
    with open(filepath, 'r') as fi:
        groundtruth = fi.readlines()
    gt = groundtruth[imnum - 1].strip().split()
    gt_abo = []
    print 'number of ground truth:\t' + gt[1]
    gt_str = 'Best IOU rate for each ground truth:\n'
    for i in range(int(gt[1])):
        x1 = int(gt[i * 4 + 2])
        y1 = int(gt[i * 4 + 3])
        w = int(gt[i * 4 + 4])
        h = int(gt[i * 4 + 5])
        ABO = 0
        for i in inds:
            bbox = dets[i, :4]
            IOU = calcIOU([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], \
                          [x1, y1, w, h])
            if IOU > ABO:
                ABO = IOU
        gt_str += str(ABO) + '\t'

    print gt_str

    # draw detection
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:.3f}'.format(score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=6, color='white')

    #draw ground truth
    for i in range(int(gt[1])):
        x1 = int(gt[i * 4 + 2])
        y1 = int(gt[i * 4 + 3])
        w = int(gt[i * 4 + 4])
        h = int(gt[i * 4 + 5])
        ABO = 0
        for i in inds:
            bbox = dets[i, :4]
            IOU = calcIOU([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], \
                          [x1, y1, w, h])
            if IOU > ABO:
                ABO = IOU
        ax.add_patch(
            plt.Rectangle((x1, y1),
                          w,
                          h, fill=False,
                          edgecolor='green', linewidth=2.0)
            )
        ax.text(x1 + w, y1 + h,
                'GT {:.3f}'.format(ABO),
                bbox=dict(facecolor='yellow', alpha=0.5),
                fontsize=6, color='black')
    
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.2f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

# function 'padzero' is used to get uniform name for demo images
def padzero(k, n_digit = 4):
    str_k = str(k)
    n_zero = 4 - len(str_k)
    str_k_with0 = '0' * n_zero + str_k
    return str_k_with0

def demo(net, image_name, classes, NUM):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', 'lp',
                            image_name + '_boxes')
    if NUM == 2000:
        box_file += '.mat'
    else:
        box_file += '1.mat'
    obj_proposals = sio.loadmat(box_file)['boxes']

    # Load the demo image
    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', 'lp', image_name)
    if int(image_name) < 6:
        im_file += '.bmp'
    else:
        im_file += '.jpg'
        
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposals)
    timer.toc()
    print ('Detection on image {:s} took {:.3f}s for '
           '{:d} object proposals').format(image_name,
                                           timer.total_time,
                                           boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.99
    NMS_THRESH = 0.3
    savefile = ''
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, im_file, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg_cnn_m_1024]',
                        choices=NETS.keys(), default='vgg_cnn_m_1024')
    parser.add_argument('--mod', dest='demo_model', help='Model to use [vgg_cnn_m_1024]'
                        , default='vgg16', type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'test.prototxt')

    caffemodel = args.demo_model
    
    caffe.set_mode_cpu()
    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # take n random samples to demo    
    n = 5

    # number of proposals for each picture
    # rp can only be 10000 or 2000 in this demo
    rp = 10000
    
    for i in range(n):
        print '------ %d / %d ------\n' % (i + 1, n)
        j = int(np.random.rand() * 5000 + 1)
        demo(net, padzero(j), ('licenseplate',), rp)
        plt.show()
        

    
