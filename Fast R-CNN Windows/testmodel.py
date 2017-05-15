

def calcIOU(b1, b2):
    x11 = b1[0]
    y11 = b1[1]
    x12 = b1[2] + x11
    y12 = b1[3] + y11
    x21 = b2[0]
    y21 = b2[1]
    x22 = b2[2] + x21
    y22 = b2[3] + y21
    x1 = max(x11,x21)
    y1 = max(y11,y21)
    x2 = min(x12,x22)
    y2 = min(y12,y22)
    I = max(x2 - x1, 0) * max( y2 - y1,0)
    x1 = min(x11,x21)
    y1 = min(y11,y21)
    x2 = max(x12,x22)
    y2 = max(y12,y22)
    U = max(x2 - x1, 1) * max( y2 - y1,1)
    return float(I)/U

def testgt(rp, gt, thresh = 0.7):
    n_pic = len(gt)
    if len(rp) != n_pic:
        raise IndexError
    n = 0
    #if n_pic > 5:
    #    n_pic = 5
    
    for i in range(n_pic):
        gt_i = gt[i].strip().split('\t')
        rp_i = rp[i].strip().split('\t')
        n_gt = int(gt_i[1])
        n_rp = int(rp_i[1])
        #if n_rp > 5:
        #    n_rp = 5
        for j in range(n_gt):
            box_j_gt_i = [int(gt_i[j * 4 + 2]), \
                          int(gt_i[j * 4 + 3]), \
                          int(gt_i[j * 4 + 4]), \
                          int(gt_i[j * 4 + 5])]
            for k in range(n_rp):
                box_k_rp_i = [int(rp_i[k * 4 + 2]), \
                              int(rp_i[k * 4 + 3]), \
                              int(rp_i[k * 4 + 4]), \
                              int(rp_i[k * 4 + 5])]
                if calcIOU(box_j_gt_i, box_k_rp_i) > thresh:
                    n += 1
                    break
    return n

def gettotalnum(record):
    n_pic = len(record)
    n = 0
    for i in range(n_pic):
        record_i = record[i].strip().split('\t')
        n += int(record_i[1])
    return n

if __name__ == '__main__':
    n1 = 1
    n2 = 5000
    test = 5000
    
    rp_num = 10000
    if test == 3000:
        with open('H:\\fast-rcnn-master - changed\\output\\10000 - 3000\\result%d_%d_%d.txt' % (rp_num, n1, n2),'r') as f:
            rp = f.readlines()
    else:
        #with open('H:\\fast-rcnn-master - changed\\output\\2000 - 5000\\result%d_%d_%d.txt'%(rp_num,n1,n2),'r') as f:
        with open('H:\\fast-rcnn-master - changed\\output\\default\\result%d_%d_%d.txt'%(rp_num,n1,n2),'r') as f:
            rp = f.readlines()

    
    print len(rp)

    with open('I:\\new_record.txt') as f:
        gt = f.readlines()

    
    gt = gt[n1 - 1: n2]

    result1 = testgt(rp, gt)
    result2 = testgt(gt, rp)

    num_gt = gettotalnum(gt)
    num_rp = gettotalnum(rp)

    with open('H:\\fast-rcnn-master - changed\\output\\10000 - 3000\\0result.txt','r+') as f:
        g = f.readlines()
        f.write('\n')
        f.write('%d - %d\n' % (n1, n2))
        f.write(str(result1) + '\t')
        f.write(str(result2) + '\t')
        f.write(str(num_gt) + '\t')
        f.write(str(num_rp) + '\t')
