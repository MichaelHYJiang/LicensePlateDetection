import cv2

with open('.\\Record.txt','r') as f:
    g = f.readlines()
    g = g[1:-1]


k = 0
d = []
new = []
intv = 1
while k < len(g):
    x = g[k]
    y = x.split()
    if len(y) < 5:
        g.remove(x)
        continue
    if len(new) <= k:
        new.append(y)
    x1 = int(new[k][1])
    y1 = int(new[k][2])
    x2 = x1 + int(new[k][3])
    y2 = y1 + int(new[k][4])
    img = cv2.imread('.\\0_Good\\'+new[k][0])
    
    sz = img.shape
    if max(sz) > 1000:
        #ratio = max(sz) / 800.0
        n = 10
        x1 = x1 * 2
        x2 = x2 * 2
        y1 = y1 * 2
        y2 = y2 * 2
        n = min(n, x1, y1, sz[0] - y2, sz[1] - x2)
        #print n
        img = img[(y1 - n):(y2 + n),(x1 - n):(x2 + n),:]
        cv2.rectangle(img,(n,n),(n+x2 - x1,n+y2 - y1),(0,0,255),2)
        cv2.rectangle(img,(n,n),(n+2,n+2),(0,255,0),2)
    else:
        #ratio = 1
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.rectangle(img,(x1,y1),(x1+2,y1+2),(0,255,0),2)
    #img = cv2.resize(img,(int(sz[1] / ratio), int(sz[0] / ratio)),
    #                interpolation = cv2.INTER_CUBIC)
    if k >= 0:#470
        cv2.imshow(new[k][0],img)
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
n = []
for x in new:
    a = ''
    for i in x:
        a += i + ' '
    n.append(a+'\n')

writefile = True    
if writefile == True:
    with open('.\\test0324.txt','w') as f:
        f.writelines(n)
print d
