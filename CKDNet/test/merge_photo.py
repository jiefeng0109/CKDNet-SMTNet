import cv2
import numpy as np
import os
im_path = r'H:\data\zeng\s\remote\150/150.bmp'
label_path = r'H:\data\zeng\s\remote\150/150.npy'
im = cv2.imread(im_path)
label = np.load(label_path)

im = im[3100:5000,0:2956]
d = []
for l in label.astype(np.int):

    if l[0]<2956 and l[1]>3100:
        l[1] -= 3100
        cv2.rectangle(im,(l[0],l[1]),(l[0]+l[2],l[1]+l[3]),(0,0,255),1)
        d.append(l)
d = np.array(d)
np.save('test.npy',d)
# cv2.imwrite('test.tif',im)
