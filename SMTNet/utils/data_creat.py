import numpy as np
import cv2
import random
import copy
import scipy.io as scio

size = 32
save_path = '/home/ning/dataset/data/u/cnn//test/'


images = np.load('/home/ning/dataset/data/u/cnn/images.npy').item()

data = []
label = []


def write_data(data_temp,label_temp):
    data.append(data_temp / data_temp.mean())
    label.append(label_temp)
    data_temp = np.rot90(data_temp)
    data.append(data_temp / data_temp.mean())
    label.append(label_temp)

for i in range(500,525):
    print(i)
    pre_images = images[i]
    later_images = images[i+1]

    for key,val in pre_images.items():

        save_im = np.zeros((size,size,2),dtype=np.float16)
        temp_label = []

        if key in later_images:
            data_temp = np.array((val,later_images[key])).transpose((1,2,0))
            write_data(data_temp,1)
        else:
            continue

        t = 1
        if key+t in later_images:
            data_temp = np.array((val,later_images[key+t])).transpose((1,2,0))
            write_data(data_temp,0)
        elif key-t in later_images:
            data_temp = np.array((val, later_images[key - t])).transpose((1, 2, 0))
            write_data(data_temp, 0)


np.save(save_path+'label.npy',label)
np.save(save_path+'data.npy',data)
# np.save(save_path+'train_images.npy',train_images)
print(len(label))