import numpy as np
import cv2
import shutil
import os
import multiprocessing

import scipy.io as scio
from functools import partial




def save_data(id,labels,save_paths,img_paths,name='8450-2180'):
    print(id)
    img = [cv2.imread(os.path.join(img_paths ,str(ind) + '.png')) for ind in [id - 5, id, id + 5]]
    img = np.concatenate(img, axis=2)
    label = []
    if name == '4570-3360':
        W, H = 1500, 700
    else:
        W, H = 1000, 1000
    # if name == '001':
    #     W, H = 400, 400
    # else:
    #     W, H = 400, 600
    for k,val in labels[id].items():
        if val[5] !=1:
            val = np.array(val,dtype=np.int)
            val[2:4] += val[0:2]
            val[0:2] = np.maximum(val[0:2], 0)
            val[2] = np.minimum(val[2], W)
            val[3] = np.minimum(val[3], H)
            val[2:4] -= val[0:2]
            label.append(val[:6])
    label = np.array(label)
    np.savez_compressed(os.path.join(save_paths,name+'-'+str(id)),img=img,label=label)
def setting(locations,mode,video='4',save_video = None,delect = True):
    if not save_video:
        save_video = video
    data_paths = r'/media/titan/SSD/zeng/s/remote/{}'.format(video)
    label_paths = r'/media/titan/SSD/zeng/s/remote/label/{}.npz'
    save_paths = r'/media/titan/SSD/zeng/u/remote/{}/{}'.format(save_video,mode)

    if delect:
        if os.path.exists(save_paths):
            shutil.rmtree(save_paths)
        os.makedirs(save_paths)
    pool = multiprocessing.Pool(processes=5)

    if mode=='train':
        for location in locations:
            img_paths = os.path.join(data_paths, location)
            labels = np.load(label_paths.format(location),allow_pickle=True)['label'].item()
            id = np.arange(5, max(labels.keys()), 5)
            pool.map(partial(save_data,labels=labels,save_paths=save_paths,img_paths=img_paths,name=location),id)
    if mode=='test':
        for location in locations:
            img_paths = os.path.join(data_paths, location)
            labels = np.load(label_paths.format(location),allow_pickle=True)['label'].item()
            id = np.arange(5, max(labels.keys()), 5)
            pool.map(partial(save_data,labels=labels,save_paths=save_paths,img_paths=img_paths,name=location),id)

if __name__ == "__main__":
    video_train = {'4':['250-2970','1960-2690'],'5':['4570-3360','8450-2180'],'6':['002']}
    video_test = {'4':['1460-1400',],'5':['9590-2960'],'6':['001']}
    video = '7'
    # video_train = {'4':['250-2970','1960-2690'],'5':['9590-2960','8450-2180'],'6':['002']}
    # video_test = {'4':['1460-1400',],'5':['4570-3360'],'6':['001']}
    # video = '10'

    if set(video).issubset(('4','5','6')):
        setting(video_train[video],'train',video)
        setting(video_test[video],'test',video)
    elif set(video).issubset(('7')):
        setting(video_train['4'],'train','4',video)
        setting(video_train['5'],'train','5',video,delect=False)
        setting(video_test['4'], 'test', '4', video)
        setting(video_test['5'], 'test', '5', video, delect=False)
