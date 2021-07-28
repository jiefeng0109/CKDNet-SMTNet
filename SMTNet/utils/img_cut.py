import scipy.io as sio
import numpy as np
import cv2
import os
import heapq
from collections import defaultdict

img_len = 6
def judge(img_id,target_id,label):
    for i in range(img_len):
        if target_id not in label[img_id+i*5]:
            return True
    return False


def cut_im(img, position):
    w,h,c = img.shape
    position[0] = np.maximum(position[0] - 2, 0)
    position[1] = np.maximum(position[1] - 2, 0)
    position[2] = np.minimum(position[2] + 4, w - position[0])
    position[3] = np.minimum(position[3] + 4, h - position[1])
    im_data = img[position[1]:position[1] + position[3], position[0]:position[0] + position[2]]
    return im_data.copy()
def select_target(img_id,ind,target_id,label,label_data):
    target_g = np.array(label[img_id+ind*5][target_id].copy())
    all_target = label_data[img_id+ind*5].copy()
    dis = np.sqrt((all_target[:,-1] - target_g[-1])**2+(all_target[:,-2] - target_g[-2])**2)
    index = heapq.nsmallest(5,range(len(dis)),dis.take)
    select = all_target[index]
    return select
if __name__=="__main__":
    mode='test'
    if mode=='train':
        locations = {'4':['250-2970','1960-2690'],'5':['4570-3360','8450-2180']}
    else:
        locations = {'4':['1460-1400'],'5':['9590-2960']}

    pos = []
    neg = []
    path = r'H:\zeng\s\remote/{}/{}'
    label_path = r'H:\zeng\s\remote/label/{}.npz'
    save_path = r'H:\zeng\u\remote_track/' + mode
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for video_id,locs in locations.items():
        for loc in locs:
            img_path = os.path.join(path.format(video_id,loc),'{}.png')
            label = np.load(label_path.format(loc),allow_pickle=True)['label'].item()
            label_data = {}
            for img_id,label_v in label.items():
                temp = np.array(list(label_v.values()))
                label_data[img_id] = temp
            for img_id,label_v in label.items():
                if img_id>max(label.keys())-img_len*5:
                    continue
                print(img_id)
                imgs = [cv2.imread(img_path.format(img_id+i*5)) for i in range(6)]
                for target_id,target_l in label_v.items():
                    if target_l[5]==1 or judge(img_id,target_id,label):
                        continue
                    img_list = []
                    target_list = []
                    graph_list = []

                    for i in range(img_len-1):
                        target_list.append(label[img_id+i*5][target_id].copy())
                        img_list.append(cut_im(imgs[i],label[img_id+i*5][target_id].copy()))
                        graph_list.append(select_target(img_id,i,target_id,label,label_data).copy())
                    temp_img = img_list+[cut_im(imgs[img_len-1],label[img_id+(img_len-1)*5][target_id].copy())]
                    temp_target = target_list+[label[img_id+(img_len-1)*5][target_id].copy()]*2
                    temp_graph = graph_list+[select_target(img_id,img_len-1,target_id,label,label_data).copy()]
                    temp_dict = {"img":temp_img.copy(),"plist":temp_target.copy(),'graph':temp_graph.copy()}
                    pos.append(temp_dict)
                    for neg_id,neg_l in label[img_id+(img_len-1)*5].items():
                        temp_l = label[img_id+(img_len-2)*5][target_id].copy()
                        d = np.sqrt((neg_l[-1] - temp_l[-1])**2+(neg_l[-2] - temp_l[-2])**2)
                        if neg_id == target_id or d>20:
                            continue
                        temp_img = img_list + [cut_im(imgs[img_len - 1], neg_l.copy())]
                        temp_target = target_list + [neg_l.copy()]+[label[img_id+(img_len-1)*5][target_id].copy()]
                        temp_graph = graph_list+[select_target(img_id,img_len-1,neg_id,label,label_data).copy()]
                        temp_dict = {"img": temp_img.copy(), "plist": temp_target.copy(),"graph":temp_graph.copy()}
                        neg.append(temp_dict)
    np.savez_compressed(save_path+'/pos.npz',pos)
    np.savez_compressed(save_path+'/neg.npz',neg)
    print(len(pos),len(neg))
    a = 1

#
# data = {}
# for i in range(100,526):
#     print(i)
#     im = cv2.imread(im_path+str(i)+'.tif', cv2.IMREAD_GRAYSCALE)
#     position = sio.loadmat(move_mat_path+str(i)+'.mat')['bbox']
#     data[i] = cut_im(im,position)
#
#     if i%6 == 0 or i%6 == 1:
#         position = sio.loadmat(stop_mat_path + str(i) + '.mat')['bbox']
#         data[i].update(cut_im(im, position))
#         a = 1
# np.save(save_path+'images.npy',data)
