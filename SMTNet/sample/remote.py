import numpy as np
import torch
import random
import cv2
from config import system_configs
# from torch_geometric.data import Data,DataLoader
def kp_track(db, k_ind, data_aug=True, debug=False):
    batch_size = system_configs.batch_size
    images = []
    positions = np.zeros((batch_size,5,4), dtype=np.float32)
    last_positions= np.zeros((batch_size,2), dtype=np.float32)
    space = np.zeros((batch_size,6,16), dtype=np.float32)
    data_list = []
    labels = np.zeros((batch_size, ), dtype=np.float32)
    db_size = len(db.db_inds)

    for b_ind in range(batch_size):
        if k_ind==0:
            db.shuffle_inds()
        db_ind = db.db_inds[k_ind]
        k_ind = (k_ind + 1) % db_size

        pimg,labels[b_ind] = db.read_data(db_ind)
        pimgs,plist,pgraph = pimg["img"],pimg["plist"],pimg["graph"]
        plist = np.array(plist)
        plist[:-1,0:2] += np.random.rand(6,2)*2-1
        # plist[:-1,2:4] += np.random.rand(6,2)*1-0.5
        plist[:-1,2:4] += plist[:-1,0:2]
        # temp_c = plist[:-2,-2:]-plist[1:-1,-2:]
        # temp_wh = plist[:-2,2:4]-plist[1:-1,2:4]
        # positions[b_ind]=np.hstack([temp_c,temp_wh])
        positions[b_ind]=plist[:-2,0:4]-plist[1:-1,0:4]
        last_positions[b_ind]=plist[-3,-2:]-plist[-1,-2:]
        temp_graph = []
        for i,g in enumerate(pgraph):
            # temp_c = g[:,-2:]
            temp_c = g[1:,-2:]-g[0,-2:]
            temp_wh = g[1:,2:4]
            temp = np.hstack([temp_c,temp_wh]).reshape(-1) + np.random.rand(16)*2-1
            space[b_ind,i]= temp
            # space[b_ind,i]= temp/10
    rate = 10
    positions = torch.from_numpy(positions/rate)
    last_positions = torch.from_numpy(last_positions/rate)
    labels = torch.from_numpy(labels)
    space = torch.clamp(torch.from_numpy(space),min=-20,max=20)/rate

    return {
        "xs": [images,positions,space],
        "ys": [labels,last_positions]
    },k_ind

def sample_data(db, k_ind, data_aug=True, debug=False):
    return globals()[system_configs.sampling_function](db, k_ind, data_aug, debug)