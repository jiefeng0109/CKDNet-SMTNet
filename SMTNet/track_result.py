import numpy as np
import cv2
import os
color = [[0,255,0],[0,0,0]]
def save_txt():
    track_npy = np.load('track2.npy').item()
    track_dict = {}
    max_id = 1
    for img_id in range(15,100,5):
        next_img_id = img_id+5
        for target_id,target_list in track_npy[img_id].items():
            if img_id==95 or target_id not in track_npy[next_img_id]:
                if np.argwhere(target_list[:,5]==0).shape[0]>2:
                    target_list[:,4]=max_id
                    target_list[:,2:4] -= target_list[:,:2]
                    track_dict[max_id] = target_list
                    max_id += 1
    write_lines = "{},{},{},{},{},{},1,-1,-1,-1\n"
    with open("res.txt","w") as f:
        for target_id,target_list in track_dict.items():
            for t in target_list:
                if t[5]==0 and t[-1]> 10:
                    f.writelines(write_lines.format(int(t[-1]),int(t[4]),t[0],t[1],t[2],t[3]))

def plot():
    track_npy = np.load('track2.npy').item()
    track_dict = {}
    max_id = 1
    path = r'D:\graduate\pro\photo\remote_video\video\4\1960-2690\show\{}.png'
    spath = r'G:\data\u\remote_track\out\{}.png'
    for img_id in range(15,100,5):
        print(img_id)
        img_path = path.format(img_id)
        img=cv2.imread(img_path)
        for target_id,target_list in track_npy[img_id].items():
            oval = target_list[-1].astype(np.int)
            if oval[5]<2:
                cv2.rectangle(img,(oval[0],oval[1]),(oval[2],oval[3]),color[oval[5]],1)
        cv2.imwrite(spath.format(img_id),img)

def save_gt():
    locs = ['1460-1400','9590-2960','001','4570-3360','8450-2180']
    path = r'H:\zeng\s\remote\label/{}.npz'
    for loc in locs:
        slabel = np.load(path.format(loc),allow_pickle=True)['label'].item()
        write_lines = "{},{},{},{},{},{},1,1,1\n"
        sp = "./results/gt/{}/gt/".format(loc)
        if not os.path.exists(sp):
            os.makedirs(sp)
        with open(sp+"/gt.txt","w") as f:
            for img_id,vlabel in slabel.items():
                if img_id>10 and img_id<max(slabel.keys()):
                    for id,label in vlabel.items():
                        if label[5]==0:
                            f.writelines(write_lines.format(img_id,label[4],label[0],label[1],label[2],label[3]))

if __name__=="__main__":
    save_gt()