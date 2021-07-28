import numpy as np
import torch
import random
import os
import shutil
import cv2
import heapq
from utils.img_cut import judge,select_target
# lambda_merge = lambda g:np.hstack([g[1:,-2:]-g[0,-2:],g[1:,2:4]]).reshape(-1)
lambda_merge = lambda g:np.hstack([g[1:,-2:]-g[0,-2:],g[1:,2:4]-g[1:,:2]]).reshape(-1)
def testing(nett,result_dir,debug=False):
    video_id = '1460-1400'
    path = r'D:/graduate/pro/photo/remote_video/video/4/1460-1400/1460-1400.npz'
    img_path = r'D:/graduate/pro/photo/remote_video/video/4/1460-1400/s/{}.png'
    slabel = np.load(path)['label'].item()
    det_result = np.load('dibai_det.npy').item()
    img_len = 6
    # target_ids = ['33']
    target_ids = ['42','43']
    for ids in target_ids:
        ids_path = os.path.join(result_dir,ids)
        if os.path.exists(ids_path):
            shutil.rmtree(ids_path)
        os.mkdir(ids_path)
    for target_id in target_ids:
        # target_id = '80'
        target_list = []
        graph_list = []
        label_data = {}
        for img_id,label_v in slabel.items():
            temp = np.array(list(label_v.values()))
            label_data[img_id] = temp
        for i in range(img_len-1):
            target_list.append(slabel[i*5][target_id].copy())
            graph_list.append(select_target(0,i,target_id,slabel,label_data).copy())
        target_list = np.array(target_list)
        target_list[:,2:4] += target_list[:,:2]
        graph_list = np.array(graph_list)
        graph_list[:,:,2:4] += graph_list[:,:,:2]
        for ind in range(25,100,5):
            print(ind)
            idet = det_result['{}-{}'.format(video_id,ind)]
            index = idet[:,4]>0.4
            idet = idet[index]
            # idet[:,2:4] -= idet[:,:2]
            # idet =np.hstack([idet,np.arange(idet.shape[0]).reshape(-1,1),idet[:,:2] + idet[:,2:4]/2])
            idet =np.hstack([idet,np.arange(idet.shape[0]).reshape(-1,1),(idet[:,:2] + idet[:,2:4])/2])
            last_p = target_list[-1,-2:]
            dis = np.sqrt((idet[:,-2] - last_p[0])**2+(idet[:,-1] - last_p[1])**2)
            index = np.argwhere(dis<20)[:,0]
            select_det = idet[index]
            tlstm = np.zeros((index.shape[0],img_len-1,4))
            tspace = np.zeros((index.shape[0],img_len,16))
            # temp_move = np.hstack([target_list[-5:-1,-2:]-target_list[-4:,-2:],target_list[-5:-1,2:4]-target_list[-4:,2:4]])
            temp_move = target_list[-5:-1,:4]-target_list[-4:,:4]
            temp_space = np.zeros([img_len-1,16])
            for i,g in enumerate(graph_list[-5:]):
                temp_space[i]=lambda_merge(g)
            for isd,sd in enumerate(select_det):
                tlstm[isd,:img_len-2] = temp_move
                # tlstm[isd,img_len-2] = np.hstack([target_list[-1,-2:]-sd[-2:],target_list[-1,2:4]-sd[2:4]])
                tlstm[isd,img_len-2] = target_list[-1,:4]-sd[:4]
                tspace[isd,:img_len-1]=temp_space
                dis = np.sqrt((idet[:,-1] - sd[-1])**2+(idet[:,-2] - sd[-2])**2)
                # index = np.argwhere(dis<15)[:,0]
                index = heapq.nsmallest(5,range(len(dis)),dis.take)
                tspace[isd,img_len-1]= lambda_merge(idet[index])
            tlstm = torch.from_numpy(tlstm.astype(np.float32))
            tspace = torch.from_numpy(tspace.astype(np.float32)/10)
            xs = [None,tlstm,tspace]
            cla_track,reg_track = nett.test(xs)
            index = np.argmax(cla_track)
            #
            track_det1 = select_det[index]
            # else:
            temp_reg = np.mean(reg_track,axis=0)*10
            temp_det = target_list[-1]
            temp_det[:2]-=temp_reg
            temp_det[2:4]-=temp_reg
            temp_det[-2:]-=temp_reg
            track_det2 = temp_det

            if cla_track[index]>0.5:
                track_det1[4] = target_list[-1,4]
                target_list = np.vstack([target_list,track_det1])
            else:
                target_list = np.vstack([target_list,track_det2])
            dis = np.sqrt((idet[:,-1] - target_list[-1,-1])**2+(idet[:,-2] - target_list[-1,-2])**2)
            # index = np.argwhere(dis<15)[:,0]
            dis_index = heapq.nsmallest(5,range(len(dis)),dis.take)
            graph_list = np.vstack([graph_list,[idet[dis_index]]])
            img = cv2.imread(img_path.format(ind))
            # plot_img = lambda im,oval,color:cv2.rectangle(img,(int(oval[0]),int(oval[1])),(int(oval[2]+oval[0]),int(oval[3]+oval[1])),color,1)
            plot_img = lambda im,oval,color:cv2.rectangle(img,(int(oval[0]),int(oval[1])),(int(oval[2]),int(oval[3])),color,1)
            for st in idet[dis_index]:
                plot_img(img,st,(0,255,255))#yellow,neighboring vehicles
            for st in select_det:
                plot_img(img,st,(255,255,0))#low blue,candidate
            plot_img(img,track_det2,(0,255,0))#green,predict
            if target_id in slabel[ind]:
                track_det3 = np.array(slabel[ind][target_id].copy())
                track_det3[2:4]+=track_det3[:2]
                plot_img(img,track_det3,(0,0,255))#red,gt
            if cla_track[index]>0.5:
                plot_img(img,track_det1,(255,0,0))#blue,success track


            cv2.imwrite(result_dir+'/{}/{}.png'.format(target_id,ind),img)
            a = 1