import numpy as np
import torch
import random
import cv2
import heapq
import collections
from scipy.optimize import linear_sum_assignment
# from utils.img_cut import judge,select_target
# lambda_merge = lambda g:np.hstack([g[1:,-2:]-g[0,-2:],g[1:,2:4]]).reshape(-1)
lambda_merge = lambda g:np.hstack([g[1:,6:8]-g[0,6:8],g[1:,2:4]-g[1:,:2]]).reshape(-1)
img_len = 6
def judge(label,img_id):
    rind = 0
    for ind in range(img_len-1):
        if ind>=label.shape[0] or img_id-5*ind != label[-(ind+1),-1]:
            break
        rind = ind
    return rind
def select_target(idet,sd):
    dis = np.sqrt((idet[:,-1] - sd[-1])**2+(idet[:,-2] - sd[-2])**2)
    index = heapq.nsmallest(5,range(len(dis)),dis.take)
    sdet = idet[index].copy()
    det_c = (sdet[:,:2] + sdet[:,2:4])/2
    return np.hstack([det_c[1:]-det_c[0],sdet[1:,2:4]-sdet[1:,:2]]).reshape(-1)

def save_txt(track_npy,loc,max_img_id):
    # track_npy = np.load('track2.npy').item()
    track_dict = {}
    max_id = 1
    for img_id in range(15,max_img_id+1,5):
        next_img_id = img_id+5
        for target_id,target_list in track_npy[img_id].items():
            if img_id==max_img_id or target_id not in track_npy[next_img_id]:
                if np.argwhere(target_list[:,5]==0).shape[0]>2:
                    target_list[:,4]=max_id
                    target_list[:,2:4] -= target_list[:,:2]
                    track_dict[max_id] = target_list
                    max_id += 1
    write_lines = "{},{},{},{},{},{},-1,-1,-1,-1\n"
    with open("./results/{}/res.txt".format(loc),"w") as f:
        for target_id,target_list in track_dict.items():
            for tind,t in enumerate(target_list):
                if (t[5]==0 )and t[-1]> 10:
                    f.writelines(write_lines.format(int(t[-1]),int(t[4]),t[0],t[1],t[2],t[3]))
def testing(nett,result_dir,debug=False):
    locs = ['1460-1400','9590-2960']
    det_results = np.load('test_det.npy').item()
    for loc in locs:
        path = r'D:/graduate/pro/photo/remote_video/video/label/{}.npz'.format(loc)
        slabel = np.load(path)['label'].item()
# det_result = np.load('dibai_det.npy').item()
        det_result = {}
        for img_ids,label in slabel.items():
            img_name = '{}-{}'.format(loc,img_ids)
            if img_name in det_results:
                temp = np.array(list(label.values()))[:,:5]
                temp[:,4]=1
                temp[:,[2,3]]+=temp[:,[0,1]]
                det_result[img_name]=temp
        dets ={}
        max_img_id = 0
        for img_names,det in det_result.items():
            if loc in img_names:
                img_id = img_names.split('-')[-1]
                dets[img_id]=det
                max_img_id = max(max_img_id,int(img_id))
        track_data = tracker(nett,slabel,dets,max_img_id)
        save_txt(track_data,loc,max_img_id)

    # target_ids = ['33']
def tracker(nett,slabel,det_result,max_img_id,use_predict = False):
    max_id = 1
    img_id = 10
    # temp = collections.defaultdict(list)
    # for img_id in [0,5,10]:
    #     for id,tlabel in slabel[img_id].items():
    #         if tlabel[5] !=1:
    #             tlabel[5]=0
    #             temp[tlabel[4]].append(tlabel+[img_id])
    # match_dict = {}
    # for key,val in temp.items():
    #     val = np.array(val)
    #     val[:,2:4] += val[:,:2]
    #     if val[-1,-1]==10 and val.shape[0]==3:
    #         match_dict[key] = val
    idet = det_result[str(img_id)]
    idet =np.hstack([idet,np.zeros([idet.shape[0],1]),(idet[:,:2] + idet[:,2:4])/2,np.zeros([idet.shape[0],1])+img_id])
    next_dets = det_result[str(img_id+5)]
    match_dict = {}
    center = (next_dets[:,:2] + next_dets[:,2:4])/2
    for ind,tdet in enumerate(idet):
        if tdet[4]>0.4:
            dis = np.sqrt((center[:,0] - tdet[6])**2+(center[:,1] - tdet[7])**2)
            next_det_index = np.argwhere(dis<20)[:,0]
            if next_det_index.shape[0]==0:
                continue
            for nind in next_det_index:
                tdet[4] = max_id
                ndet = tdet.copy()
                ndet[:4] = 2*ndet[:4] - next_dets[nind,:4]
                ndet[6:8] = (ndet[2:4]+ndet[:2])/2
                ndet[5] = 1
                max_id += 1
                match_dict[ndet[4]] = np.vstack([ndet,tdet])

    image_match_dict = {}
    image_match_dict[10]=match_dict
    for img_id in range(15,max_img_id+1,5):
        # print(img_id)
        pre_img_id = img_id-5
        pre_match_dict = image_match_dict[pre_img_id].copy()
        pre_target_id = list(pre_match_dict.keys())
        pre_target_id_ind = np.arange(len(pre_target_id))
        idet = det_result[str(img_id)]
        index = idet[:,4]>0.3
        idet = idet[index]
        # idet[:,2:4] -= idet[:,:2]
        # idet =np.hstack([idet,np.arange(idet.shape[0]).reshape(-1,1),idet[:,:2] + idet[:,2:4]/2])
        idet =np.hstack([idet,np.zeros([idet.shape[0],1]),(idet[:,:2] + idet[:,2:4])/2,np.zeros([idet.shape[0],1])+img_id])
        matrix = np.zeros((pre_target_id_ind.shape[0],idet.shape[0]+pre_target_id_ind.shape[0]))
        matrix_position = np.zeros((pre_target_id_ind.shape[0],idet.shape[0]+pre_target_id_ind.shape[0],9))
        sdi_list = []
        sd_list = []
        lstm_list = []
        space_list = []
        id_list = []
        for tind,target_id in enumerate(pre_target_id):
            # target_id = '80'
            target_list = pre_match_dict[target_id].copy()
            judge_len = judge(target_list,pre_img_id)
            # rate = 0
            # if target_id>500:
            #     a = 1
            # if target_list[-1,5]>2:
            #     continue
            # if judge_len==0:
            #     rate = 0.2
            last_p = target_list[-1,6:8]
            dis = np.sqrt((idet[:,6] - last_p[0])**2+(idet[:,7] - last_p[1])**2)
            select_det_index = np.argwhere(dis<20)[:,0]
            if select_det_index.shape[0]==0:
                select_det = target_list[-1:]
            else:
                select_det = idet[select_det_index]
            tlstm = np.zeros((select_det.shape[0],img_len-1,4))
            tspace = np.zeros((select_det.shape[0],img_len,16))
            # temp_move = np.hstack([target_list[-5:-1,-2:]-target_list[-4:,-2:],target_list[-5:-1,2:4]-target_list[-4:,2:4]])
            temp_move = target_list[:-1,:4]-target_list[1:,:4]
            temp_space = []
            for ind in range(img_len):
                if pre_img_id-ind*5>0:
                    temp_select_det = select_target(det_result[str(pre_img_id-ind*5)],target_list[-1])
                    temp_space.append(temp_select_det)
            if judge_len<5:
                for ind in range(5-len(temp_space)):
                    temp_space += temp_space[-1:]
                for ind in range(4-temp_move.shape[0]):
                    temp_move=np.vstack([temp_move[:1],temp_move])
            temp_space = np.array(temp_space[::-1])

            for isd,sd in enumerate(select_det):
                tlstm[isd,:img_len-2] = temp_move[-4:]
                # tlstm[isd,img_len-2] = np.hstack([target_list[-1,-2:]-sd[-2:],target_list[-1,2:4]-sd[2:4]])
                tlstm[isd,img_len-2] = target_list[-1,:4]-sd[:4]
                tspace[isd,:img_len-1]=temp_space[-5:]
                tspace[isd,img_len-1]= select_target(det_result[str(img_id)],sd)
            tlstm = torch.from_numpy(tlstm.astype(np.float32))
            tspace = torch.from_numpy(tspace.astype(np.float32)/10)
            sdi_list.append(select_det_index)
            sd_list.append(select_det)
            lstm_list.append(tlstm)
            space_list.append(tspace)
            id_list.append(np.ones(select_det.shape[0],dtype=np.int)*target_id)
        id_list = np.hstack(id_list)
        lstm_list = torch.cat(lstm_list,dim=0)
        space_list = torch.cat(space_list,dim=0)
        xs = [None,lstm_list,space_list]
        cla_tracks,reg_tracks = nett.test(xs)

        for tind,target_id in enumerate(pre_target_id):
            id_index = id_list==target_id
            cla_track = cla_tracks[id_index]
            reg_track = reg_tracks[id_index]
            target_list = pre_match_dict[target_id].copy()
            judge_len = judge(target_list,pre_img_id)
            rate = 0
            if target_id>500:
                a = 1
            if target_list[-1,5]>2:
                continue
            if judge_len==0:
                rate = 0.2
            for ind,sind in enumerate(sdi_list[tind]):
                matrix[tind,sind]=max(cla_track[ind]-rate,0)
                temp = sd_list[tind][ind].copy()
                temp[4] = target_id
                matrix_position[tind,sind] = temp

            # index = np.argmax(cla_track)
            # track_det1 = select_det[index]
            temp_reg = np.mean(reg_track,axis=0)*10
            temp_det = target_list[-1].copy()
            temp_det[:2]-=temp_reg
            temp_det[2:4]-=temp_reg
            temp_det[6:8]-=temp_reg
            track_det2 = temp_det
            track_det2[5]+=1
            track_det2[-1] = img_id
            matrix[tind,idet.shape[0]+tind]=0.25-rate
            matrix_position[tind,idet.shape[0]+tind] = track_det2

        row_ind, col_ind = linear_sum_assignment(1-matrix)
        match_dict = {}
        for tind,dind in enumerate(col_ind):
            target_id = pre_target_id[tind]
            temp = pre_match_dict[target_id]
            if temp[-1,5]<3 and temp[-1,-1]==img_id-5:
                match_position = matrix_position[tind,dind]
                if match_position[4] != target_id or (not use_predict and dind>idet.shape[0]):
                    continue
                target_list = np.vstack([pre_match_dict[target_id],match_position])
                match_dict[target_id] = target_list
        if img_id<max_img_id:
            next_dets = det_result[str(img_id+5)]
            center = (next_dets[:,:2] + next_dets[:,2:4])/2
            for ind,tdet in enumerate(idet):
                if ind not in col_ind and tdet[4]>0.4:
                    dis = np.sqrt((center[:,0] - tdet[6])**2+(center[:,1] - tdet[7])**2)
                    next_det_index = np.argwhere(dis<20)[:,0]
                    if next_det_index.shape[0]==0:
                        continue
                    for nind in next_det_index:
                        tdet[4] = max_id
                        ndet = tdet.copy()
                        ndet[:4] = 2*ndet[:4] - next_dets[nind,:4]
                        ndet[6:8] = (ndet[2:4]+ndet[:2])/2
                        ndet[5] = 1
                        max_id += 1
                        match_dict[ndet[4]] = np.vstack([ndet,tdet])

        image_match_dict[img_id]=match_dict
        a = 1
    # np.save('track2.npy',image_match_dict)
    return image_match_dict