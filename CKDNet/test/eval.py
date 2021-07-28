# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import scipy.io as scio
import collections

import numpy as np
import os
import cv2
import heapq
from external import soft_nms,soft_nms_merge
from test.iou_array import box_iou,iou_array
road_dir = r"D:\zeng\data\s\remote\road.png"
source_dir = r"D:\zeng\data\s\remote\150/"
save_dir = "D:\zeng\data\\u\\remote\\"+'result/'
def read_road():
    im = cv2.imread(road_dir)
    label = np.zeros_like(im)

    temp = im[:,:,1] + im[:,:,2]
    index = np.argwhere(temp == 0)
    label[index[:,0],index[:,1],0] = 1#None
    temp = im[:, :, 1]
    index = np.argwhere(temp == 128)
    label[index[:,0],index[:,1],1] = 1#road
    temp = im[:, :, 2]
    index = np.argwhere(temp == 128)
    label[index[:,0],index[:,1],2] = 1#desroad

    return label
def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_single_label_dict(predict_dict,mode='all'):
    def ignor(data):
        d = data.copy().astype(np.int)
        index = np.argwhere(road[d[:,1],d[:,0]]==0).reshape(-1)
        return data[index]

    gboxes = {}
    # gboxes[0] = np.load(source_dir+'150.npy')
    gboxes[0] = np.load('test/test.npy')
    road = read_road()[3100:,:3000,2]
    gboxes[0] = ignor(gboxes[0])
    predict_dict[0] = ignor(predict_dict[0])
    return predict_dict, gboxes


def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def tpfp_default(det_bboxes, gt_bboxes, gt_ignore, iou_thr, area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): the detected bbox
        gt_bboxes (ndarray): ground truth bboxes of this image
        gt_ignore (ndarray): indicate if gts are ignored for evaluation or not
        iou_thr (float): the iou thresholds

    Returns:
        tuple: (tp, fp), two arrays whose elements are 0 and 1
    """
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + 1) * (
                det_bboxes[:, 3] - det_bboxes[:, 1] + 1)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp
    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap

def voc_eval(rboxes, gboxes, iou_th, use_07_metric, mode,confidence_th=0.4,area_ranges=None):
    rbox_images = list(rboxes.keys())
    tpfp = [
        tpfp_default(rboxes[j], gboxes[j], np.zeros(gboxes[j].shape[0]), iou_th,
                  area_ranges) for j in rbox_images
    ]
    tp, fp = tuple(zip(*tpfp))
    # calculate gt number of each scale, gts ignored or beyond scale
    # are not counted
    num_gts = np.zeros(1, dtype=int)
    for j, bbox in gboxes.items():
        if area_ranges is None:
            num_gts[0] += bbox.shape[0]
            # num_gts[0] += np.sum(np.logical_not(cls_gt_ignore[j]))
        # else:
        #     gt_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (
        #             bbox[:, 3] - bbox[:, 1] + 1)
        #     for k, (min_area, max_area) in enumerate(area_ranges):
        #         num_gts[k] += np.sum(
        #             np.logical_not(cls_gt_ignore[j])
        #             & (gt_areas >= min_area) & (gt_areas < max_area))
    # sort all det bboxes by score, also sort tp and fp
    cls_dets = np.vstack(rboxes.values())
    num_dets = cls_dets.shape[0]
    sort_inds = np.argsort(-cls_dets[:, -1])
    tp = np.hstack(tp)[:, sort_inds]
    fp = np.hstack(fp)[:, sort_inds]
    # calculate recall and precision with tp and fp
    tp = np.cumsum(tp, axis=1)
    fp = np.cumsum(fp, axis=1)
    eps = np.finfo(np.float32).eps
    recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
    precisions = tp / np.maximum((tp + fp), eps)
    # calculate AP
    # if scale_ranges is None:
    recalls = recalls[0, :]
    precisions = precisions[0, :]
    num_gts = num_gts.item()
    mode = 'area'
    ap = average_precision(recalls, precisions, mode)
    return {
        'num_gts': num_gts,
        'num_dets': num_dets,
        'recall': recalls,
        'precision': precisions,
        'ap': ap
    }
# def voc_eval(rboxes, gboxes, iou_th, use_07_metric, mode,confidence_th=0.4):
#     rbox_images = list(rboxes.keys())
#     fp = np.zeros(len(rbox_images))
#     tp = np.zeros(len(rbox_images))
#     box_num = 0
#
#     for i in range(len(rbox_images)):
#         rbox_image = rbox_images[i]
#         if len(rboxes[rbox_image]) > 0:
#
#             rbox_lists = np.array(rboxes[rbox_image])
#             keep_inds = (rbox_lists[:, -1] >= confidence_th)
#             rbox_lists = rbox_lists[keep_inds]
#             if len(gboxes[rbox_image]) > 0:
#                 gbox_list = gboxes[rbox_image]
#                 # gbox_list = np.array([[obj[0],obj[1],obj[0]+obj[2],obj[1]+obj[3] ]for obj in gboxes[rbox_image]])
#                 box_num = box_num + len(gbox_list)
#                 gbox_list = np.concatenate((gbox_list, np.zeros((np.shape(gbox_list)[0], 1))), axis=1)
#                 confidence = rbox_lists[:, -1]
#                 box_index = np.argsort(-confidence)
#
#                 rbox_lists = rbox_lists[box_index, :]
#                 for rbox_list in rbox_lists:
#                     if mode == 0:
#                         ixmin = np.maximum(gbox_list[:, 0], rbox_list[0])
#                         iymin = np.maximum(gbox_list[:, 1], rbox_list[1])
#                         ixmax = np.minimum(gbox_list[:, 2], rbox_list[2])
#                         iymax = np.minimum(gbox_list[:, 3], rbox_list[3])
#                         iw = np.maximum(ixmax - ixmin + 1., 0.)
#                         ih = np.maximum(iymax - iymin + 1., 0.)
#                         inters = iw * ih  # 重合的面积
#
#                         # union
#                         uni = ((rbox_list[2] - rbox_list[0] + 1.) * (rbox_list[3] - rbox_list[1] + 1.) +
#                                (gbox_list[:, 2] - gbox_list[:, 0] + 1.) *
#                                (gbox_list[:, 3] - gbox_list[:, 1] + 1.) - inters)  # 并的面积
#                         overlaps = inters / uni
#                     ovmax = np.max(overlaps)  # 取最大重合的面积
#                     # print(ovmax)
#                     jmax = np.argmax(overlaps)  # 取最大重合面积的索引
#                     rw = (rbox_list[2] + rbox_list[0])/2
#                     rh = (rbox_list[3] + rbox_list[1])/2
#                     gw = (gbox_list[jmax,2] + gbox_list[jmax,0])/2
#                     gh = (gbox_list[jmax,3] + gbox_list[jmax,1])/2
#                     # dist = np.sqrt((rw-gw)**2+(rh-gh)**2)
#                     if ovmax > iou_th:
#                     # if dist < iou_th:
#                         if gbox_list[jmax, -1] == 0:
#                             tp[i] += 1
#                             gbox_list[jmax, -1] = 1
#                         else:
#                             fp[i] += 1
#                     else:
#                         fp[i] += 1
#
#             else:
#                 fp[i] += len(rboxes[rbox_image][0]['bbox'])
#         else:
#             continue
#     rec = np.zeros(len(rbox_images))
#     prec = np.zeros(len(rbox_images))
#     if box_num == 0:
#         for i in range(len(fp)):
#             if fp[i] != 0:
#                 prec[i] = 0
#             else:
#                 prec[i] = 1
#     else:
#
#         fp = np.cumsum(fp)
#         tp = np.cumsum(tp)
#
#         prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#         rec = tp / (box_num + 0)
#     # avoid division by zero in case first detection matches a difficult ground ruth
#     ap = voc_ap(rec, prec, use_07_metric)
#
#     return rec, prec, ap, box_num

def get_detection(det):
    # det = merge_position(data)
    data = {}
    data[0] = []
    for key,val in det.items():
        temp = key.strip().split('/')[-1]
        temp = temp.strip().split('_')
        aw,ah = int(temp[-2])-3100,int(temp[-1][0:4])
        # aw,ah = int(temp[-2]),int(temp[-1][0:4])
        # data_temp = collections.defaultdict(list)
        for ind,val_d in val.items():
            # if ind not in [4,5,6,9]:
            #     re_s = 2
            # else:
            #     re_s = 1.3
            re_s = 1
            for d in val_d:
                if d[-1] > 0.1:
                    data[0].append([d[0] + ah, d[1] + aw, d[2] - d[0], d[3] - d[1], min(d[4]*re_s,0.99)])
        # if len(data_temp) == 0:
        #     continue

    return data

def det_nms(data,th = 0.3):
    def ignor(temp):
        pass
    def fitler(data_temp):
        # data_temp = np.array(data_temp)
        s = data_temp[:,2]*data_temp[:,3]
        ind_s = np.argsort(-s)
        data_temp = data_temp[ind_s]

        for ind,data_val in enumerate(data_temp):
            max_iou = box_iou(data_temp[ind:],data_val)
            max_iou = np.reshape(max_iou,[-1])
            # max_matix = np.array(heapq.nlargest(5, max_iou))
            max_index = np.argwhere(max_iou>0)
            # max_index = np.argwhere(max_iou == max_matix[index])
            if len(max_index)>2:
                tt = np.reshape(data_temp[ind+max_index], [-1, 5])


                index = np.argwhere(tt[:,4]>0.5)

                tt2 = tt[index].copy().reshape(-1,5)
                if tt2.shape[0] == 0 :
                    continue
                smean = np.mean(tt[index[:, 0], 4])
                if smean < 0.8*data_temp[ind, 4] :
                    continue
                tt2[:,2:4] += tt2[:,0:2]
                tt2[:,0:2] = np.maximum(tt2[:,0:2],data_temp[ind, 0:2])
                tt2[:,2:4] = np.minimum(tt2[:,2:4],data_temp[ind, 0:2]+data_temp[ind, 2:4])
                tt2[:, 2:4] -= tt2[:, 0:2]
                tiou = iou_array(tt,tt2)
                index = np.argwhere(tiou > 0.9)
                others = np.min(tt[:, 2] * tt[:, 3])

                self_s = data_temp[ind, 2] * data_temp[ind, 3]
                if len(index)>2:
                   data_temp[ind, 4] *= others/self_s
                elif len(index)>1:
                    data_temp[ind, 4] *= 0.7

        return data_temp
        # data[im_name].append(data_temp)
    sdata = {}
    # for key,dict_val in data.items():
    #     sdata[key] = {}
    #     for ind in range(12):
    #         sdata[key][ind + 1] = []
    #     np_temp = []
    for id,val in data.items():
        temp = np.array(val).astype(np.float32)
        if temp.shape[0] > 0:

            temp =  fitler(temp)
            temp[:,2:4] += temp[:,0:2]
            # if id in [4, 5, 6, 9]:
            #     temp[:,4] =np.minimum(temp[:,4]*1.2,0.99)
            temp_dict = {}
            temp_dict[0] = temp
            soft_nms(temp_dict[id], Nt=0.3, method=1)
            # soft_nms(temp_dict[id], sigma = 0.5, method=2)
            temp = [d for d in temp_dict[id] if d[4]>th]

            sdata[id] = np.array(temp,dtype=np.float32)


    return sdata

def plot_result(data_det,data_gt,th = 0.5):
    def plot(data,index,color):
        index = index.reshape(-1)
        da = data[index]
        for d in da:
            if d.shape[0]>4:
                cv2.putText(im, '%.1f' % (float(d[4])*100), (d[0], d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            d = np.array(d, dtype=np.int)
            cv2.rectangle(im, (d[0], d[1]), (d[0]+d[2], d[1]+d[3]), color)


    save_path = save_dir+r'images/'
    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    color_list = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
    im = cv2.imread(source_dir+'test.tif')
    # im = cv2.imread(source_dir+'150.bmp')
    det,gt = data_det[0].copy(), data_gt[0].copy()
    det[:, 2:4] -= det[:, 0:2]
    iou_mat = iou_array(det,gt.astype(np.float))
    gt_exist = np.zeros(gt.shape[0])
    det_exist = np.zeros(det.shape[0])
    iou_index = np.argwhere(iou_mat>th)
    for iou in iou_index:
        gt_exist[iou[0]] = 1
        det_exist[iou[1]] = 1

    index = np.argwhere(det_exist == 1)
    plot(det,index,color_list[0])
    index = np.argwhere(det_exist == 0)
    plot(det,index,color_list[2])
    index = np.argwhere(gt_exist == 0)
    plot(gt,index,color_list[1])


    cv2.imwrite(save_path+'result.png',im)

def save_result(data,save_path =save_dir+r'txt/'):

    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    for key, dict_val in data.items():
        txt_path = os.path.join(save_path,key+'.txt')

        with open(txt_path,'w') as f:
            num = 0
            for id, val in dict_val.items():
                for d in val:
                    num += 1
                    sores = d[4]
                    d = np.round(d).astype(np.int)
                    line = '{},{},{},{},{:.4f},{},-1,-1\n'.format(d[0],d[1],d[2]-d[0],d[3]-d[1],sores,id)
                    f.writelines(line)
            # if num == 0:
            #     line = '{},{},{},{},{:.4f},{},-1,-1\n'.format(0, 0, 0, 0, 0, 0)
            #     f.writelines(line)
def cal_map(rboxes, gboxes):
    # 0: horizontal standard 1: rotate standard
    mode = 0

    R, P, AP, F, num = [], [], [], [], []

    label = 'car'
    # data = get_detection(data)
    # data = det_nms(data,th = 0.4)

    # save_result(data)
    # np.save('det',data)
    # rboxes, gboxes = get_single_label_dict(data,mode='all')
    # rboxes, gboxes = [],[]
    # plot_result(rboxes,gboxes)
    iou_th = [0.3]
    cth=[0.5]
    for c in cth:
        print(c)
        for th in iou_th:

            result = voc_eval(rboxes, gboxes, th, False, mode=mode,confidence_th=c)
            # print(th)
            # print(result)
            # rec, prec, ap, box_num = voc_eval(rboxes, gboxes, th, False, mode=mode,confidence_th=0.4)

            recall = result['recall'][-1]
            precision = result['precision'][-1]
            F_measure = (2 * precision * recall) / (recall + precision)
            print('\n{}\t{}\tR:{}\tP:{}\tap:{}\tF:{}'.format(label, th, recall, precision, result['ap'], F_measure))

def read_det(th,mode):
    if mode=='val':
        save_path = save_dir+ r'txt/'
    else:
        save_path = save_dir+ r'txt/'
    txt_list = os.listdir(save_path)
    sdata = {}
    for txt_name in txt_list:
        txt_path = os.path.join(save_path,txt_name)
        dict_name = txt_name.split('.txt')[0]
        sdata[dict_name] = {}
        for ind in range(12):
            sdata[dict_name][ind] = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip(' ').split(',')
                x, y, w, h, s, c = int(line[0]), int(line[1]), int(line[2]), int(line[3]), float(line[4]), int(line[5])
                w =   x + w
                h = y + h
                # s = min(2*s,0.99)
                s = min(0.99,s *1.5)
                if s < th or w<2  or h<2 :
                    continue
                if (c == 4 and ( w/(h + 1e-4)< 0.25 or h/(w + 1e-4) < 0.25)) or ( w/(h + 1e-4)< 0.2 or h/(w + 1e-4) < 0.2):
                    continue
                sdata[dict_name][c].append([x, y, w, h, s])
    return sdata

if __name__ == "__main__":
    # th = 0.4
    # mode = 'val'
    # data = read_det(th,mode)
    # print('nms')
    # data = det_nms(data,th)
    # save_path = save_dir+ r'txt1/'
    # print('save')
    # save_result(data,save_path)
    # print('plot')
    # plot_result(data,mode)
    a,b =get_single_label_dict(None)
    plot_result(b,b)







