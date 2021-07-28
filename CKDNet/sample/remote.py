import cv2
import math
import numpy as np
import torch
import random
import string
import scipy.io as scio
import collections

from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_
from .utils import random_crop, draw_gaussian, gaussian_radius, draw_offset,draw_weight
from .data_loader import CervicalDataset
from test.iou_array import box_iou

def _full_image_crop(image, detections=None):
    height, width = image.shape[0:2]

    max_hw = max(height, width)
    center = [height // 2, width // 2]
    size = [max_hw, max_hw]

    image, border, offset = crop_image(image, center, size)
    if detections is not None:
        detections = detections.copy()
        detections[:, 0:4:2] += border[2]
        detections[:, 1:4:2] += border[0]
    return image, detections


def _resize_image(image, detections, size):
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))

    height_ratio = new_height / height
    width_ratio = new_width / width
    if detections is not None:
        detections = detections.copy()
        detections[:, 0:4:2] *= width_ratio
        detections[:, 1:4:2] *= height_ratio
    return image, detections


def _clip_detections(image, detections):
    detections = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds = ((detections[:, 2] - detections[:, 0]) > 0) & \
                ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections


def rot(width, detections):
    temp = detections.copy()
    detections[:, [0, 2]] = temp[:, [1, 3]]
    detections[:, [1, 3]] = width - temp[:, [2, 0]] - 1
    return detections


def kp_detection(db, k_ind, data_aug, debug):
    data_rng = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories = db.configs["categories"]
    input_size = db.configs["input_size"]
    output_size = db.configs["output_sizes"][0]

    border = db.configs["border"]
    lighting = db.configs["lighting"]
    rand_crop = db.configs["rand_crop"]
    rand_color = db.configs["rand_color"]
    rand_scales = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou = db.configs["gaussian_iou"][0]
    gaussian_rad = db.configs["gaussian_radius"]

    max_tag_len = 1024

    # allocating memory
    images = np.zeros((batch_size, 9, input_size[0], input_size[1]), dtype=np.float32)
    tl_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    br_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    tl_offsets = np.zeros((batch_size, 2, output_size[0], output_size[1]), dtype=np.float32)
    br_offsets = np.zeros((batch_size, 2, output_size[0], output_size[1]), dtype=np.float32)
    tl_sizes = np.zeros((batch_size, 2, output_size[0], output_size[1]), dtype=np.float32)
    br_sizes = np.zeros((batch_size, 2, output_size[0], output_size[1]), dtype=np.float32)
    # small_weights = np.zeros((batch_size,4, output_size[0], output_size[1]), dtype=np.float32)
    db_size = db.db_inds.size
    # if not debug and k_ind == 0:
    #     db.shuffle_inds(quiet=True)
    # db_ind = db.db_inds[k_ind]
    # k_ind = (k_ind + 1) % db_size
    # tdata = CervicalDataset(db.image_file(db_ind), batch_size)
    for b_ind in range(batch_size):
        if not debug and k_ind == 0:
            db.shuffle_inds(quiet=True)
        db_ind = db.db_inds[k_ind]
        k_ind = (k_ind + 1) % db_size
        tdata = CervicalDataset(db.image_file(db_ind), 1)
        image = tdata.imgs[0]
        detections = tdata.labels[0]
        # image = tdata.imgs[b_ind]
        # detections = tdata.labels[b_ind]
        # cropping an image randomly
        if not debug and rand_crop:
            image, detections = random_crop(image, detections, rand_scales, input_size, border=border)
        else:
            image, detections = _full_image_crop(image, detections)

        image, detections = _resize_image(image, detections, input_size)
        detections = _clip_detections(image, detections)

        if not debug and np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            width = image.shape[1]
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1

        if np.random.uniform() > 0.5:
            temp = image[:, :, 0:3]
            image[:, :, 0:3] = image[:, :, 6:9]
            image[:, :, 6:9] = temp

        if not debug:
            rot_num = np.random.randint(4)
            image = np.rot90(image, rot_num)
            for _ in range(rot_num):
                detections = rot(image.shape[0], detections)

            image = image.astype(np.float32) / 255
            # if rand_color:
            #     color_jittering_(data_rng, image)
            #     if lighting:
            #         lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
            normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))

        width_ratio = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        # flipping an image randomly

        # for ind, detection in enumerate(detections):
        #     # category = int(detection[-1]) - 1
        #     category = 0
        #
        #     xtl, ytl = detection[0], detection[1]
        #     xbr, ybr = detection[2], detection[3]
        #     xct, yct = (detection[2] + detection[0]) / 2., (detection[3] + detection[1]) / 2.
        #     # tdet = detection[:4].copy()
        #     # tdet[2:4]-=tdet[:2]
        #     # if p_detection.shape[0]==0 or l_detection.shape[0]==0:
        #     #     iou=0
        #     # else:
        #     #     iou = box_iou(p_detection, tdet).max()+ box_iou(l_detection, tdet).max()
        #     fxtl = (xtl * width_ratio)
        #     fytl = (ytl * height_ratio)
        #     fxbr = (xbr * width_ratio)
        #     fybr = (ybr * height_ratio)
        #     fxct = (xct * width_ratio)
        #     fyct = (yct * height_ratio)
        #
        #     xtl = int(fxtl)
        #     ytl = int(fytl)
        #     xbr = int(fxbr)
        #     ybr = int(fybr)
        #     xct = int(fxct)
        #     yct = int(fyct)
        #
        #     if gaussian_bump:
        #         width = detection[2] - detection[0]
        #         height = detection[3] - detection[1]
        #
        #         width = math.ceil(width * width_ratio)
        #         height = math.ceil(height * height_ratio)
        #
        #         if width * height < 9:
        #             gaussian_rad = 0
        #         else:
        #             gaussian_rad = -1
        #
        #         if gaussian_rad == -1:
        #             radius = gaussian_radius((height, width), gaussian_iou)
        #             radius = radius + min(min(width, height) - radius, 0)
        #             radius = max(0, int(radius))
        #         else:
        #             radius = gaussian_rad
        #         # gaussian = np.ones((2*radius+1,2*radius+1),dtype=np.float32)
        #         #
        #         # for gh in range(-radius,radius+1):
        #         #     for gw in range(-radius,radius+1):
        #         #         inter = min(width,width+gw)*min(height,height+gh)
        #         #         gaussian[gh+radius,gw+radius] = inter/(width*height+(width+gw)*(height+gh)-inter)
        #         gaussian = None
        #         draw_gaussian(tl_heatmaps[b_ind, category], [xtl, ytl], radius, gaussian=gaussian)
        #         draw_gaussian(br_heatmaps[b_ind, category], [xbr, ybr], radius, gaussian=gaussian)
        #         draw_gaussian(ct_heatmaps[b_ind, category], [xct, yct], radius, gaussian=gaussian)
        #
        #         draw_offset(offsets[b_ind, 0], [xtl, ytl], radius, k=height)
        #         draw_offset(offsets[b_ind, 1], [xtl, ytl], radius, k=width)
        #         draw_offset(offsets[b_ind, 2], [xbr, ybr], radius, k=height)
        #         draw_offset(offsets[b_ind, 3], [xbr, ybr], radius, k=width)
        #
        #         # draw_weight(small_weights[b_ind, 0], [xtl, ytl], radius, k=iou)
        #         # draw_weight(small_weights[b_ind, 1], [xtl, ytl], radius, k=iou)
        #         # draw_weight(small_weights[b_ind, 2], [xbr, ybr], radius, k=iou)
        #         # draw_weight(small_weights[b_ind, 3], [xbr, ybr], radius, k=iou)
        #         # print(iou)
        #     else:
        #         tl_heatmaps[category, ytl, xtl] = 1
        #         br_heatmaps[category, ybr, xbr] = 1
        #         ct_heatmaps[category, yct, xct] = 1
        #
        #     tag_ind = tag_lens[b_ind]
        #     tl_regrs[b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
        #     br_regrs[b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
        #     ct_regrs[b_ind, tag_ind, :] = [fxct - xct, fyct - yct]
        #     tl_tags[b_ind, tag_ind] = ytl * output_size[1] + xtl
        #     br_tags[b_ind, tag_ind] = ybr * output_size[1] + xbr
        #     ct_tags[b_ind, tag_ind] = yct * output_size[1] + xct
        #     tag_lens[b_ind] += 1
        for ind, detection in enumerate(detections):
            # category = int(detection[-1]) - 1
            category = 0

            xtl, ytl = detection[0], detection[1]
            xbr, ybr = detection[2], detection[3]
            xct, yct = (detection[2] + detection[0]) / 2., (detection[3] + detection[1]) / 2.
            fxtl = (xtl * width_ratio)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)
            fxct = (xct * width_ratio)
            fyct = (yct * height_ratio)

            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)-1
            ybr = int(fybr)-1

            if gaussian_bump:
                width = detection[2] - detection[0]
                height = detection[3] - detection[1]

                width = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)

                if width * height < 9:
                    gaussian_rad = 0
                else:
                    gaussian_rad = -1

                if gaussian_rad == -1:
                    radius = gaussian_radius((height, width), gaussian_iou)
                    radius = radius + min(min(width, height) - radius, 0)
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad
                # gaussian = np.ones((2*radius+1,2*radius+1),dtype=np.float32)
                #
                # for gh in range(-radius,radius+1):
                #     for gw in range(-radius,radius+1):
                #         inter = min(width,width+gw)*min(height,height+gh)
                #         gaussian[gh+radius,gw+radius] = inter/(width*height+(width+gw)*(height+gh)-inter)
                gaussian = None
                draw_gaussian(tl_heatmaps[b_ind,category], [xtl, ytl], radius, gaussian=gaussian)
                draw_gaussian(br_heatmaps[b_ind,category], [xbr, ybr], radius, gaussian=gaussian)
                tl_sizes[b_ind,:,ytl, xtl] = [fybr-fytl,fxbr-fxtl]
                br_sizes[b_ind,:,ybr, xbr] = [fybr-fytl,fxbr-fxtl]
                tl_offsets[b_ind,:,ytl, xtl] = [fytl-ytl,fxtl-xtl]
                br_offsets[b_ind,:,ybr, xbr] = [fybr-ybr,fxbr-xbr]
    images = torch.from_numpy(images)
    tl_heatmaps = torch.from_numpy(tl_heatmaps)
    br_heatmaps = torch.from_numpy(br_heatmaps)
    tl_offsets = torch.from_numpy(tl_offsets)
    br_offsets = torch.from_numpy(br_offsets)
    tl_sizes = torch.from_numpy(tl_sizes)
    br_sizes = torch.from_numpy(br_sizes)

    # images = torch.from_numpy(images)
    # tl_heatmaps = torch.from_numpy(tl_heatmaps)
    # br_heatmaps = torch.from_numpy(br_heatmaps)
    # ct_heatmaps = torch.from_numpy(ct_heatmaps)
    # tl_regrs = torch.from_numpy(tl_regrs)
    # br_regrs = torch.from_numpy(br_regrs)
    # ct_regrs = torch.from_numpy(ct_regrs)
    # tl_tags = torch.from_numpy(tl_tags)
    # br_tags = torch.from_numpy(br_tags)
    # ct_tags = torch.from_numpy(ct_tags)
    # tag_masks = torch.from_numpy(tag_masks)
    # offsets = torch.from_numpy(offsets)
    # small_weights = torch.from_numpy(small_weights+1)

    return {
               "xs": [images],
               "ys": [tl_heatmaps,br_heatmaps,tl_offsets,br_offsets,tl_sizes,br_sizes]
           }, k_ind


def sample_data(db, k_ind, data_aug=True, debug=False):
    return globals()[system_configs.sampling_function](db, k_ind, data_aug, debug)
