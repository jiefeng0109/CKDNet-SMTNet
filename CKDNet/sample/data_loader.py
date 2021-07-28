from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from PIL import Image

from config import system_configs


class CervicalDataset(object):
    """Cervical dataset."""

    def __init__(self, data_path, batch_size,patch_size=[256,256], transform=None):
        self.patch_size = patch_size
        self.transform = transform
        self.imgs,self.labels = self.load_data(data_path, batch_size)
        # raise IOError
    def crop_patch(self, img, labels,img_num):
        H, W, C = img.shape
        px, py = self.patch_size
        spatch  = []
        slabel = []
        for i in range(img_num):
            obj_num = labels.shape[0]
            index = np.random.randint(0, obj_num)
            ux, uy, vx, vy = labels[index, :4]

            nx = np.random.randint(max(vx - px, 0), min(ux+1 , W - px + 1))
            ny = np.random.randint(max(vy - py, 0), min(uy+1 , H - py + 1))
            patch_coord = np.zeros((1, 4), dtype="int")
            patch_coord[0, 0] = nx
            patch_coord[0, 1] = ny
            patch_coord[0, 2] = nx + px
            patch_coord[0, 3] = ny + py

            index = self.compute_overlap(patch_coord, labels, 0.5)
            index = np.squeeze(index, axis=0)
            label = labels[index, :]
            if label.shape[0]==0:
                continue
            patch = img[ny: ny + py, nx: nx + px, :]

            label[:, 0] = np.maximum(label[:, 0] - patch_coord[0, 0], 0)
            label[:, 1] = np.maximum(label[:, 1] - patch_coord[0, 1], 0)
            label[:, 2] = np.minimum(self.patch_size[0] + label[:, 2] - patch_coord[0, 2], self.patch_size[0])
            label[:, 3] = np.minimum(self.patch_size[1] + label[:, 3] - patch_coord[0, 3], self.patch_size[1])
            spatch.append(patch)
            slabel.append(label)
            if len(spatch)>=img_num:
                break
        return spatch, slabel

    def compute_overlap(self, a, b, over_threshold=0.5):
        """
        Parameters
        ----------
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
        Returns
        -------
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        # ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
        ua = area

        ua = np.maximum(ua, np.finfo(float).eps)

        intersection = iw * ih

        overlap = intersection / ua
        index = overlap > over_threshold
        return index

    def load_data(self, data_path,batch_size):
        data = np.load(data_path)
        img, label = data['img'], data['label']
        label[:,[2,3]]+=label[:,[0,1]]
        img = img.astype(np.float32)
        label = label.astype(np.float32)
        return self.crop_patch(img, label,batch_size)
