
import os
import numpy as np
from glob import glob
import math
import torch
from config import system_configs
from torch.utils.data import Dataset
from .data_loader import CervicalDataset
from .detection import DETECTION
from .utils import random_crop, draw_gaussian, gaussian_radius
class REMOTE(Dataset):
    def __init__(self, db, split):
        super(REMOTE, self).__init__()
        data_dir   = system_configs.data_dir
        self.db = DETECTION(db)
        self.categories = self.db.configs["categories"]
        self.input_size = self.db.configs["input_size"]
        self.output_size = self.db.configs["output_sizes"][0]

        self.lighting = self.db.configs["lighting"]
        self.rand_crop = self.db.configs["rand_crop"]
        self.rand_color = self.db.configs["rand_color"]
        self.rand_scales = self.db.configs["rand_scales"]
        self.gaussian_bump = self.db.configs["gaussian_bump"]
        self.gaussian_iou = self.db.configs["gaussian_iou"][0]
        self.gaussian_rad = self.db.configs["gaussian_radius"]
        self._split = split
        self._dataset = {
            "trainval": "train",
            "minival": "test",
            "testdev": "test"
        }[self._split]
        self._coco_dir = os.path.join(data_dir)

        # self._label_dir  = os.path.join(self._coco_dir, self._dataset,'bbox/')


        self._image_dir  = os.path.join(self._coco_dir, self._dataset)
        self._image_file = os.path.join(self._image_dir, "{}")
        self._image_ids = glob(os.path.join(self._image_dir,'*.npz'))
        # self._seg_dir = os.path.join(self._coco_dir, self._dataset,'seg')
        # self._seg_file = os.path.join(self._seg_dir, "{}")

        self._data = "remote"

        self._mean = np.array([0.40789654, 0.40789654, 0.40789654,0.40789654, 0.40789654, 0.40789654,0.40789654, 0.40789654, 0.40789654], dtype=np.float32)
        self._std  = np.array([0.27408164, 0.27408164, 0.27408164,0.27408164, 0.27408164, 0.27408164,0.27408164, 0.27408164, 0.27408164], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

    def normalize_(self,image, mean, std):
        image -= mean
        image /= std

    def rot(self,width, detections):
        temp = detections.copy()
        detections[:, [0, 2]] = temp[:, [1, 3]]
        detections[:, [1, 3]] = width - temp[:, [2, 0]] - 1
        return detections

    def __getitem__(self, index):
        # images = np.zeros((9, self.input_size[0], self.input_size[1]), dtype=np.float32)
        tl_heatmaps = np.zeros((self.categories, self.output_size[0], self.output_size[1]), dtype=np.float32)
        br_heatmaps = np.zeros((self.categories, self.output_size[0], self.output_size[1]), dtype=np.float32)
        tl_offsets = np.zeros((2, self.output_size[0], self.output_size[1]), dtype=np.float32)
        br_offsets = np.zeros((2, self.output_size[0], self.output_size[1]), dtype=np.float32)
        tl_sizes = np.zeros((2, self.output_size[0], self.output_size[1]), dtype=np.float32)
        br_sizes = np.zeros((2, self.output_size[0], self.output_size[1]), dtype=np.float32)
        data = CervicalDataset(self._image_ids[index])
        image, labels = data.imgs,data.labels
        if np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            width = image.shape[1]
            labels[:, [0, 2]] = width - labels[:, [2, 0]] - 1

        if np.random.uniform() > 0.5:
            temp = image[:, :, 0:3]
            image[:, :, 0:3] = image[:, :, 6:9]
            image[:, :, 6:9] = temp

        rot_num = np.random.randint(4)
        image = np.rot90(image, rot_num)
        for _ in range(rot_num):
            labels = self.rot(image.shape[0], labels)

        image = image.astype(np.float32) / 255
        # if rand_color:
        #     color_jittering_(data_rng, image)
        #     if lighting:
        #         lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
        self.normalize_(image, self._mean, self._std)
        images= image.transpose((2, 0, 1))
        assert images.shape[0]==9,images.shape[1]==self.input_size[0]
        width_ratio = self.output_size[1] / self.input_size[1]
        height_ratio = self.output_size[0] / self.input_size[0]

        # flipping an image randomly

        for ind, detection in enumerate(labels):
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

            if self.gaussian_bump:
                width = detection[2] - detection[0]
                height = detection[3] - detection[1]

                width = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)

                if width * height < 9:
                    gaussian_rad = 0
                else:
                    gaussian_rad = -1

                if gaussian_rad == -1:
                    radius = gaussian_radius((height, width), self.gaussian_iou)
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
                draw_gaussian(tl_heatmaps[category], [xtl, ytl], radius, gaussian=gaussian)
                draw_gaussian(br_heatmaps[category], [xbr, ybr], radius, gaussian=gaussian)
                tl_sizes[:,ytl, xtl] = [fybr-fytl,fxbr-fxtl]
                br_sizes[:,ybr, xbr] = [fybr-fytl,fxbr-fxtl]
                tl_offsets[:,ytl, xtl] = [fytl-ytl,fxtl-xtl]
                br_offsets[:,ybr, xbr] = [fybr-ybr,fxbr-xbr]
        images = torch.from_numpy(images)
        tl_heatmaps = torch.from_numpy(tl_heatmaps)
        br_heatmaps = torch.from_numpy(br_heatmaps)
        tl_offsets = torch.from_numpy(tl_offsets)
        br_offsets = torch.from_numpy(br_offsets)
        tl_sizes = torch.from_numpy(tl_sizes)
        br_sizes = torch.from_numpy(br_sizes)

        return images,tl_heatmaps,br_heatmaps,tl_offsets,br_offsets,tl_sizes,br_sizes
    def __len__(self):
        return len(self._image_ids)