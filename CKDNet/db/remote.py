import sys
sys.path.insert(0, "data/coco/PythonAPI/")
import cv2
import scipy.io as scio
import os
import json
import numpy as np
import pickle
from glob import glob
from tqdm import tqdm
from db.detection import DETECTION
from config import system_configs

class REMOTE(DETECTION):
    def __init__(self, db_config, split):
        super(REMOTE, self).__init__(db_config)
        data_dir   = system_configs.data_dir
        cache_dir  = system_configs.cache_dir
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

        # self._cache_file = os.path.join(cache_dir, self._data+"_{}.pkl".format(self._dataset))
        #
        # self._load_data()
        self._db_inds = np.arange(len(self._image_ids))


    # def _load_data(self):
    #     if not os.path.exists(self._cache_file):
    #         print("No cache file found...")
    #         self._extract_data()
    #         with open(self._cache_file, "wb") as f:
    #             pickle.dump([self._detections, self._image_ids], f)
    #     else:
    #         with open(self._cache_file, "rb") as f:
    #             self._detections, self._image_ids = pickle.load(f)
    #
    #
    #
    # def _extract_data(self):
    #
    #     image_ids = os.listdir(self._image_dir)
    #     self._detections = {}
    #     self._image_ids = []
    #     for ind, image_id in enumerate(tqdm(image_ids)):
    #         # image      = self._coco.loadImgs(coco_image_id)[0]
    #         bboxes     = []
    #         categories = []
    #         label_name = image_id.strip().split('.npy')[0]
    #         self._image_ids.append(label_name)
    #         annotations = np.load(self._label_dir+label_name+'.npy')
    #
    #         for annotation in annotations:
    #             bbox = np.array(annotation)
    #             bbox[[2, 3]] += bbox[[0, 1]]
    #             bboxes.append(bbox)
    #
    #             categories.append('1')
    #
    #         bboxes     = np.array(bboxes, dtype=float)
    #         categories = np.array(categories, dtype=float)
    #         if bboxes.size == 0 or categories.size == 0:
    #             self._detections[label_name] = np.zeros((0, 5), dtype=np.float32)
    #         else:
    #             self._detections[label_name] = np.hstack((bboxes, categories[:, None]))
    #
    # def detections(self, name):
    #
    #     detections = self._detections[name]
    #
    #     return detections.astype(float).copy()
    #
    # def image_array(self,name):
    #
    #     path = self._image_file.format(name+'.npy')
    #
    #     return np.load(path)
    #
    # def seg_array(self,name):
    #     path = self._seg_file.format(name+'.npy')
    #
    #     return np.load(path)[:,:,1]
    #
    # def _to_float(self, x):
    #     return float("{:.2f}".format(x))


