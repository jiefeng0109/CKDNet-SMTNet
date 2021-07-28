import os
import numpy as np
from config import system_configs
import cv2
class REMOTE(object):
    def __init__(self,split):
        super(REMOTE,self).__init__()
        self._split = split
        self._dataset = {
            "trainval": "train",
            "minival": "test",
            "testdev": "test"
        }[self._split]
        self._main_dir = os.path.join(system_configs.data_dir,self._dataset)
        # self._images_dir = os.path.join(self._main_dir,'s')
        #
        # self._images_file = os.path.join(self._images_dir,"{}.png")

        self._data = "remote"
        self._load_data()

        self._db_inds = np.arange(len(self._label))


    def _load_data(self):
        _pos = np.load(self._main_dir+'/pos.npz',allow_pickle=True)["arr_0"]
        _neg = np.load(self._main_dir+'/neg.npz',allow_pickle=True)["arr_0"]
        self._dict = list(_pos)
        self._dict += list(_neg)
        self._label = np.hstack([np.ones(len(_pos)),np.zeros(len(_neg))])
        a = 2
    def shuffle_inds(self, quiet=False):
        self._data_rng = np.random.RandomState(os.getpid())

        # if not quiet:
        #     print("shuffling indices...")
        rand_perm = self._data_rng.permutation(len(self._db_inds))

        self._db_inds = self._db_inds[rand_perm]

    @property
    def data(self):
        return self._data
    @property
    def db_inds(self):
        return self._db_inds


    def read_data(self,id):
        return self._dict[id],self._label[id].copy()

