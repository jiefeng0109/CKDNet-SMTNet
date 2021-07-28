import os
import numpy as np
from config import system_configs
class WPAFB(object):
    def __init__(self,split):
        super(WPAFB,self).__init__()
        self._split = split
        self._dataset = {
            "trainval": "train",
            "minival": "test",
            "testdev": "test"
        }[self._split]
        self._main_dir = os.path.join(system_configs.data_dir,self._dataset+'/')
        self._images_dir = os.path.join(self._main_dir,'images')

        self._images_file = os.path.join(self._images_dir,"{}")

        self._data = "wpafb"
        self._load_data()

        self._db_inds = np.arange(len(self._images_ids))

    def _load_data(self):

        self._images_ids = np.load(self._main_dir+'data.npy')
        self._label = np.load(self._main_dir+'label.npy')

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

    def image_ids(self, ind):
        return self._images_ids[ind]

    def label(self,name):
        if self._label[name]==1:
            temp = np.array((0,1),dtype=np.float32)
        else:
            temp = np.array((1, 0), dtype=np.float32)
        # return self._label[name].copy()
        return temp
    def image(self,name):
        # name = os.path.join(self._images_dir,name)

        return self._images_ids[name].copy()