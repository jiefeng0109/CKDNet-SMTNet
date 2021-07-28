import numpy as np
import torch
from config import system_configs

def kp_track(db, k_ind, data_aug=True, debug=False):
    batch_size = system_configs.batch_size
    images = np.zeros((batch_size, 2, 32, 32), dtype=np.float32)
    labels = np.zeros((batch_size, 2), dtype=np.float32)
    db_size = db.db_inds.size

    for b_ind in range(batch_size):
        db.shuffle_inds()
        db_ind = db.db_inds[k_ind]
        k_ind = (k_ind + 1) % db_size

        images[b_ind] = db.image_ids(db_ind).transpose((2, 0, 1))
        # images[b_ind] = db.image(image_file).transpose((2, 0, 1))
        labels[b_ind] = db.label(db_ind)
        a = 1
    images      = torch.from_numpy(images)
    labels = torch.from_numpy(labels)

    return {
        "xs": [images],
        "ys": [labels]
    },k_ind

def sample_data(db, k_ind, data_aug=True, debug=False):
    return globals()[system_configs.sampling_function](db, k_ind, data_aug, debug)