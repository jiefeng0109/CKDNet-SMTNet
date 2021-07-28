import numpy as np


def iou_array(a1, a2):
    iou_a = np.array([]).reshape(-1, np.array(a1).shape[0])
    for d in a2:
        s = box_iou(a1, d).reshape(-1)
        iou_a = np.vstack([iou_a, s])
    return iou_a


def box_iou(b1, b2):
    '''Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    '''
    b2 = np.maximum(b2, 0)
    # Expand dim to apply broadcasting.
    b1 = np.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    # b1_wh_half = b1_wh/2.
    b1_mins = b1_xy
    b1_maxes = b1_xy + b1_wh

    # Expand dim to apply broadcasting.
    b2 = np.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    # b2_wh_half = b2_wh/2.
    b2_mins = b2_xy
    b2_maxes = b2_xy + b2_wh

    intersect_mins = np.maximum(b1_mins, b2_mins)
    intersect_maxes = np.minimum(b1_maxes, b2_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)


    return iou