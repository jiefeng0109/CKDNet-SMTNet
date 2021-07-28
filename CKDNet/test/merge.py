import numpy as np
import collections
from external.nms import soft_nms, soft_nms_merge

def get_detection(det):
    # det = merge_position(data)
    data = collections.defaultdict(list)
    for key,val in det.items():
        temp = key.strip().split('/')[-1]
        temp = temp.strip().split('_')
        im_name,w,h = temp[0],int(temp[1]),int(temp[2][0:4])

        data[im_name].append([[float(d[0]) + h, float(d[1]) + w, float(d[2]) - float(d[0]), float(d[3]) - float(d[1]), float(d[4])] for d in val[1]][0])
    for key,val in data.items():
        data[key] = np.array(data[key],dtype=np.float32)
        soft_nms(data[key], Nt=0.5, method=2)

    return data
