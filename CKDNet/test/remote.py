import os
import cv2
import time
import json
import copy
import numpy as np
import torch
import threading
import copy
import time
import traceback
import scipy.io as scio
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import multiprocessing
import threading
import queue
from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_
from torch.multiprocessing import Queue
# from external.nms import soft_nms, soft_nms_merge
from external import soft_nms, soft_nms_merge
from test.eval import cal_map

colours = np.random.rand(80, 3)
height, width = 256, 256

new_height = int(height)
new_width = int(width)
new_center = np.array([new_height // 2, new_width // 2])

inp_height = new_height
inp_width = new_width


def _rescale_dets(detections, ratios, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs /= ratios[:, 1][:, None, None]
    ys /= ratios[:, 0][:, None, None]
    # xs    -= borders[:, 2][:, None, None]
    # ys    -= borders[:, 0][:, None, None]
    tx_inds = xs[:, :, 0] <= -5
    bx_inds = xs[:, :, 1] >= sizes[0, 1] + 5
    ty_inds = ys[:, :, 0] <= -5
    by_inds = ys[:, :, 1] >= sizes[0, 0] + 5

    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)
    detections[:, tx_inds[0, :], 4] = -1
    detections[:, bx_inds[0, :], 4] = -1
    detections[:, ty_inds[0, :], 4] = -1
    detections[:, by_inds[0, :], 4] = -1


def save_image(data, fn):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data)
    plt.savefig(fn, dpi=height)
    plt.close()


class my_threads(threading.Thread):
    def __init__(self, db, queue, imgs, filename, position):
        threading.Thread.__init__(self)
        self.db = db
        self.queue = queue
        self.filename = filename
        self.imgs = imgs
        self.pinds = len(position)
        self.batch_size = system_configs.batch_size
        self.position = position

    def run(self):
        for pind in range(0, self.pinds, self.batch_size):
            batch_size_num = self.pinds - pind if self.pinds - pind < self.batch_size else self.batch_size
            images = np.zeros((batch_size_num, 9, new_width, new_height), dtype=np.float32)

            image_files = []
            for b_ind in range(batch_size_num):
                x, y = self.position[pind + b_ind]
                image = self.imgs[y:y + inp_height, x:x + inp_height]

                image_files.append([x, y])
                resized_image = cv2.resize(image, (new_width, new_height)) if inp_height != new_height else image
                # resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

                resized_image = resized_image / 255.
                normalize_(resized_image, self.db.mean, self.db.std)
                images[b_ind] = resized_image.transpose((2, 0, 1))
            # images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            images = torch.from_numpy(images).pin_memory()
            try:
                data = [images, image_files, self.filename]
                self.queue.put(data)
            except Exception as e:
                traceback.print_exc()
                raise e


def calc_split_num(image_shape, patch_size=[256, 256], strides=[220, 220]):
    # strides = [cfg.patch_size[0]//2, cfg.patch_size[1]//2]
    x_num = (image_shape[1] - patch_size[0]) // strides[0] + 2
    y_num = (image_shape[0] - patch_size[1]) // strides[1] + 2
    position = []
    for i in range(0, x_num):
        for j in range(0, y_num):
            x = strides[0] * i if i < x_num - 1 else image_shape[1] - patch_size[0]
            y = strides[1] * j if j < y_num - 1 else image_shape[0] - patch_size[1]
            position.append([x, y])
    return position


def kp_decode(nnet, images, K, ae_threshold=0.5, kernel=3, test=True, **kwargs):
    detections = nnet.test([images], ae_threshold=ae_threshold, K=K, kernel=kernel, test=test, **kwargs)
    detections = [det.data.cpu().numpy() for det in detections]
    # center = [cent.data.cpu().numpy() for cent in center ]
    return detections


def kp_detection(db, nnet, result_dir, debug=False, decode_func=kp_decode, **kwargs):
    batch_size = system_configs.batch_size
    # batch_size = 2

    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    if db.split != "trainval":
        db_inds = db.db_inds[:100] if debug else db.db_inds
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds[:5000]
    db_inds = db.db_inds
    num_images = db_inds.size

    K = db.configs["top_k"]
    ae_threshold = db.configs["ae_threshold"]
    nms_kernel = db.configs["nms_kernel"]

    scales = db.configs["test_scales"]
    output_sizes = db.configs["output_sizes"]
    weight_exp = db.configs["weight_exp"]
    merge_bbox = db.configs["merge_bbox"]
    categories = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1,
        "exp_soft_nms": 2
    }[db.configs["nms_algorithm"]]

    ratios = np.zeros((1, 2), dtype=np.float32)
    sizes = np.zeros((1, 2), dtype=np.float32)
    out_height, out_width = output_sizes[0]
    height_ratio = out_height / inp_height
    width_ratio = out_width / inp_width

    sizes[0] = [int(height), int(width)]
    ratios[0] = [height_ratio, width_ratio]
    det_bboxs = {}
    gt_bboxs = {}
    top_bboxes = {}
    for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
        data_path = db.image_file(ind)
        filename = data_path.split('/')[-1].split('.')[0]
        data = np.load(data_path)
        imgs, labels = data['img'], data['label']
        position = calc_split_num(imgs.shape[0:2])
        test_queue = Queue(2)
        read_thread = my_threads(db, test_queue, imgs.copy(), filename, position)
        read_thread.start()
        plen = len(position)
        queue_ind = 0
        all_bboxes = []
        while queue_ind < plen:
            # start1 = time.clock()
            data = test_queue.get(block=True)
            images, image_files, filename_queue = data
            batch_size_num = images.size()[0]
            output_dets = decode_func(nnet, images, K, ae_threshold=ae_threshold, kernel=nms_kernel, **kwargs)
            for b_ind in range(batch_size_num):
                queue_ind += 1
                if filename != filename_queue:
                    raise EOFError
                output_det = [det[b_ind::batch_size_num] for det in output_dets]
                dets = output_det[0].reshape(1, -1, 8)
                _rescale_dets(dets, ratios, sizes)
                detections = dets[0]

                box_width = detections[:, 2] - detections[:, 0]
                box_height = detections[:, 3] - detections[:, 1]
                ignor_ind = (box_width * box_height) < 25
                detections[ignor_ind, -1] = -1
                detections = detections[np.argsort(-detections[:, 4])]
                classes = detections[..., -1]

                keep_inds = (detections[:, 4] > -1)
                detections = detections[keep_inds]
                classes = classes[keep_inds]

                image_id = str(image_files[b_ind][0]) + '-' + str(image_files[b_ind][1])
                top_bboxes[image_id] = {}
                for j in range(categories):
                    keep_inds = (classes == j)
                    top_bboxes[image_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
                    if merge_bbox:
                        soft_nms_merge(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm,
                                       weight_exp=weight_exp)
                    else:
                        soft_nms(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm)
                    top_bboxes[image_id][j + 1] = top_bboxes[image_id][j + 1][:, 0:5]

                scores = np.hstack([
                    top_bboxes[image_id][j][:, -1]
                    for j in range(1, categories + 1)
                ])
                if len(scores) > max_per_image:
                    kth = len(scores) - max_per_image
                    thresh = np.partition(scores, kth)[kth]
                    for j in range(1, categories + 1):
                        keep_inds = (top_bboxes[image_id][j][:, -1] >= thresh)
                        top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]
                dets = top_bboxes[image_id][1]
                dets[:, 0:4:2] += image_files[b_ind][0]
                dets[:, 1:4:2] += image_files[b_ind][1]
                all_bboxes += [[int(d[0]), int(d[1]), int(d[2]), int(d[3]),
                                round(float(d[4]), 4)] for d in dets]

        all_bboxes = np.array(all_bboxes, dtype=np.float32)
        soft_nms(all_bboxes, sigma=0.3, method=2)
        keep_inds = (all_bboxes[:, -1] >= 0.3)
        det_bboxs[filename] = all_bboxes[keep_inds]

        labels[:, [2, 3]] += labels[:, [0, 1]]
        gt_bboxs[filename] = labels

        if debug:

            im = imgs[:, :, 3:6].copy()
            keep_inds = (all_bboxes[:, -1] >= 0.3)
            color = [[0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 255, 255]]
            tlabels = labels.copy()
            tlabels[:, 2:4] -= tlabels[:, 0:2]
            tbboxes = all_bboxes[keep_inds].copy()
            tbboxes[:, 2:4] -= tbboxes[:, 0:2]
            # for bbox in all_bboxes[keep_inds]:
            #     s = bbox[-1]
            #     bbox = bbox[0:4].astype(np.int32)
            #     xmin = bbox[0]
            #     ymin = bbox[1]
            #     xmax = bbox[2]
            #     ymax = bbox[3]
            #     from test.iou_array import box_iou
            #     bbox[2:4] -= bbox[0:2]
            #     iou = np.max(box_iou(tlabels, bbox))
            #     # print(iou)
            #     c = color[2] if iou > 0.3 else color[0]
            #     cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
            #     # cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color[2], 1)
            #     # cv2.putText(im, '%.2f'%s, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color[0], 1)
            # for bbox in labels:
            #     bbox = bbox[0:4].astype(np.int32)
            #     xmin = bbox[0]
            #     ymin = bbox[1]
            #     xmax = bbox[2]
            #     ymax = bbox[3]
            #     from test.iou_array import box_iou
            #     bbox[2:4] -= bbox[0:2]
            #     iou = np.max(box_iou(tbboxes, bbox))
            #     # print(iou)
            #     if iou > 0.3:
            #         continue
            #     c = color[3]
            #     # c = color[1] if iou > 0.3 else color[3]
            #     cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
            for bbox in all_bboxes[keep_inds]:
                s = bbox[-1]
                bbox = bbox[0:4].astype(np.int32)
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[2]
                ymax = bbox[3]
                from test.iou_array import box_iou
                bbox[2:4] -= bbox[0:2]
                iou = box_iou(tlabels, bbox)
                max_index = np.argmax(iou)
                if iou[max_index] > 0.3 and tlabels[max_index, 4] != -1:
                    c = color[2]
                    tlabels[max_index, 4] = -1
                else:
                    c = color[0]
                cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
            tlabels[:, 2:4] += tlabels[:, 0:2]
            for bbox in tlabels:
                s = bbox[4]
                bbox = bbox[0:4].astype(np.int32)
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[2]
                ymax = bbox[3]
                # from test.iou_array import box_iou
                # bbox[2:4] -= bbox[0:2]
                # iou = np.max(box_iou(tbboxes, bbox))
                # # print(iou)
                if s == -1:
                    continue
                cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color[3], 1)

            debug_file2 = os.path.join(debug_dir, "{}.png".format(filename))
            # im = cv2.putText(im, 'frame {}'.format(filename.split('-')[-1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imwrite(debug_file2, im)
            # plt.close()
            # cv2.imwrite(debug_file, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    np.save('tr_jilin_det.npy', det_bboxs)
    cal_map(det_bboxs, gt_bboxs)
    # result_json = os.path.join(result_dir, "results.json")
    # detections  = db.convert_to_coco(top_bboxes)
    # with open(result_json, "w") as f:
    #     json.dump(detections, f)
    #
    # cls_ids   = list(range(1, categories + 1))
    # image_ids = [db.image_ids(ind) for ind in db_inds]
    # db.evaluate(result_json, cls_ids, image_ids)
    return 0


#
def testing(db, nnet, result_dir, debug=False,**kwargs):
    return globals()[system_configs.sampling_function](db, nnet, result_dir, debug=debug,**kwargs)