import pdb
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import convolution, residual
from .utils import make_layer, make_layer_revr

from .kp_utils import _tranpose_and_gather_feat, _decode
from .kp_utils import _sigmoid, _ae_loss, _l1_loss, _neg_loss ,_offset_loss
from .kp_utils import make_tl_layer, make_br_layer, make_kp_layer, make_ct_layer
from .kp_utils import make_pool_layer, make_unpool_layer
from .kp_utils import make_merge_layer, make_inter_layer, make_cnv_layer
# from .kp_utils import Pooling_Offset,MergeTensor,Heatmaps,BD_layer
from .diff_cuda import Diff_cuda
from config import system_configs
frame = system_configs.frame
class ResNet_top(nn.Module):

    def __init__(self):
        super(ResNet_top, self).__init__()
        # self.conv = conv_bn_relu(3, 64, kernel_size=7, stride=2, padding=3,
        #         has_bn=True, has_relu=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pre = nn.Sequential(
            convolution(3,3, 64),
            convolution(3,64, 64),
            convolution(3,64, 128,2),
            convolution(3,128, 128),
        )
        self.diff = Diff_cuda()
        # self.diff2 = Diff_cuda()
        # self.diff3 = Diff_cuda()

        # self.conv_1 = nn.Sequential(
        #     conv_bn_relu(64*3, 128, kernel_size=3, stride=1, padding=1, has_bn=True, has_relu=True),
        #     conv_bn_relu(128, 64, kernel_size=3, stride=1, padding=1, has_bn=True, has_relu=True)
        # )
        self.conv_1 = convolution(3,128*frame, 128)
        self.conv_2 = convolution(3,256, 256)
        self.conv_3 = convolution(3,128, 128)

    def forward(self, x):
        x1 = self.pre(x[:,0:3])
        x3 = self.pre(x[:,6:9])
        x2 = self.pre(x[:,3:6])
        dx1 = self.diff(x1,x2)
        dx2 = self.diff(x2,x3)
        dx3 = self.diff(x1,x3)
        x1 = self.conv_1(torch.cat((dx3,dx2,dx1),dim=1))
        x2 = self.conv_3(x2)
        x = torch.cat((x2,x1),dim=1)
        x = self.conv_2(x)
        return x

# class kp_module(nn.Module):
#     def __init__(
#         self, n, dims, modules, layer=residual,
#         make_up_layer=make_layer, make_low_layer=make_layer,
#         make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
#         make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
#         make_merge_layer=make_merge_layer, **kwargs
#     ):
#         super(kp_module, self).__init__()
#         self.n   = n
#         self.up1 = nn.ModuleList([
#             make_up_layer(
#             3, curr_dim, curr_dim, curr_mod,
#             layer=layer, **kwargs
#         ) for curr_mod,curr_dim in zip(modules[:-1],dims[:-1])])
#         self.max1 = nn.ModuleList([
#             make_pool_layer(curr_dim) for curr_dim in dims[:-1]])
#         self.low1 = nn.ModuleList([
#             make_hg_layer(
#             3, curr_dim, next_dim, curr_mod,
#             layer=layer, **kwargs
#         )for curr_mod,curr_dim,next_dim in zip(modules[:-1],dims[:-1],dims[1:])])
#         self.low2 = make_low_layer(
#             3, dims[-1], dims[-1], modules[-1],
#             layer=layer, **kwargs
#         )
#         self.low3 = nn.ModuleList([
#             make_hg_layer_revr(
#             3, next_dim, curr_dim, curr_mod,
#             layer=layer, **kwargs
#         )for curr_mod,curr_dim,next_dim in zip(modules[:-1],dims[:-1],dims[1:])])
#         self.up2  = nn.ModuleList([make_unpool_layer(curr_dim)for curr_dim in dims[:-1]])
#
#         self.merge = nn.ModuleList([make_merge_layer(curr_dim)for curr_dim in dims[:-1]])
#
#     def forward(self,x):
#         up = []
#         low = [x]
#         out = []
#         for i in range(self.n):
#             up.append(self.up1[i](low[-1]))
#             max1 = self.max1[i](low[-1])
#             low.append(self.low1[i](max1))
#         out.append(self.low2(low[-1]))
#
#         for i in range(self.n):
#             i = self.n-1-i
#             low3 = self.low3[i](out[-1])
#             up2 = self.up2[i](low3)
#             out.append(self.merge[i](up[i],up2))
#
#         return out[-1]

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer,
            make_up_layer=make_up_layer,
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)

class kp(nn.Module):
    def __init__(
        self, db, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256,
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer, make_ct_layer=make_ct_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
        kp_layer=residual
    ):
        super(kp, self).__init__()

        self.nstack             = nstack
        self._decode            = _decode
        self._db                = db
        self.K                  = self._db.configs["top_k"]
        self.ae_threshold       = self._db.configs["ae_threshold"]
        self.kernel             = self._db.configs["nms_kernel"]
        self.input_size         = self._db.configs["input_size"][0]
        self.output_sizes        = self._db.configs["output_sizes"]
        self.train_mode         = self._db.configs["train_mode"]

        self.det_layer_num = len(self.output_sizes)
        self.out_dim = out_dim
        curr_dim = dims[0]

        self.kps = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.diff = True if system_configs.train_mode == 'd' else False
        if not self.diff:
            print('no diff')
            self.pre = nn.Sequential(
                convolution(3, 3, 128),
                residual(3, 128, 256, stride=2)
            ) if pre is None else pre
        else:
            print('diff')
            self.pre = ResNet_top()
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim,cnv_dim) for _ in range(nstack)
        ])
        self.offsets_cnvs = nn.ModuleList([
            make_cnv_layer(cnv_dim,4) for _ in range(nstack)
        ])
        self.sizes_cnvs = nn.ModuleList([
            make_cnv_layer(cnv_dim,4) for _ in range(nstack)
        ])
        self.heat_cnvs = nn.ModuleList([
            make_cnv_layer(cnv_dim,out_dim*2) for _ in range(nstack)
        ])
        self.merge_cnvs = nn.ModuleList([
            None for _ in range(nstack)
        ])

    def predict_module(self,inter,layers):
        outs = []

        return outs

    def _train(self, *xs,**kwargs):
        if not self.diff:
            image = xs[0][:, 3:6]
        else:
            image = xs[0]

        # inter      = self.pre(image)
        outs       = []
        inter = self.pre(image)

        layers = zip(
            self.heat_cnvs, self.offsets_cnvs,
            self.sizes_cnvs,self.cnvs,
            self.kps
        )
        for ind, layer in enumerate(layers):
            heat_cnv_, offsets_cnv_ = layer[0:2]
            sizes_cnv_, cnv_ = layer[2:4]
            kps_ = layer[4]
            kps = kps_(inter)
            cnv = cnv_(kps)

            # b, _, h, w = cnv.size()
            offsets = offsets_cnv_(cnv)

            sizes = sizes_cnv_(cnv)
            heat = heat_cnv_(cnv)

            outs += [heat,offsets,sizes]

        return outs

    def _test(self, *xs, **kwargs):
        if not self.diff:
            image = xs[0][:, 3:6]
        else:
            image = xs[0]

        inter = self.pre(image)

        detections = []
        centers = []
        layers = zip(
            self.heat_cnvs, self.offsets_cnvs,
            self.sizes_cnvs,self.cnvs,
            self.kps
        )
        for ind, layer in enumerate(layers):
            heat_cnv_, offsets_cnv_ = layer[0:2]
            sizes_cnv_, cnv_ = layer[2:4]
            kps_ = layer[4]
            kps = kps_(inter)
            cnv = cnv_(kps)

            # b, _, h, w = cnv.size()


            if ind == self.nstack - 1:
                b, _, h, w = cnv.size()
                offsets = offsets_cnv_(cnv)

                sizes = sizes_cnv_(cnv)
                heat = heat_cnv_(cnv)

                tl_offsets, br_offsets = offsets[:, :2], offsets[:, 2:]

                tl_heat, br_heat = heat[:, :self.out_dim], heat[:, self.out_dim:]

                outs = [tl_heat, br_heat,  tl_offsets, br_offsets, sizes]


        temp_det = self._decode(*outs, **kwargs)
        detections.append(temp_det)
            # centers.append(temp_cent)
        return detections



    def forward(self, *xs, **kwargs):

        if 'test' in kwargs and kwargs['test'] is True:
            return self._test(*xs, **kwargs)
        else:
            with torch.autograd.set_detect_anomaly(True):
                return self._train(*xs, **kwargs)

class AELoss(nn.Module):
    def __init__(self, nstack,offsets_weight=1, sizes_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()
        self.nstack = nstack
        self.sizes_weight = sizes_weight
        self.offsets_weight = offsets_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.l1_loss   = _l1_loss

    # def forward(self, outs, targets,**kwargs):
    #     target_stride = 8
    #     out_stride = 7*self.nstack
    #     focal_loss, pull_loss, push_loss, regr_loss, offset_loss =0,0,0,0,0
    #
    #     for ind in range(1):
    #         temp_target = targets[ind*target_stride:(ind+1)*target_stride]
    #         temp_out = outs[ind*out_stride:(ind+1)*out_stride]
    #         tfocal_loss, tpull_loss, tpush_loss, tregr_loss, toffset_loss, div =self.loss(temp_out,temp_target,**kwargs)
    #         focal_loss += tfocal_loss
    #         pull_loss += tpull_loss
    #         push_loss += tpush_loss
    #         regr_loss += tregr_loss
    #         focal_loss += tfocal_loss
    #         offset_loss += toffset_loss
    #     div *= 1
    #     loss = (focal_loss + pull_loss + push_loss + regr_loss + offset_loss) / div
    #     return loss.unsqueeze(0), (focal_loss / div).unsqueeze(0), (pull_loss / div).unsqueeze(0), (push_loss / div).unsqueeze(0), (regr_loss / div).unsqueeze(0), (offset_loss / div).unsqueeze(0)


    def forward(self,outs, targets, **kwargs):
        stride = 3

        heats = outs[0::stride]
        offsets = outs[1::stride]
        sizes = outs[2::stride]

        gt_heats = torch.cat([targets[0],targets[1]],dim=1)
        gt_offsets = torch.cat([targets[2],targets[3]],dim=1)
        gt_sizes = torch.cat([targets[4],targets[5]],dim=1)

        # focal loss
        heats = [_sigmoid(c) for c in heats]
        focal_loss = self.focal_loss(heats,gt_heats )

        # focal_loss += self.focal_loss(ct_heats, gt_ct_heat)
        # tl_mask = torch.cat([gt_tl_heat,gt_tl_heat],dim=1).detach()
        # br_mask = torch.cat([gt_br_heat,gt_br_heat],dim=1).detach()
        mask = ~gt_sizes.eq(0)
        offsets_loss = self.offsets_weight * self.l1_loss(offsets,gt_offsets,mask)

        sizes_loss = self.sizes_weight * self.l1_loss(sizes,gt_sizes,mask)


        loss = (focal_loss + offsets_loss + sizes_loss) / len(heats)
        # return focal_loss, pull_loss, push_loss, regr_loss, offset_loss,len(tl_heats)
        return loss, focal_loss,sizes_loss,offsets_loss,
        # return loss.unsqueeze(0), (focal_loss / len(tl_heats)).unsqueeze(0), (pull_loss / len(tl_heats)).unsqueeze(0), (push_loss / len(tl_heats)).unsqueeze(0), (regr_loss / len(tl_heats)).unsqueeze(0), (offset_loss / len(tl_heats)).unsqueeze(0)
