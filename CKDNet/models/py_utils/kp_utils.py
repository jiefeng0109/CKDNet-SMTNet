import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convolution_lstm import ConvLSTM
from .utils import convolution, residual


class Pooling_Offset(nn.Module):
    def __init__(self, dim):
        super(Pooling_Offset, self).__init__()
        self.p_conv1 = residual(3, dim, dim)
        self.p_conv2 = convolution(3, dim, 64)
        self.conv1 = nn.Conv2d(64, 16, (3, 3), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(16, 4, (3, 3), padding=(1, 1), bias=False)

        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):#top left bottom right
        out = self.p_conv1(x)
        out = self.p_conv2(out)
        out = self.conv1(out)
        out = self.conv2(out)

        return _sigmoid(out)

class Heatmaps(nn.Module):
    def __init__(self,cnv_dim, curr_dim, out_dim):
        super(Heatmaps,self).__init__()
        self.conv = nn.Sequential(
            convolution(3, cnv_dim, curr_dim, with_bn=False),
            nn.Conv2d(curr_dim, curr_dim, (3, 3), padding=(1, 1), bias=False),
            nn.Conv2d(curr_dim, out_dim, (1, 1))
        )

    def forward(self, x):
        x = self.conv(x)
        return _sigmoid(x)

class MergeTensor(nn.Module):
    def __init__(self,cnv_dim,out_dim):
        super(MergeTensor, self).__init__()
        self.conv_pre = nn.Sequential(
            nn.Conv2d(cnv_dim, cnv_dim, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(cnv_dim),
        )
        self.conv_up = nn.Sequential(
            nn.Conv2d(cnv_dim, cnv_dim, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(cnv_dim),
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = convolution(3,cnv_dim,out_dim)

    def forward(self, x,up_x):
        x = self.conv_pre(x)

        _,_,h,w = x.size()
        up_x = F.interpolate(up_x, size=(h,w), mode='bilinear',
                             align_corners=True)
        up_x = self.conv_up(up_x)
        out = self.relu(x + up_x)
        out = self.conv_out(out)

        return out

class BD_layer(nn.Module):
    def __init__(self,cnv_dim,size=128):
        super(BD_layer, self).__init__()
        self.conv_pre = convolution(3, cnv_dim, cnv_dim, with_bn=False)
        self.h_conv = nn.Conv2d(cnv_dim, 128, (9, 1), padding=(4, 0), bias=True)
        self.v_conv = nn.Conv2d(cnv_dim, 128, (1, 9), padding=(0, 4), bias=True)
        self.out = nn.Sequential(
            convolution(3, 256, 64, with_bn=False),
            nn.Conv2d(64, 4, (1, 1), padding=(0, 0), bias=True)
        )

    def forward(self, x):
        x = self.conv_pre(x)
        x1 = self.h_conv(x)
        x2 = self.v_conv(x)
        x = torch.cat((x1,x2),dim=1)
        x = self.out(x)
        return _sigmoid(x)

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2
def make_merge_layer(dim):
    return MergeUp()

def make_tl_layer(dim):
    return None

def make_br_layer(dim):
    return None

def make_ct_layer(dim):
    return None

def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_inter_layer(dim):
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    # return nn.Sequential(
    #     convolution(1, inp_dim, 128, with_bn=False),
    #     convolution(3, 128, 128, with_bn=False),
    #     convolution(3, 128, 64, with_bn=False),
    #     nn.Conv2d(64, out_dim, (1, 1))
    # )
    return nn.Sequential(
        convolution(3, inp_dim, inp_dim//2, with_bn=False),
        nn.Conv2d(inp_dim//2, out_dim, (1, 1))
    )
    # return convolution(3, inp_dim, out_dim)

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))

    feat = _gather_feat(feat, ind)
    return feat

def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds // (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds // width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def _decode(
    tl_heat, br_heat,  tl_regr, br_regr,  offset,
    K=100, kernel=1, ae_threshold=1, num_dets=1000,
    mms='gaussion',sigma=0.85,Nt=0.25,**kwargs
):
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)
    # ct_heat = torch.sigmoid(ct_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)
    # ct_heat = _nms(ct_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    # ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk(ct_heat, K=K)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)
    # ct_ys = ct_ys.view(batch, 1, K).expand(batch, K, K)
    # ct_xs = ct_xs.view(batch, 1, K).expand(batch, K, K)

    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)
        # ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)
        # ct_regr = ct_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]
        # ct_xs = ct_xs + ct_regr[..., 0]
        # ct_ys = ct_ys + ct_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    # tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    # tl_tag = tl_tag.view(batch, K, 1)
    # br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    # br_tag = br_tag.view(batch, 1, K)
    # dists  = torch.abs(tl_tag - br_tag)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores    = (tl_scores + br_scores) / 2

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances
    # dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    wh_inds  = ((br_xs < tl_xs)|(br_ys < tl_ys))
    # height_inds =

    #reject boxes based on offset
    # b,c,h,w = offset.size()
    tl_off = _tranpose_and_gather_feat(offset[:,0:2], tl_inds)
    tl_off = tl_off.view(batch, K, 1, 2)
    br_off = _tranpose_and_gather_feat(offset[:,2:4], br_inds)
    br_off = br_off.view(batch, 1, K, 2)


    ttl_xs = torch.clamp(tl_xs + tl_off[..., 1],min = 0,max = width)
    ttl_ys = torch.clamp(tl_ys + tl_off[..., 0],min = 0,max = width)
    tbr_xs = torch.clamp(br_xs - br_off[..., 1],min = 0,max = width)
    tbr_ys = torch.clamp(br_ys - br_off[..., 0],min = 0,max = width)
    tmin = lambda x,y:torch.where(x<y,x,y)
    tmax = lambda x,y:torch.where(x<y,y,x)
    siou = ((tmin(ttl_xs,br_xs)-tmax(tl_xs,tbr_xs))/(tmax(ttl_xs,br_xs)-tmin(tbr_xs,tl_xs)+1e-4))*\
           ((tmin(ttl_ys,br_ys)-tmax(tl_ys,tbr_ys))/(tmax(ttl_ys,br_ys)-tmin(tbr_ys,tl_ys)+1e-4))

    none_siou = ((ttl_xs<tbr_xs) |(ttl_ys<tbr_ys))


    max_siou = 1

    if mms=='hard':
        scores[siou<Nt]    = -1
    elif mms=='linear':
        sind = siou<Nt
        scores[sind] *= siou[sind]/max_siou
    elif mms=='gaussion':
        siou = 1-siou
        scores *= torch.exp(-(siou*siou)/sigma)
    else:
        raise TypeError('SIoU must be in {hard,linear,gaussion}')

    scores[cls_inds]    = -1
    scores[wh_inds]    = -1
    scores[none_siou] = -1
    # scores[dist_inds]   = -1
    # scores[width_inds]  = -1
    # scores[height_inds] = -1
    # scores[l_xs_inds] = -1
    # scores[l_ys_inds] = -1
    # scores[s_xs_inds] = -1
    # scores[s_ys_inds] = -1

    # rate = 0.4
    # ttl_xs = torch.clamp(tl_xs + tl_off[..., 1]*rate,min = 0,max = width)
    # ttl_ys = torch.clamp(tl_ys + tl_off[..., 0]*rate,min = 0,max = width)
    # tbr_xs = torch.clamp(br_xs - br_off[..., 1]*rate,min = 0,max = width)
    # tbr_ys = torch.clamp(br_ys - br_off[..., 0]*rate,min = 0,max = width)
    #
    # xs_inds = (ttl_xs>tbr_xs)
    # ys_inds = (ttl_ys>tbr_ys)
    # scores[xs_inds] = -1
    # scores[ys_inds] = -1
    # scores[xs_inds]  *= (tl_off_xs[xs_inds]+br_off_xs[xs_inds])/(br_xs[xs_inds] - tl_xs[xs_inds])
    # scores[ys_inds] *= (tl_off_ys[ys_inds]+br_off_ys[ys_inds])/(br_ys[ys_inds] - tl_ys[ys_inds])
    # scores[xs_inds]  *= 0.7*(tl_off_xs[xs_inds]+br_off_xs[xs_inds])/(br_xs[xs_inds] - tl_xs[xs_inds])
    # scores[ys_inds] *= 0.7*(tl_off_ys[ys_inds]+br_off_ys[ys_inds])/(br_ys[ys_inds] - tl_ys[ys_inds])

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)
    
    #width = (bboxes[:,:,2] - bboxes[:,:,0]).unsqueeze(2)
    #height = (bboxes[:,:,2] - bboxes[:,:,0]).unsqueeze(2)
    
    clses  = tl_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    # ct_xs = ct_xs[:,0,:]
    # ct_ys = ct_ys[:,0,:]
    #
    # center = torch.cat([ct_xs.unsqueeze(2), ct_ys.unsqueeze(2), ct_clses.float().unsqueeze(2), ct_scores.unsqueeze(2)], dim=2)
    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections
    # return detections, center

def _neg_loss(preds, gt):

    loss = 0
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
    for ind,pred in enumerate(preds):


        neg_weights = torch.pow(1 - gt[neg_inds], 4)

        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _offset_loss(preds, gt):

    loss = 0
    neg_inds = gt.eq(0)
    pos_inds = gt.gt(0)
    for ind,pred in enumerate(preds):
        if pred is None:
            loss = torch.zeros(1).cuda()
            continue

        # gt_temp = gt[:,ind][pos_inds]*128
        # pos_temp = pred[pos_inds]*128
        # neg_temp = pred[neg_inds]
        pos_loss = torch.pow(pred[pos_inds]-gt[pos_inds], 2)
        neg_loss = torch.pow(pred[neg_inds], 2)

        num_pos  = pos_inds.float().sum()
        num_neg  = neg_inds.float().sum()
        pos_loss = pos_loss.sum() / (num_pos + 1e-4)
        neg_loss = neg_loss.sum() / (num_neg + 1e-4)

        loss += (3*pos_loss + neg_loss)
    return loss

def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return x

def _ae_loss(tag0, tag1, mask):
    num  = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    temp0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    mask_ind = mask.eq(1)
    temp0 = temp0[mask_ind].sum()
    temp1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    temp1 = temp1[mask_ind].sum()
    pull = temp0 + temp1


    if len(tag_mean.shape)>1:
        mask = mask.unsqueeze(1) + mask.unsqueeze(2)
        mask = mask.eq(2)
        num = num.unsqueeze(2)
        num2 = (num - 1) * num
        dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    else:
        mask = mask.unsqueeze(1) + mask.unsqueeze(2)
        mask = mask.eq(2)
        num = num.unsqueeze(1)
        num2 = (num - 1) * num
        dist = tag_mean.unsqueeze(0) - tag_mean.unsqueeze(1)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)

    dist = dist[mask]
    push = dist.sum()
    return pull, push


def _regr_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)
    mask = mask.eq(1)
    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _l1_loss(pres,gt,mask):
    loss  = []
    for pre in pres:
        # mask_ind = ~gt.eq(0)
        loss.append(F.l1_loss(pre[mask],gt[mask]))
    return sum(loss)/len(loss)