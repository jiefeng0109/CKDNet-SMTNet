import torch
import torch.nn as nn

from .py_utils import kp, AELoss, _neg_loss, convolution, residual,_offset_loss

def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)

nstack = 1
class model(kp):
    def __init__(self, db):
        n       = 4
        dims    = [256, 256, 384, 384, 512]
        modules = [2, 2, 2, 2, 4]
        out_dim = 1

        super(model, self).__init__(
            db, n, nstack, dims, modules, out_dim,
            # make_tl_layer=make_tl_layer,
            # make_br_layer=make_br_layer,
            # make_ct_layer=make_ct_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=128
        )

loss = AELoss(nstack = nstack,offsets_weight=1, sizes_weight=1, focal_loss=_neg_loss)
