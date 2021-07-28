import numpy as np
import torch

from torch.autograd import Function
import sys
import  os
sys.path.append(os.path.join(os.path.dirname(__file__),'dist/zpool-0.0.0-py3.6-linux-x86_64.egg'))
import top_pooling,bottom_pooling,right_pooling,left_pooling

class TPFunction(Function):
    def __init__(self,pool):
        self.pool = pool

    def forward(self,x,offset):

        bsize, c, h, w = x.size()
        offset = offset.view(bsize, 1, h, w)
        # offset = torch.ones_like(offset)*w
        if x.is_cuda:
            output = self.pool.forward(x, offset)
        else:
            raise ImportError('Require cuda data')
        self.forward_ind = output[1]
        return output[0]

    def backward(self, grad_output):
        bsize, c, h, w = grad_output.size()

        backward_ind = self.forward_ind
        assert (grad_output.is_cuda)

        grad_input = self.pool.backward(grad_output, backward_ind)

        return grad_input,None


class TPLayer(torch.nn.Module):
    def __init__(self):
        super(TPLayer,self).__init__()
    def forward(self, x,offset):
        # return self.test.forward(x,self.w)
        return TPFunction(top_pooling)(x,offset)
    # def backward(self,grad_out):
    #     return self.test.backward(grad_out)

class BPLayer(torch.nn.Module):
    def __init__(self):
        super(BPLayer,self).__init__()

    def forward(self, x,offset):
        return TPFunction(bottom_pooling)(x,offset)

class RPLayer(torch.nn.Module):
    def __init__(self):
        super(RPLayer,self).__init__()

    def forward(self, x,offset):
        return TPFunction(right_pooling)(x,offset)

class LPLayer(torch.nn.Module):
    def __init__(self):
        super(LPLayer,self).__init__()

    def forward(self, x,offset):
        return TPFunction(left_pooling)(x,offset)
