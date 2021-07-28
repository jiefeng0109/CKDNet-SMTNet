import torch
# import  zddiff_cuda as zdiff_cuda
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
# class Diff_Function(Function):
#     def __index__(self):
#         pass
#     def forward(self,x1,x2):
#
#         assert x1.is_cuda and x2.is_cuda , 'input must be in cuda'
#
#         kernels = torch.from_numpy(np.array([2], dtype=np.int)).cuda()
#         output = zdiff_cuda.forward(x1,x2,kernels)
#         self.forward_ind = [x1,x2,output,kernels]
#         return output
#
#     def backward(self,grad_output):
#         assert grad_output.is_cuda
#
#         x1,x2,x3,kernels = self.forward_ind
#
#         grad_input = zdiff_cuda.backward(x1,x2,x3,grad_output,kernels)
#         del self.forward_ind
#         return grad_input,-grad_input

class Diff_cuda(torch.nn.Module):
    def __init__(self):
        super(Diff_cuda,self).__init__()
        kernel = torch.ones((3,3))
        kernel = torch.FloatTensor(kernel).expand(128, 1, 3, 3)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)
    def forward(self,x1,x2):
        x = torch.pow(x1-x2,2)
        weight = self.weight.cuda()
        x =F.conv2d(x,weight,stride=1, padding=1, groups=128)
        return torch.tanh(x)
        # return torch.tanh(Diff_Function()(x1,x2))