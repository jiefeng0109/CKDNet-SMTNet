#include <THC/THC.h>
#include "diff_cuda_kernel.h"

#include <torch/extension.h>
extern THCState *state;

//int top_pooling_forward_cuda(THCudaTensor *x_tensor, int w, int h, int c, int batch, THCudaTensor *stride_tensor, THCudaTensor *forward_tensor, THCudaTensor *out_tensor)
//{
//    float * x = THCudaTensor_data(state, x_tensor);
//    float * out = THCudaTensor_data(state, out_tensor);
//    float * stride = THCudaTensor_data(state, stride_tensor);
//    float * forward_ind = THCudaTensor_data(state, forward_tensor);
//
//    cudaStream_t stream = THCState_getCurrentStream(state);
//    top_pooling_forward_ongpu(x, w, h, c, batch, stride, forward_ind, out, stream);
//
//    return 1;
//}
//
//int top_pooling_backward_cuda(THCudaTensor *x_tensor, int w, int h, int c, int batch, THCudaTensor *backward_tensor, THCudaTensor *out_tensor)
//{
//    float * x = THCudaTensor_data(state, x_tensor);
//    float * out = THCudaTensor_data(state, out_tensor);
//    float * backward_ind = THCudaTensor_data(state, backward_tensor);
//
//    cudaStream_t stream = THCState_getCurrentStream(state);
//    top_pooling_backward_ongpu(x, w, h, c, batch, backward_ind, out, stream);
//
//    return 1;
//}

torch::Tensor diff_forward(torch::Tensor x1, torch::Tensor x2, torch::Tensor kernels) {
  AT_CHECK(x1.type().is_cuda(), "x1 must be a CUDA tensor");
  AT_CHECK(x2.type().is_cuda(), "x2 must be a CUDA tensor");
  int batch   = x1.size(0);
  int channel = x1.size(1);
  int height  = x1.size(2);
  int width   = x1.size(3);

  auto output = torch::zeros_like(x1);

  diff_forward_ongpu(
      x1.data<float>(), x2.data<float>(), width, height, channel, batch, kernels.data<int>(), output.data<float>());

  return output;
}
torch::Tensor diff_backward(torch::Tensor x1, torch::Tensor x2, torch::Tensor x3, torch::Tensor backward_data, torch::Tensor kernels) {
  AT_CHECK(x1.type().is_cuda(), "x must be a CUDA tensor");
  int batch   = x1.size(0);
  int channel = x1.size(1);
  int height  = x1.size(2);
  int width   = x1.size(3);

  auto output = torch::zeros_like(x1);

  diff_backward_ongpu(
      x1.data<float>(), x2.data<float>(), x3.data<float>(), width, height, channel, batch, backward_data.data<float>(), kernels.data<int>(), output.data<float>());
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &diff_forward, "diff_forward_cuda");
  m.def("backward", &diff_backward, "diff_backward_cuda");
}