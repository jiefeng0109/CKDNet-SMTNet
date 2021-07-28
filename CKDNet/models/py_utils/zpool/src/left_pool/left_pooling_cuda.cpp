#include <THC/THC.h>
#include "left_pooling_cuda_kernel.h"

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

std::vector<torch::Tensor> pooling_forward(torch::Tensor x, torch::Tensor offset) {
  AT_CHECK(x.type().is_cuda(), "x must be a CUDA tensor");
  int batch   = x.size(0);
  int channel = x.size(1);
  int height  = x.size(2);
  int width   = x.size(3);

  auto output = torch::zeros_like(x);
  auto forward_ind = torch::zeros_like(x);

  left_pooling_forward_ongpu(
      x.data<float>(), width, height, channel, batch, offset.data<float>(), forward_ind.data<float>(), output.data<float>());
  return {output,forward_ind};
}
torch::Tensor pooling_backward(torch::Tensor x, torch::Tensor backward_ind) {
  AT_CHECK(x.type().is_cuda(), "x must be a CUDA tensor");
  int batch   = x.size(0);
  int channel = x.size(1);
  int height  = x.size(2);
  int width   = x.size(3);
  auto output = torch::zeros_like(x);
  left_pooling_backward_ongpu(
      x.data<float>(), width, height, channel, batch, backward_ind.data<float>(), output.data<float>());
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pooling_forward, "left_pooling_forward_cuda");
  m.def("backward", &pooling_backward, "left_pooling_backward_cuda");
}