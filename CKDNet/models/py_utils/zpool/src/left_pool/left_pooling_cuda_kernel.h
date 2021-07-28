#ifndef _LEFTPOOLING_CUDA_KERNEL
#define _LEFTPOOLING_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

void left_pooling_forward_ongpu(float *x, int w, int h, int c, int batch, float *offset, float *forward_ind, float *out);
void left_pooling_backward_ongpu(float *x, int w, int h, int c, int batch, float *backward_ind, float *out);


#ifdef __cplusplus
}
#endif

#endif