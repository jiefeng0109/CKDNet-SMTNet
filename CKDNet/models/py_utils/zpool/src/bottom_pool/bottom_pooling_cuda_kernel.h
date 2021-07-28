#ifndef _BOTTOMPOOLING_CUDA_KERNEL
#define _BOTTOMPOOLING_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

void bottom_pooling_forward_ongpu(float *x, int w, int h, int c, int batch, float *offset, float *forward_ind, float *out);
void bottom_pooling_backward_ongpu(float *x, int w, int h, int c, int batch, float *backward_ind, float *out);


#ifdef __cplusplus
}
#endif

#endif