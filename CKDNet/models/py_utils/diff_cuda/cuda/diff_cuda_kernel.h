#ifndef _BOTTOMPOOLING_CUDA_KERNEL
#define _BOTTOMPOOLING_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

void diff_forward_ongpu(float *x1, float *x2, int w, int h, int c, int batch, int *kernel, float *out);
void diff_backward_ongpu(float *x1, float *x2, float *x3, int w, int h, int c, int batch, float *backward_data, int *kernels,float *out);


#ifdef __cplusplus
}
#endif

#endif