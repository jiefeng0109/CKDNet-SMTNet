#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "diff_cuda_kernel.h"

#define BLOCK 512

dim3 cuda_gridsize(int n)
{
    int k = (n-1) / BLOCK + 1;
    int x = k;
    int y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d(x, y, 1);
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

__global__ void diff_forward_kernel(int N,  float const *x1,  float const *x2, int w, int h, int c, int batch, int *kernels, float *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i/w;
    int in_h = i%h;
    i = i/h;
    int in_c = i%c;
    i = i/c;
    int b = i%batch;

    int kernel =kernels[0];

    int start_h = in_h-kernel;
    int end_h = in_h+kernel;
    if(start_h < 0) start_h = 0;
    if(end_h >= h) end_h = h;

    int start_w = in_w-kernel;
    int end_w = in_w+kernel;
    if(start_w < 0) start_w = 0;
    if(end_w >= w) end_w = w;


    int now_ind = 0;
    float now_ed = 0;
    for(int ind_h = start_h ;ind_h<end_h;ind_h++)
        for(int ind_w = start_w ;ind_w<end_w;ind_w++)
        {
            now_ind = ind_w + w*(ind_h + h*(in_c + c*b));
            now_ed += pow((x1[now_ind]-x2[now_ind]),2);
        }

    out[in_index] = sqrt(now_ed);
}

__global__ void diff_backward_kernel(int N,  float const *x1,  float const *x2,   float const *x3, int w, int h, int c, int batch, float *backward_data, int *kernels,  float *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i/w;
    int in_h = i%h;
    i = i/h;
    int in_c = i%c;
    i = i/c;
    int b = i%batch;

    if(x3[i]<0.0000001) return;

    int kernel =kernels[0];

    int start_h = in_h-kernel;
    int end_h = in_h+kernel;
    if(start_h < 0) start_h = 0;
    if(end_h >= h) end_h = h;

    int start_w = in_w-kernel;
    int end_w = in_w+kernel;
    if(start_w < 0) start_w = 0;
    if(end_w >= w) end_w = w;


    int now_ind = 0;
    float now_backward = 0;
    for(int ind_h = start_h ;ind_h<end_h;ind_h++)
        for(int ind_w = start_w ;ind_w<end_w;ind_w++)
        {
            now_ind = ind_w + w*(ind_h + h*(in_c + c*b));
            now_backward = 2*backward_data[i]*(x1[now_ind]-x2[now_ind])/x3[i];
            atomicAdd(&out[now_ind],now_backward);
        }


}

void diff_forward_ongpu(float *x1, float *x2, int w, int h, int c, int batch, int *kernels, float *out)
{
    int size = w*h*c*batch;
    cudaError_t err;
    diff_forward_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x1, x2, w, h, c, batch, kernels, out);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}

void diff_backward_ongpu(float *x1, float *x2, float *x3, int w, int h, int c, int batch, float *backward_data, int *kernels,float *out)
{
    int size = w*h*c*batch;
    cudaError_t err;
    diff_backward_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x1, x2, x3, w, h, c, batch, backward_data, kernels, out);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}


#ifdef __cplusplus
}
#endif
