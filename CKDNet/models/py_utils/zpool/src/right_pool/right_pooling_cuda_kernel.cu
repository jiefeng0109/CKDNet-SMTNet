#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "right_pooling_cuda_kernel.h"

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

__global__ void right_pooling_forward_kernel(int N,  float const *x, int w, int h, int c, int batch, float *offset, float *forward_ind, float *out)
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

    int sp = in_w + w*(in_h + h*b);
    sp = in_w - int(offset[sp]);
    if(sp < 0)
        sp = 0;

    int max_ind = in_index;

    int out_index = 0;
    for(int ind=in_w ; ind >= sp ; --ind)
    {
       out_index = ind + w*(in_h + h*(in_c + c*b));

       if(x[max_ind] < x[out_index])
            max_ind = out_index;
    }

    out[in_index] = x[max_ind];
    forward_ind[in_index] = max_ind;
}

__global__ void right_pooling_backward_kernel(int N,  float const *x, int w, int h, int c, int batch, float *backward_ind, float *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int max_ind = backward_ind[i];
    atomicAdd(&out[max_ind],x[i]);
}

void right_pooling_forward_ongpu(float *x, int w, int h, int c, int batch, float *offset, float *forward_ind, float *out)
{
    int size = w*h*c*batch;
    cudaError_t err;
    right_pooling_forward_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x, w, h, c, batch, offset, forward_ind, out);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}

void right_pooling_backward_ongpu(float *x, int w, int h, int c, int batch, float *backward_ind, float *out)
{
    int size = w*h*c*batch;
    cudaError_t err;
    right_pooling_backward_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x, w, h, c, batch, backward_ind, out);

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
