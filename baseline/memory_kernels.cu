#include "kernels.cuh"

__global__ void sequential(float *a, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i];
}

__global__ void strided(float *a, float *out, int stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * stride;
    if (idx < N) out[i] = a[idx];
}

__global__ void random_access(float *a, float *out, int *idx, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[idx[i]];
}