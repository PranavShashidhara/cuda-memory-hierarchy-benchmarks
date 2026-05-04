#pragma once

#define TILE 32

// Memory kernels
__global__ void sequential(float *a, float *out, int N);
__global__ void strided(float *a, float *out, int stride, int N);
__global__ void random_access(float *a, float *out, int *idx, int N);

// Matmul kernels
__global__ void matmul_naive(float *A, float *B, float *C, int n);
__global__ void matmul_tiled(float *A, float *B, float *C, int n);