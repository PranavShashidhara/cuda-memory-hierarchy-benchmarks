#pragma once
#include <climits>

#define TILE_DIM  32
#define WPT       4           // Work per thread (register blocking)
#define RTS       (TILE_DIM / WPT)   // = 8: threads per tile dim for v2

// ============================================================================
// MEMORY KERNELS
// ============================================================================
__global__ void sequential      (float *a, float *out, int N);
__global__ void strided         (float *a, float *out, int stride, int N);
__global__ void random_access   (float *a, float *out, int *idx, int N);
__global__ void sequential_shared(float *a, float *out, int N);

// ============================================================================
// MATMUL KERNELS
// ============================================================================
__global__ void matmul_naive    (float *A, float *B, float *C, int n);
__global__ void matmul_tiled    (float *A, float *B, float *C, int n);
__global__ void matmul_tiled_v2 (float *A, float *B, float *C, int n);
__global__ void matmul_tiled_v3 (float *A, float *B, float *C, int n);