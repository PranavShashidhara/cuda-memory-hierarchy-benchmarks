#pragma once
#include <cuda_fp16.h>

#define TILE 32

// ─────────────────────────────────────────────────────────────
//  Memory access kernels  (Phase 3)
// ─────────────────────────────────────────────────────────────
__global__ void sequential(float *a, float *out, int N);
__global__ void strided(float *a, float *out, int stride, int N);
__global__ void random_access(float *a, float *out, int *idx, int N);

// ─────────────────────────────────────────────────────────────
//  Matrix multiply kernels  (Phases 1 / 2 / 5)
// ─────────────────────────────────────────────────────────────
// 1. FP32 naive         — global memory only, no tiling
__global__ void matmul_naive(float *A, float *B, float *C, int n);

// 2. FP32 tiled         — shared-memory blocking (TILE×TILE)
__global__ void matmul_tiled(float *A, float *B, float *C, int n);

// 3. TC basic (WMMA)    — 64×64 block tile, 1 frag/warp, single buffer
__global__ void matmul_wmma(const __half *A, const __half *B, float *C, int n);


// ─────────────────────────────────────────────────────────────
//  Helper: in-kernel fp32 → fp16 conversion
// ─────────────────────────────────────────────────────────────
void convert_to_half(const float *d_fp32, __half *d_fp16, int n);