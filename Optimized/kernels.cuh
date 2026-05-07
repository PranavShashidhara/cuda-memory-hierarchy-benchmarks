#pragma once
// Optimized/kernels.cuh
// Declares all kernels used in the Optimized build.
// Adds Tensor Core (WMMA) declarations on top of the fp32 set.

#include <climits>
#include <functional>
#include <cuda_fp16.h>

// ─── Tile / blocking geometry (shared by fp32 matmul kernels) ─────────────────
#define TILE_DIM  32
#define WPT       4
#define RTS       (TILE_DIM / WPT)   // 8

// ─── Memory kernels ───────────────────────────────────────────────────────────
__global__ void sequential       (float *a, float *out, int N);
__global__ void strided          (float *a, float *out, int stride, int N);
__global__ void random_access    (float *a, float *out, int *idx, int N);
__global__ void sequential_shared(float *a, float *out, int N);

// ─── fp32 matmul kernels ──────────────────────────────────────────────────────
__global__ void matmul_naive    (float *A, float *B, float *C, int n);
__global__ void matmul_tiled    (float *A, float *B, float *C, int n);
__global__ void matmul_tiled_v2 (float *A, float *B, float *C, int n);
__global__ void matmul_tiled_v3 (float *A, float *B, float *C, int n);

// ─── Tensor Core (WMMA fp16→fp32) kernels ────────────────────────────────────
//
//  matmul_wmma      — basic WMMA, 64×64 block tile, single-buffer
//  matmul_wmma_opt  — optimised WMMA:
//                       128×128 block tile
//                       2×4 warp tile  (8 MMA frags/warp)
//                       double-buffered smem (hides DRAM latency)
//                       +8-half padding per row (eliminates bank conflicts)
//                       #pragma unroll throughout
__global__ void matmul_wmma    (const __half * __restrict__ A,
                                 const __half * __restrict__ B,
                                 float        * __restrict__ C,
                                 int n);

__global__ void matmul_wmma_opt(const __half * __restrict__ A,
                                 const __half * __restrict__ B,
                                 float        * __restrict__ C,
                                 int n);

// Host-callable fp32 → fp16 bulk conversion (launches a device kernel)
void convert_to_half(const float *d_fp32, __half *d_fp16, int n);
