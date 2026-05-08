// kernels/wmma_matmul_kernels.cu
// Tensor Core WMMA kernels — Jetson Orin Nano (Ampere sm_87)
//
// Two kernels exported:
//
//  matmul_wmma      — basic WMMA
//                       64×64 block tile, 1 fragment per warp, single buffer
//
// ─────────────────────────────────────────────────────────────────────────────

#include <mma.h>
#include <cuda_fp16.h>
#include "kernels.cuh"

using namespace nvcuda::wmma;

// ═══════════════════════════════════════════════════════════════════════════
//  WMMA fragment shape (fixed for Ampere fp16→fp32)
// ═══════════════════════════════════════════════════════════════════════════
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ═══════════════════════════════════════════════════════════════════════════
//  KERNEL 1  —  Basic WMMA
//  64×64 block tile, 4×4 warps (16 warps, 512 threads), 1 frag per warp
// ═══════════════════════════════════════════════════════════════════════════
#define BM_B 64
#define BN_B 64
#define BK_B 16

__global__ void matmul_wmma(const __half* __restrict__ A,
                             const __half* __restrict__ B,
                             float*        __restrict__ C,
                             int n)
{
    const int warpId  = threadIdx.x / 32;
    const int warpRow = warpId / (BN_B / WMMA_N);
    const int warpCol = warpId % (BN_B / WMMA_N);

    const int blockRow      = blockIdx.y * BM_B;
    const int blockCol      = blockIdx.x * BN_B;
    const int globalWarpRow = blockRow + warpRow * WMMA_M;
    const int globalWarpCol = blockCol + warpCol * WMMA_N;

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    fill_fragment(acc, 0.0f);

    __shared__ __half sA[BM_B][BK_B];
    __shared__ __half sB[BK_B][BN_B];

    for (int k = 0; k < n; k += BK_B) {
        for (int idx = threadIdx.x; idx < BM_B * BK_B; idx += blockDim.x) {
            int r = idx / BK_B, c = idx % BK_B;
            int gR = blockRow + r, gC = k + c;
            sA[r][c] = (gR < n && gC < n) ? A[gR * n + gC] : (__half)0.f;
        }
        for (int idx = threadIdx.x; idx < BK_B * BN_B; idx += blockDim.x) {
            int r = idx / BN_B, c = idx % BN_B;
            int gR = k + r, gC = blockCol + c;
            sB[r][c] = (gR < n && gC < n) ? B[gR * n + gC] : (__half)0.f;
        }
        __syncthreads();

        if (globalWarpRow < n && globalWarpCol < n) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
            load_matrix_sync(a_frag,
                             (const __half*)sA + warpRow * WMMA_M * BK_B,
                             BK_B);
            load_matrix_sync(b_frag,
                             (const __half*)sB + warpCol * WMMA_N,
                             BN_B);
            mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncthreads();
    }

    if (globalWarpRow < n && globalWarpCol < n)
        store_matrix_sync(C + globalWarpRow * n + globalWarpCol,
                          acc, n, mem_row_major);
}

// ═══════════════════════════════════════════════════════════════════════════
//  fp32 → fp16 conversion helper
// ═══════════════════════════════════════════════════════════════════════════
__global__ void float2half_kernel(const float* __restrict__ src,
                                   __half*       __restrict__ dst,
                                   int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __float2half(src[i]);
}

void convert_to_half(const float* d_fp32, __half* d_fp16, int n)
{
    int total   = n * n;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    float2half_kernel<<<blocks, threads>>>(d_fp32, d_fp16, total);
}
