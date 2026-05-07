// Optimized/wmma_matmul_kernels.cu
// Tensor Core WMMA kernels — Jetson Orin Nano (Ampere sm_87, fp16 TC)
//
// Kernel 1: matmul_wmma  (basic)
//   • 64×64 block tile, 4×4 warps (16 warps / 512 threads)
//   • One 16×16 WMMA fragment per warp
//   • Single-buffered shared memory
//
// Kernel 2: matmul_wmma_opt  (optimised)
//   • 128×128 block tile, 4×2 warps (8 warps / 256 threads)
//   • 2×4 warp tile — 8 MMA fragment pairs per warp per K-step
//   • Double-buffered shared memory — prefetch tile k+1 while computing tile k
//   • +8-half row padding — eliminates smem bank conflicts on load_matrix_sync
//   • __launch_bounds__(256, 2) — guides register allocator for max occupancy
//   • #pragma unroll on every inner loop

#include <mma.h>
#include <cuda_fp16.h>
#include "kernels.cuh"
#include "utils.cuh"

using namespace nvcuda::wmma;

// ─── WMMA fragment shape (fixed for Ampere fp16→fp32) ────────────────────────
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// =============================================================================
//  KERNEL 1 — Basic WMMA
//  Block tile : 64 rows × 64 cols
//  Warp layout: 4 rows × 4 cols  (16 warps, 512 threads/block)
//  K-slice    : WMMA_K = 16
// =============================================================================
#define BM_B 64
#define BN_B 64
#define BK_B 16

__global__ void matmul_wmma(const __half * __restrict__ A,
                             const __half * __restrict__ B,
                             float        * __restrict__ C,
                             int n)
{
    // Warp coordinates within the block
    const int warpId  = threadIdx.x / 32;
    const int warpRow = warpId / (BN_B / WMMA_N);   // 0..3
    const int warpCol = warpId % (BN_B / WMMA_N);   // 0..3

    // Global C origin for this warp's 16×16 output tile
    const int blockRow      = blockIdx.y * BM_B;
    const int blockCol      = blockIdx.x * BN_B;
    const int globalWarpRow = blockRow + warpRow * WMMA_M;
    const int globalWarpCol = blockCol + warpCol * WMMA_N;

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    fill_fragment(acc, 0.0f);

    // Single-buffered smem
    __shared__ __half sA[BM_B][BK_B];
    __shared__ __half sB[BK_B][BN_B];

    for (int k = 0; k < n; k += BK_B) {
        // Load A tile: all threads cooperate, strided by blockDim.x
        for (int idx = threadIdx.x; idx < BM_B * BK_B; idx += blockDim.x) {
            int r = idx / BK_B, c = idx % BK_B;
            int gR = blockRow + r, gC = k + c;
            sA[r][c] = (gR < n && gC < n) ? A[gR * n + gC] : (__half)0.f;
        }
        // Load B tile
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


// =============================================================================
//  KERNEL 2 — Optimised WMMA
//
//  Block tile  : 128 rows × 128 cols   (4× area vs basic)
//  Warp layout : WARPS_ROW(4) × WARPS_COL(2) = 8 warps = 256 threads
//  Warp tile   : WARP_TILE_M(2) × WARP_TILE_N(4) fragments per warp
//                → 8 MMA ops per warp per K-step
//  K-tile      : BK_O = 32  (= 2 × WMMA_K)
//  Buffering   : double (load tile k+1 while computing tile k)
//  smem padding: +8 halves/row → avoids 2-way bank conflicts on
//                load_matrix_sync (each 16-row fragment row maps to a
//                unique bank group after padding)
// =============================================================================

// ── Geometry ──────────────────────────────────────────────────────────────────
#define BM_O          128
#define BN_O          128
#define BK_O           32   // 2 × WMMA_K — two MMA K-steps per smem stage
#define WARP_TILE_M     2   // fragments per warp along M
#define WARP_TILE_N     4   // fragments per warp along N
#define WARPS_ROW_O     4   // warp rows  (128 / (2×16) = 4)
#define WARPS_COL_O     2   // warp cols  (128 / (4×16) = 2)
#define BLOCK_WARPS_O  (WARPS_ROW_O * WARPS_COL_O)   // 8
#define BLOCK_THR_O    (BLOCK_WARPS_O * 32)           // 256

// ── smem layout ───────────────────────────────────────────────────────────────
// Padding shifts consecutive rows to different 128-byte (32-bank × 4-byte)
// smem bank groups, preventing conflicts when load_matrix_sync reads 16-row
// fragments from adjacent warps simultaneously.
#define PAD     8
#define SA_LD  (BK_O + PAD)   // stride of sA rows in halves  = 40
#define SB_LD  (BN_O + PAD)   // stride of sB rows in halves  = 136

__global__ __launch_bounds__(BLOCK_THR_O, 2)
void matmul_wmma_opt(const __half * __restrict__ A,
                     const __half * __restrict__ B,
                     float        * __restrict__ C,
                     int n)
{
    // ── Thread / warp IDs ─────────────────────────────────────────────────────
    const int tid    = threadIdx.x;
    const int warpId = tid / 32;
    const int warpRow = warpId / WARPS_COL_O;   // 0..3
    const int warpCol = warpId % WARPS_COL_O;   // 0..1

    // Global C origin for this block
    const int bRowOrig = blockIdx.y * BM_O;
    const int bColOrig = blockIdx.x * BN_O;

    // ── Accumulators — one per warp-tile output cell ───────────────────────────
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_TILE_M][WARP_TILE_N];
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++)
            fill_fragment(acc[i][j], 0.0f);

    // ── Double-buffered shared memory ─────────────────────────────────────────
    //   sA[buf][BM_O rows][SA_LD halves/row]   — A tile
    //   sB[buf][BK_O rows][SB_LD halves/row]   — B tile
    __shared__ __half sA[2][BM_O][SA_LD];
    __shared__ __half sB[2][BK_O][SB_LD];

    const int numTiles = (n + BK_O - 1) / BK_O;

    // ── Prefetch tile 0 → buffer 0 ────────────────────────────────────────────
    {
        #pragma unroll 4
        for (int idx = tid; idx < BM_O * BK_O; idx += BLOCK_THR_O) {
            int r = idx / BK_O, c = idx % BK_O;
            int gR = bRowOrig + r, gC = c;          // tile 0: k=0
            sA[0][r][c] = (gR < n && gC < n) ? A[gR * n + gC] : (__half)0.f;
        }
        #pragma unroll 4
        for (int idx = tid; idx < BK_O * BN_O; idx += BLOCK_THR_O) {
            int r = idx / BN_O, c = idx % BN_O;
            int gR = r, gC = bColOrig + c;          // tile 0: k=0
            sB[0][r][c] = (gR < n && gC < n) ? B[gR * n + gC] : (__half)0.f;
        }
    }
    __syncthreads();

    // ── Main K-loop ───────────────────────────────────────────────────────────
    for (int tile = 0; tile < numTiles; tile++) {
        const int cur = tile & 1;
        const int nxt = 1 - cur;

        // ── Prefetch next tile into nxt buffer ────────────────────────────────
        // Runs concurrently with the MMA below on the current buffer.
        if (tile + 1 < numTiles) {
            const int kBase = (tile + 1) * BK_O;
            #pragma unroll 4
            for (int idx = tid; idx < BM_O * BK_O; idx += BLOCK_THR_O) {
                int r = idx / BK_O, c = idx % BK_O;
                int gR = bRowOrig + r, gC = kBase + c;
                sA[nxt][r][c] = (gR < n && gC < n) ? A[gR * n + gC] : (__half)0.f;
            }
            #pragma unroll 4
            for (int idx = tid; idx < BK_O * BN_O; idx += BLOCK_THR_O) {
                int r = idx / BN_O, c = idx % BN_O;
                int gR = kBase + r, gC = bColOrig + c;
                sB[nxt][r][c] = (gR < n && gC < n) ? B[gR * n + gC] : (__half)0.f;
            }
        }

        // ── MMA over BK_O / WMMA_K = 2 WMMA_K steps ─────────────────────────
        // Each ks step issues WARP_TILE_M × WARP_TILE_N = 8 MMA ops.
        #pragma unroll
        for (int ks = 0; ks < BK_O / WMMA_K; ks++) {

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major>
                a_frag[WARP_TILE_M];
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major>
                b_frag[WARP_TILE_N];

            // Load A fragments for WARP_TILE_M rows owned by this warp
            #pragma unroll
            for (int i = 0; i < WARP_TILE_M; i++) {
                int smRow = warpRow * WARP_TILE_M * WMMA_M + i * WMMA_M;
                int smCol = ks * WMMA_K;
                load_matrix_sync(a_frag[i],
                                 (const __half*)sA[cur] + smRow * SA_LD + smCol,
                                 SA_LD);
            }

            // Load B fragments for WARP_TILE_N cols owned by this warp
            #pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                int smRow = ks * WMMA_K;
                int smCol = warpCol * WARP_TILE_N * WMMA_N + j * WMMA_N;
                load_matrix_sync(b_frag[j],
                                 (const __half*)sB[cur] + smRow * SB_LD + smCol,
                                 SB_LD);
            }

            // Outer-product accumulation — WARP_TILE_M × WARP_TILE_N MMA ops
            #pragma unroll
            for (int i = 0; i < WARP_TILE_M; i++)
                #pragma unroll
                for (int j = 0; j < WARP_TILE_N; j++)
                    mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        // Ensure nxt-tile stores from the prefetch loop above are globally
        // visible to all threads before the next iteration flips cur/nxt.
        __syncthreads();
    }

    // ── Store accumulators → global C ─────────────────────────────────────────
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            int gRow = bRowOrig + warpRow * WARP_TILE_M * WMMA_M + i * WMMA_M;
            int gCol = bColOrig + warpCol * WARP_TILE_N * WMMA_N + j * WMMA_N;
            if (gRow < n && gCol < n)
                store_matrix_sync(C + gRow * n + gCol,
                                  acc[i][j], n, mem_row_major);
        }
    }
}


// =============================================================================
//  fp32 → fp16 conversion helper
//  Launched by the host before either WMMA kernel.
// =============================================================================
__global__ void float2half_kernel(const float * __restrict__ src,
                                   __half      * __restrict__ dst,
                                   int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __float2half(src[i]);
}

void convert_to_half(const float *d_fp32, __half *d_fp16, int n)
{
    const int total   = n * n;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    float2half_kernel<<<blocks, threads>>>(d_fp32, d_fp16, total);
    CHECK(cudaGetLastError());
}
