#include "kernels.cuh"
#include <cuda_pipeline.h>   // __pipeline_memcpy_async, commit, wait_prior

// ============================================================================
// NAIVE: baseline, #pragma unroll 8 for ILP
// ============================================================================
__global__ void matmul_naive(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    float sum = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < n; k++)
        sum += A[row * n + k] * B[k * n + col];
    C[row * n + col] = sum;
}

// ============================================================================
// TILED V1: shared memory tile, bank-conflict-free (+1 pad)
// Block = TILE_DIM x TILE_DIM
// ============================================================================
__global__ void matmul_tiled(float *A, float *B, float *C, int n) {
    __shared__ float sA[TILE_DIM][TILE_DIM + 1];
    __shared__ float sB[TILE_DIM][TILE_DIM + 1];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_DIM - 1) / TILE_DIM; t++) {
        int tA_col = t * TILE_DIM + tx;
        int tB_row = t * TILE_DIM + ty;

        sA[ty][tx] = (row < n && tA_col < n) ? A[row * n + tA_col] : 0.f;
        sB[ty][tx] = (tB_row < n && col < n) ? B[tB_row * n + col] : 0.f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++)
            sum += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

// ============================================================================
// TILED V2: Thread coarsening WPT x WPT register blocking
// Each thread accumulates WPT*WPT outputs in registers — never spills to smem.
// Reads each smem element WPT times across the acc loop, maximizing L1 reuse.
// Block = RTS x RTS (= TILE_DIM/WPT each dim)
// ============================================================================
__global__ void matmul_tiled_v2(float *A, float *B, float *C, int n) {
    __shared__ float sA[TILE_DIM][TILE_DIM + 1];
    __shared__ float sB[TILE_DIM][TILE_DIM + 1];

    int ty = threadIdx.y, tx = threadIdx.x;
    int base_row = blockIdx.y * TILE_DIM + ty * WPT;
    int base_col = blockIdx.x * TILE_DIM + tx * WPT;

    float acc[WPT][WPT] = {};

    int num_tiles = (n + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; t++) {
        // Each thread loads WPT rows of A and WPT cols of B
        #pragma unroll
        for (int w = 0; w < WPT; w++) {
            int A_row = base_row + w,  A_col = t * TILE_DIM + tx;
            int B_row = t * TILE_DIM + ty, B_col = base_col + w;

            sA[ty * WPT + w][tx] = (A_row < n && A_col < n) ? A[A_row * n + A_col] : 0.f;
            sB[ty][tx * WPT + w] = (B_row < n && B_col < n) ? B[B_row * n + B_col] : 0.f;
        }
        __syncthreads();

        // Register-cache rows of sA and cols of sB before the inner loop.
        // This keeps values in RF across all WPT*WPT MACs, so smem is read
        // once per k and each value is reused WPT times — boosting L1 hits.
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a_reg[WPT], b_reg[WPT];
            #pragma unroll
            for (int w = 0; w < WPT; w++) {
                a_reg[w] = sA[ty * WPT + w][k];   // row w, col k  → stays in RF
                b_reg[w] = sB[k][tx * WPT + w];   // row k, col w  → stays in RF
            }
            #pragma unroll
            for (int wi = 0; wi < WPT; wi++)
                #pragma unroll
                for (int wj = 0; wj < WPT; wj++)
                    acc[wi][wj] += a_reg[wi] * b_reg[wj];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int wi = 0; wi < WPT; wi++)
        #pragma unroll
        for (int wj = 0; wj < WPT; wj++) {
            int r = base_row + wi, c = base_col + wj;
            if (r < n && c < n)
                C[r * n + c] = acc[wi][wj];
        }
}

// ============================================================================
// TILED V3: cp.async double-buffered pipeline + WPT register blocking
//
// Key ideas:
//   1. cp.async copies global → shared DIRECTLY, bypassing registers.
//      The thread issues the copy and moves on — zero stall on LDG.
//   2. Double buffer: while computing tile t from buf[cur],
//      cp.async is already filling buf[nxt] with tile t+1.
//   3. __pipeline_wait_prior(1): wait until all but the most recent
//      in-flight copy is done — ensures cur buffer is safe to read
//      WITHOUT waiting for the nxt buffer copy to finish.
//   4. Register blocking (WPT x WPT) same as v2 — maximizes L1 reuse
//      of shared memory data once it lands.
//
// Block = RTS x RTS, Grid = (n/TILE_DIM) x (n/TILE_DIM)
// ============================================================================
__global__ void matmul_tiled_v3(float *A, float *B, float *C, int n) {
    // Double-buffered shared memory — 2 x (TILE_DIM x (TILE_DIM+1)) floats each
    __shared__ float sA[2][TILE_DIM][TILE_DIM + 1];
    __shared__ float sB[2][TILE_DIM][TILE_DIM + 1];

    int ty = threadIdx.y, tx = threadIdx.x;
    int base_row = blockIdx.y * TILE_DIM + ty * WPT;
    int base_col = blockIdx.x * TILE_DIM + tx * WPT;

    float acc[WPT][WPT] = {};
    int num_tiles = (n + TILE_DIM - 1) / TILE_DIM;

    // ── Prologue: async-load tile 0 into buffer 0 ──────────────────────────
    #pragma unroll
    for (int w = 0; w < WPT; w++) {
        int A_row = base_row + w,  A_col = tx;           // tile 0 col = tx
        int B_row = ty,            B_col = base_col + w; // tile 0 row = ty

        // cp.async: 4 bytes, global → shared, no register touch
        __pipeline_memcpy_async(
            &sA[0][ty * WPT + w][tx],
            (A_row < n && A_col < n) ? &A[A_row * n + A_col] : nullptr,
            sizeof(float));
        __pipeline_memcpy_async(
            &sB[0][ty][tx * WPT + w],
            (B_row < n && B_col < n) ? &B[B_row * n + B_col] : nullptr,
            sizeof(float));
    }
    __pipeline_commit();  // seal group 0

    // ── Main loop ──────────────────────────────────────────────────────────
    for (int t = 0; t < num_tiles; t++) {
        int cur = t & 1;
        int nxt = 1 - cur;

        // Async-load tile t+1 into nxt buffer BEFORE waiting on cur
        if (t + 1 < num_tiles) {
            int base_k = (t + 1) * TILE_DIM;
            #pragma unroll
            for (int w = 0; w < WPT; w++) {
                int A_row = base_row + w,  A_col = base_k + tx;
                int B_row = base_k + ty,   B_col = base_col + w;

                __pipeline_memcpy_async(
                    &sA[nxt][ty * WPT + w][tx],
                    (A_row < n && A_col < n) ? &A[A_row * n + A_col] : nullptr,
                    sizeof(float));
                __pipeline_memcpy_async(
                    &sB[nxt][ty][tx * WPT + w],
                    (B_row < n && B_col < n) ? &B[B_row * n + B_col] : nullptr,
                    sizeof(float));
            }
            __pipeline_commit();  // seal group t+1
        }

        // Wait for tile t's group (all but the most recent in-flight copy)
        // This unblocks as soon as cur is ready, not waiting for nxt.
        __pipeline_wait_prior(1);
        __syncthreads();  // all threads must see cur buffer before computing

        // Compute tile t from cur buffer — register-cached for L1 reuse
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a_reg[WPT], b_reg[WPT];
            #pragma unroll
            for (int w = 0; w < WPT; w++) {
                a_reg[w] = sA[cur][ty * WPT + w][k];
                b_reg[w] = sB[cur][k][tx * WPT + w];
            }
            #pragma unroll
            for (int wi = 0; wi < WPT; wi++)
                #pragma unroll
                for (int wj = 0; wj < WPT; wj++)
                    acc[wi][wj] += a_reg[wi] * b_reg[wj];
        }
        __syncthreads();  // done with cur buffer before nxt group overwrites it
    }

    // ── Epilogue: write results ─────────────────────────────────────────────
    #pragma unroll
    for (int wi = 0; wi < WPT; wi++)
        #pragma unroll
        for (int wj = 0; wj < WPT; wj++) {
            int r = base_row + wi, c = base_col + wj;
            if (r < n && c < n)
                C[r * n + c] = acc[wi][wj];
        }
}
