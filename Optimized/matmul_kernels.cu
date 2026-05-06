#include "kernels.cuh"

// ============================================================================
// NAIVE: Unrolled with #pragma unroll — compiler handles ILP
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
// TILED: 32x32 shared memory tile, +1 padding to eliminate bank conflicts
// Block = 32x32 threads, Grid = (n/32) x (n/32)
// ============================================================================
__global__ void matmul_tiled(float *A, float *B, float *C, int n) {
    __shared__ float sA[TILE_DIM][TILE_DIM + 1];
    __shared__ float sB[TILE_DIM][TILE_DIM + 1];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_DIM - 1) / TILE_DIM; t++) {
        int tA_col = t * TILE_DIM + threadIdx.x;
        int tB_row = t * TILE_DIM + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < n && tA_col < n) ? A[row * n + tA_col] : 0.f;
        sB[threadIdx.y][threadIdx.x] = (tB_row < n && col < n) ? B[tB_row * n + col] : 0.f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

// ============================================================================
// THREAD COARSENING (TILED V2): Each thread computes a WPT x WPT tile
// of output using register blocking. Reduces redundant shared memory loads
// and dramatically increases arithmetic intensity per memory transaction.
//
// Block = (TILE_DIM/WPT) x (TILE_DIM/WPT) threads
// Each thread owns WPT rows and WPT cols of the output tile.
// WPT=4 → each thread computes 4x4=16 output values, 16x more work per load.
// ============================================================================
#define WPT 4   // Work per thread (tile side)
#define RTS (TILE_DIM / WPT)  // = 8 threads per tile dim

__global__ void matmul_tiled_v2(float *A, float *B, float *C, int n) {
    __shared__ float sA[TILE_DIM][TILE_DIM + 1];
    __shared__ float sB[TILE_DIM][TILE_DIM + 1];

    int tid_row = threadIdx.y;  // 0..RTS-1
    int tid_col = threadIdx.x;  // 0..RTS-1

    int base_row = blockIdx.y * TILE_DIM + tid_row * WPT;
    int base_col = blockIdx.x * TILE_DIM + tid_col * WPT;

    float acc[WPT][WPT] = {};  // WPT x WPT register accumulator, zero-init

    int num_tiles = (n + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; t++) {
        // Each thread loads WPT elements of A and WPT elements of B into smem
        #pragma unroll
        for (int w = 0; w < WPT; w++) {
            int A_row = base_row + w;
            int A_col = t * TILE_DIM + tid_col;
            sA[tid_row * WPT + w][tid_col] =
                (A_row < n && A_col < n) ? A[A_row * n + A_col] : 0.f;

            int B_row = t * TILE_DIM + tid_row;
            int B_col = base_col + w;
            sB[tid_row][tid_col * WPT + w] =
                (B_row < n && B_col < n) ? B[B_row * n + B_col] : 0.f;
        }
        __syncthreads();

        // Compute WPT x WPT output block from tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a_regs[WPT], b_regs[WPT];
            #pragma unroll
            for (int w = 0; w < WPT; w++) {
                a_regs[w] = sA[tid_row * WPT + w][k];
                b_regs[w] = sB[k][tid_col * WPT + w];
            }
            #pragma unroll
            for (int wi = 0; wi < WPT; wi++)
                #pragma unroll
                for (int wj = 0; wj < WPT; wj++)
                    acc[wi][wj] += a_regs[wi] * b_regs[wj];
        }
        __syncthreads();
    }

    // Write WPT x WPT results
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
// DOUBLE-BUFFERED TILED V3: Proper async prefetch using two smem buffers
// While computing tile t, loads tile t+1 into the other buffer.
// Hides global memory latency behind compute — true pipelining.
// ============================================================================
__global__ void matmul_tiled_v3(float *A, float *B, float *C, int n) {
    // Two buffers, index alternates each tile
    __shared__ float sA[2][TILE_DIM][TILE_DIM + 1];
    __shared__ float sB[2][TILE_DIM][TILE_DIM + 1];

    int ty  = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;
    float sum = 0.0f;

    int num_tiles = (n + TILE_DIM - 1) / TILE_DIM;

    // Load tile 0 into buffer 0
    {
        int tA_col = tx, tB_row = ty;
        sA[0][ty][tx] = (row < n && tA_col < n) ? A[row * n + tA_col] : 0.f;
        sB[0][ty][tx] = (tB_row < n && col < n) ? B[tB_row * n + col] : 0.f;
    }
    __syncthreads();

    for (int t = 0; t < num_tiles; t++) {
        int cur = t & 1;
        int nxt = 1 - cur;

        // Prefetch tile t+1 into the next buffer while computing tile t
        if (t + 1 < num_tiles) {
            int nA_col = (t + 1) * TILE_DIM + tx;
            int nB_row = (t + 1) * TILE_DIM + ty;
            sA[nxt][ty][tx] = (row < n && nA_col < n) ? A[row * n + nA_col] : 0.f;
            sB[nxt][ty][tx] = (nB_row < n && col < n) ? B[nB_row * n + col] : 0.f;
        }

        // Compute from current buffer
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++)
            sum += sA[cur][ty][k] * sB[cur][k][tx];

        __syncthreads();  // ensure prefetch writes complete before next iter
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}