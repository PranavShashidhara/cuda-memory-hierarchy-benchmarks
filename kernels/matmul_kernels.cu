#include "kernels.cuh"

__global__ void matmul_naive(float *A, float *B, float *C, int n) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {

        float sum = 0.0f;

        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = sum;
    }
}

__global__ void matmul_tiled(float *A, float *B, float *C, int n) {

    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {

        int tiled_col = t * TILE + threadIdx.x;
        int tiled_row = t * TILE + threadIdx.y;

        // load A tile safely
        if (row < n && tiled_col < n)
            sA[threadIdx.y][threadIdx.x] = A[row * n + tiled_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // load B tile safely
        if (tiled_row < n && col < n)
            sB[threadIdx.y][threadIdx.x] = B[tiled_row * n + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}