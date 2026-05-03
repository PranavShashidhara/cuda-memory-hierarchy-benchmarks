#include <stdio.h>
#include <stdlib.h>
#include "kernels.cuh"
#include "utils.cuh"

#define N (1 << 20)

int main() {

    size_t size = N * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    float *d_a, *d_b, *d_c;

    CHECK(cudaMalloc(&d_a, size));
    CHECK(cudaMalloc(&d_b, size));
    CHECK(cudaMalloc(&d_c, size));

    CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    cudaEventRecord(s);
    sequential<<<blocks, threads>>>(d_a, d_c, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float seq_ms;
    cudaEventElapsedTime(&seq_ms, s, e);

    printf("Sequential: %f ms\n", seq_ms);

    cudaEventRecord(s);
    strided<<<blocks, threads>>>(d_a, d_c, 4, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float stride_ms;
    cudaEventElapsedTime(&stride_ms, s, e);

    printf("Strided: %f ms\n", stride_ms);

    int n = 256;
    dim3 block(16,16);
    dim3 grid(n/16, n/16);

    cudaEventRecord(s);
    matmul_naive<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float naive_ms;
    cudaEventElapsedTime(&naive_ms, s, e);

    printf("Naive: %f ms\n", naive_ms);

    cudaEventRecord(s);
    matmul_tiled<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float tiled_ms;
    cudaEventElapsedTime(&tiled_ms, s, e);

    printf("Tiled: %f ms\n", tiled_ms);

    printf("Speedup: %fx\n", naive_ms / tiled_ms);

    return 0;
}