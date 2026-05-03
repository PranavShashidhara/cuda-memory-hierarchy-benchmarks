#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernels.cuh"
#include "utils.cuh"

#define RUNS 10

// -----------------------------
// 1D KERNEL BENCHMARK (sequential)
// -----------------------------
void benchmark_1D(const char* name,
                  void (*kernel)(float*, float*, int),
                  float *d_a, float *d_c,
                  int N, int blocks, int threads)
{
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // warmup
    kernel<<<blocks, threads>>>(d_a, d_c, N);
    cudaDeviceSynchronize();

    float total = 0;

    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);

        kernel<<<blocks, threads>>>(d_a, d_c, N);

        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms;
        cudaEventElapsedTime(&ms, s, e);
        total += ms;
    }

    printf("%s avg: %f ms\n", name, total / RUNS);
}

// -----------------------------
// STRIDED BENCHMARK (special signature)
// -----------------------------
void benchmark_strided(const char* name,
                       void (*kernel)(float*, float*, int, int),
                       float *d_a, float *d_c,
                       int N, int stride,
                       int blocks, int threads)
{
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // warmup
    kernel<<<blocks, threads>>>(d_a, d_c, stride, N);
    cudaDeviceSynchronize();

    float total = 0;

    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);

        kernel<<<blocks, threads>>>(d_a, d_c, stride, N);

        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms;
        cudaEventElapsedTime(&ms, s, e);
        total += ms;
    }

    printf("%s avg: %f ms\n", name, total / RUNS);
}

// -----------------------------
// MAIN
// -----------------------------
int main()
{
    int N = 1 << 22;   // ~4M elements

    size_t size = N * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    printf("\n===== MEMORY ACCESS BENCHMARKS =====\n");

    benchmark_1D("Sequential", sequential, d_a, d_c, N, blocks, threads);
    benchmark_strided("Strided", strided, d_a, d_c, N, 4, blocks, threads);

    printf("\n===== MATRIX MULTIPLY =====\n");

    int n = 2048;
    dim3 block(16, 16);
    dim3 grid(n / 16, n / 16);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    float total_naive = 0;
    float total_tiled = 0;

    // warmup
    matmul_naive<<<grid, block>>>(d_a, d_b, d_c, n);
    matmul_tiled<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // naive
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        matmul_naive<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms;
        cudaEventElapsedTime(&ms, s, e);
        total_naive += ms;
    }

    // tiled
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        matmul_tiled<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms;
        cudaEventElapsedTime(&ms, s, e);
        total_tiled += ms;
    }

    printf("MatMul Naive avg: %f ms\n", total_naive / RUNS);
    printf("MatMul Tiled avg: %f ms\n", total_tiled / RUNS);

    printf("\nSpeedup: %fx\n", total_naive / total_tiled);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}