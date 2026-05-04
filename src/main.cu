#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "kernels.cuh"
#include "utils.cuh"

#define RUNS 10

// -----------------------------
// Bandwidth helper
// bytes_accessed = total bytes read + written by the kernel
// -----------------------------
static inline float bandwidth_GBs(float ms, size_t bytes_accessed) {
    return (float)bytes_accessed / (ms * 1e-3f) / 1e9f;
}

// -----------------------------
// 1D KERNEL BENCHMARK (sequential / any single-ptr kernel)
// Returns avg time in ms
// -----------------------------
float benchmark_1D(const char* name,
                   void (*kernel)(float*, float*, int),
                   float *d_a, float *d_c,
                   int N, int blocks, int threads,
                   size_t bytes_accessed)
{
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // warmup
    kernel<<<blocks, threads>>>(d_a, d_c, N);
    cudaDeviceSynchronize();

    float total = 0.0f;

    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        kernel<<<blocks, threads>>>(d_a, d_c, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms;
        cudaEventElapsedTime(&ms, s, e);
        total += ms;
    }

    float avg_ms = total / RUNS;
    float bw     = bandwidth_GBs(avg_ms, bytes_accessed);
    printf("  %-20s  %8.3f ms   %6.2f GB/s\n", name, avg_ms, bw);

    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return avg_ms;
}

// -----------------------------
// STRIDED BENCHMARK
// Returns avg time in ms
// -----------------------------
float benchmark_strided(const char* name,
                        void (*kernel)(float*, float*, int, int),
                        float *d_a, float *d_c,
                        int N, int stride,
                        int blocks, int threads,
                        size_t bytes_accessed)
{
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // warmup
    kernel<<<blocks, threads>>>(d_a, d_c, stride, N);
    cudaDeviceSynchronize();

    float total = 0.0f;

    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        kernel<<<blocks, threads>>>(d_a, d_c, stride, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms;
        cudaEventElapsedTime(&ms, s, e);
        total += ms;
    }

    float avg_ms = total / RUNS;
    float bw     = bandwidth_GBs(avg_ms, bytes_accessed);
    printf("  %-20s  %8.3f ms   %6.2f GB/s\n", name, avg_ms, bw);

    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return avg_ms;
}

// -----------------------------
// RANDOM ACCESS BENCHMARK
// Returns avg time in ms
// -----------------------------
float benchmark_random(const char* name,
                       float *d_a, float *d_c,
                       int *d_idx,
                       int N, int blocks, int threads,
                       size_t bytes_accessed)
{
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // warmup
    random_access<<<blocks, threads>>>(d_a, d_c, d_idx, N);
    cudaDeviceSynchronize();

    float total = 0.0f;

    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        random_access<<<blocks, threads>>>(d_a, d_c, d_idx, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms;
        cudaEventElapsedTime(&ms, s, e);
        total += ms;
    }

    float avg_ms = total / RUNS;
    float bw     = bandwidth_GBs(avg_ms, bytes_accessed);
    printf("  %-20s  %8.3f ms   %6.2f GB/s\n", name, avg_ms, bw);

    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return avg_ms;
}

// -----------------------------
// RUN ONE SIZE — allocates, benchmarks all 3 patterns, frees
// -----------------------------
void run_memory_size(int N)
{
    size_t size = N * sizeof(float);

    // --- host alloc ---
    float *h_a   = (float*)malloc(size);
    int   *h_idx = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) h_a[i] = (float)(rand() % 100);

    // random index permutation (Fisher-Yates)
    for (int i = 0; i < N; i++) h_idx[i] = i;
    for (int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = h_idx[i]; h_idx[i] = h_idx[j]; h_idx[j] = tmp;
    }

    // --- device alloc ---
    float *d_a, *d_c;
    int   *d_idx;

    CHECK(cudaMalloc(&d_a,   size));
    CHECK(cudaMalloc(&d_c,   size));
    CHECK(cudaMalloc(&d_idx, N * sizeof(int)));

    CHECK(cudaMemcpy(d_a,   h_a,   size,            cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_idx, h_idx, N * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // bytes: each kernel reads N floats (d_a) + writes N floats (d_c)
    // random_access also reads N ints (d_idx) for the index array
    size_t bytes_rw     = 2 * size;                    // read + write
    size_t bytes_random = 2 * size + N * sizeof(int);  // + index array read

    printf("\n  N = %d  (~%.0f MB)\n", N, (double)size / (1024.0 * 1024.0));
    printf("  %-20s  %8s     %s\n", "Kernel", "Avg ms", "Bandwidth");
    printf("  %s\n", "------------------------------------------------------");

    float ms_seq    = benchmark_1D     ("Sequential",    sequential,   d_a, d_c, N, blocks, threads, bytes_rw);
    float ms_stride = benchmark_strided("Strided (x4)",  strided,      d_a, d_c, N, 4, blocks, threads, bytes_rw);
    float ms_rand   = benchmark_random ("Random access",               d_a, d_c, d_idx, N, blocks, threads, bytes_random);

    // degradation relative to sequential
    printf("  Strided slowdown vs sequential: %.2fx\n",   ms_stride / ms_seq);
    printf("  Random  slowdown vs sequential: %.2fx\n",   ms_rand   / ms_seq);

    // --- free ---
    cudaFree(d_a); cudaFree(d_c); cudaFree(d_idx);
    free(h_a); free(h_idx);
}

// -----------------------------
// MATMUL BENCHMARK (unchanged logic, cleaner output)
// -----------------------------
void run_matmul()
{
    int n = 2048;

    size_t size = (size_t)n * n * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    for (int i = 0; i < n * n; i++) { h_a[i] = rand() % 100; h_b[i] = rand() % 100; }

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, size));
    CHECK(cudaMalloc(&d_b, size));
    CHECK(cudaMalloc(&d_c, size));
    CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(n / 16, n / 16);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    float total_naive = 0.0f, total_tiled = 0.0f;

    // warmup
    matmul_naive<<<grid, block>>>(d_a, d_b, d_c, n);
    matmul_tiled<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        matmul_naive<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        total_naive += ms;
    }

    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        matmul_tiled<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        total_tiled += ms;
    }

    float avg_naive = total_naive / RUNS;
    float avg_tiled = total_tiled / RUNS;

    // GFLOPS: 2*n^3 floating point ops
    double flops = 2.0 * (double)n * n * n;
    float  gf_naive = (float)(flops / (avg_naive * 1e-3) / 1e9);
    float  gf_tiled = (float)(flops / (avg_tiled * 1e-3) / 1e9);

    printf("\n  Matrix size: %dx%d\n", n, n);
    printf("  %-20s  %8.3f ms   %6.2f GFLOPS\n", "Naive",  avg_naive, gf_naive);
    printf("  %-20s  %8.3f ms   %6.2f GFLOPS\n", "Tiled",  avg_tiled, gf_tiled);
    printf("  Speedup: %.2fx\n", avg_naive / avg_tiled);

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b);
}

// -----------------------------
// MAIN
// -----------------------------
int main()
{
    srand((unsigned)time(NULL));

    printf("=========================================\n");
    printf("  CUDA Benchmark Suite — Jetson Orin Nano\n");
    printf("=========================================\n");

    // --- Phase 3: memory hierarchy sweep ---
    // Six sizes: 64K → 64M elements (each 4 bytes = 256 KB → 256 MB)
    printf("\n\n===== MEMORY ACCESS BENCHMARK (size sweep) =====\n");

    int sizes[] = {
        1 << 16,   //  64K elements —  256 KB  (fits in L2)
        1 << 18,   // 256K elements —    1 MB
        1 << 20,   //   1M elements —    4 MB
        1 << 22,   //   4M elements —   16 MB
        1 << 24,   //  16M elements —   64 MB
        1 << 26,   //  64M elements —  256 MB
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        run_memory_size(sizes[s]);
    }

    // --- Phase 1/2: matmul ---
    printf("\n\n===== MATRIX MULTIPLY =====\n");
    run_matmul();

    printf("\n=========================================\n");
    printf("  Done.\n");
    printf("=========================================\n");

    return 0;
}