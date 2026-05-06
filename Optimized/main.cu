#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <climits>

#include "kernels.cuh"
#include "utils.cuh"

#define RUNS 10

static inline float bandwidth_GBs(float ms, size_t bytes) {
    return (float)bytes / (ms * 1e-3f) / 1e9f;
}

#define PEAK_MEM_BW 51.2f

// ============================================================================
// 1D BENCHMARK (sequential / sequential_shared)
// ============================================================================
float benchmark_1D(const char* name,
                   void (*kernel)(float*, float*, int),
                   float *d_a, float *d_c,
                   int N, int blocks, int threads,
                   size_t smem, size_t bytes)
{
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    kernel<<<blocks, threads, smem>>>(d_a, d_c, N);
    cudaDeviceSynchronize();

    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        kernel<<<blocks, threads, smem>>>(d_a, d_c, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        total += ms;
    }

    float avg = total / RUNS;
    float bw  = bandwidth_GBs(avg, bytes);
    printf("  %-28s  %8.3f ms   %6.2f GB/s  (%5.1f%% peak)\n",
           name, avg, bw, bw / PEAK_MEM_BW * 100.f);

    cudaEventDestroy(s); cudaEventDestroy(e);
    return avg;
}

// ============================================================================
// STRIDED BENCHMARK
// ============================================================================
float benchmark_strided(const char* name,
                        void (*kernel)(float*, float*, int, int),
                        float *d_a, float *d_c,
                        int N, int stride, int blocks, int threads,
                        size_t bytes)
{
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    kernel<<<blocks, threads>>>(d_a, d_c, stride, N);
    cudaDeviceSynchronize();

    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        kernel<<<blocks, threads>>>(d_a, d_c, stride, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        total += ms;
    }

    float avg = total / RUNS;
    float bw  = bandwidth_GBs(avg, bytes);
    printf("  %-28s  %8.3f ms   %6.2f GB/s  (%5.1f%% peak)\n",
           name, avg, bw, bw / PEAK_MEM_BW * 100.f);

    cudaEventDestroy(s); cudaEventDestroy(e);
    return avg;
}

// ============================================================================
// RANDOM ACCESS BENCHMARK
// ============================================================================
float benchmark_random(const char* name,
                       float *d_a, float *d_c, int *d_idx,
                       int N, int blocks, int threads,
                       size_t bytes)
{
    // shared mem = one int per thread for bitonic sort
    size_t smem = threads * sizeof(int);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    random_access<<<blocks, threads, smem>>>(d_a, d_c, d_idx, N);
    cudaDeviceSynchronize();

    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        random_access<<<blocks, threads, smem>>>(d_a, d_c, d_idx, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        total += ms;
    }

    float avg = total / RUNS;
    float bw  = bandwidth_GBs(avg, bytes);
    printf("  %-28s  %8.3f ms   %6.2f GB/s  (%5.1f%% peak)\n",
           name, avg, bw, bw / PEAK_MEM_BW * 100.f);

    cudaEventDestroy(s); cudaEventDestroy(e);
    return avg;
}

// ============================================================================
// RUN ONE MEMORY SIZE
// ============================================================================
void run_memory_size(int N)
{
    size_t size = N * sizeof(float);

    float *h_a   = (float*)malloc(size);
    int   *h_idx = (int*)  malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) h_a[i] = (float)(rand() % 100);
    for (int i = 0; i < N; i++) h_idx[i] = i;
    for (int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = h_idx[i]; h_idx[i] = h_idx[j]; h_idx[j] = tmp;
    }

    float *d_a, *d_c; int *d_idx;
    CHECK(cudaMalloc(&d_a,   size));
    CHECK(cudaMalloc(&d_c,   size));
    CHECK(cudaMalloc(&d_idx, N * sizeof(int)));
    CHECK(cudaMemcpy(d_a,   h_a,   size,            cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_idx, h_idx, N * sizeof(int), cudaMemcpyHostToDevice));

    // float4 kernel: each thread handles 4 floats
    int threads  = 256;
    int blocks4  = (N / 4 + threads - 1) / threads;
    int blocks1  = (N     + threads - 1) / threads;

    size_t bytes_rw  = 2 * size;
    size_t bytes_rnd = 2 * size + N * sizeof(int);

    printf("\n  N = %d  (~%.0f MB)  [Threads: %d]\n",
           N, (double)size / (1024.0 * 1024.0), threads);
    printf("  %-28s  %8s     %s         %s\n",
           "Kernel", "Avg ms", "Bandwidth", "Util");
    printf("  %s\n",
           "------------------------------------------------------------------");

    float ms_seq    = benchmark_1D("Sequential (float4)",
                                   sequential, d_a, d_c, N,
                                   blocks4, threads, 0, bytes_rw);
    float ms_stride = benchmark_strided("Strided x4 (float4 load)",
                                        strided, d_a, d_c, N, 4,
                                        blocks4, threads, bytes_rw);
    float ms_rand   = benchmark_random("Random (bitonic coalesce)",
                                       d_a, d_c, d_idx, N,
                                       blocks1, threads, bytes_rnd);

    printf("\n  Slowdown vs sequential:\n");
    printf("    Strided:       %.2fx\n", ms_stride / ms_seq);
    printf("    Random access: %.2fx\n", ms_rand   / ms_seq);

    cudaFree(d_a); cudaFree(d_c); cudaFree(d_idx);
    free(h_a); free(h_idx);
}

// ============================================================================
// MATMUL BENCHMARK HELPER
// ============================================================================
static float bench_matmul(const char* name,
                           void (*kernel)(float*, float*, float*, int),
                           float *d_a, float *d_b, float *d_c,
                           dim3 grid, dim3 block, int n,
                           cudaEvent_t s, cudaEvent_t e)
{
    // warmup
    kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        kernel<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        total += ms;
    }

    float avg   = total / RUNS;
    double flops = 2.0 * (double)n * n * n;
    float gf    = (float)(flops / (avg * 1e-3) / 1e9);
    printf("  %-28s  %8.3f ms   %6.2f GFLOPS\n", name, avg, gf);
    return avg;
}

// ============================================================================
// MATMUL SECTION
// ============================================================================
void run_matmul()
{
    int n    = 2048;
    size_t sz = (size_t)n * n * sizeof(float);

    float *h_a = (float*)malloc(sz), *h_b = (float*)malloc(sz);
    for (int i = 0; i < n * n; i++) { h_a[i] = rand() % 100; h_b[i] = rand() % 100; }

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, sz)); CHECK(cudaMalloc(&d_b, sz)); CHECK(cudaMalloc(&d_c, sz));
    CHECK(cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice));

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // Naive: 16x16 block (256 threads, more blocks, higher occupancy)
    dim3 block_naive(16, 16);
    dim3 grid_naive(n / 16, n / 16);

    // Tiled v1 & v3: 32x32 block (matches TILE_DIM=32)
    dim3 block_tile(TILE_DIM, TILE_DIM);
    dim3 grid_tile(n / TILE_DIM, n / TILE_DIM);

    // Tiled v2 (thread coarsening WPT=4): RTS x RTS block
    dim3 block_v2(RTS, RTS);
    dim3 grid_v2(n / TILE_DIM, n / TILE_DIM);

    printf("\n  Matrix size: %dx%d\n", n, n);
    printf("  %-28s  %8s     %s\n", "Kernel", "Avg ms", "GFLOPS");
    printf("  %s\n", "------------------------------------------------------------------");

    float t_naive = bench_matmul("Naive (16x16, unroll8)",
                                  matmul_naive, d_a, d_b, d_c,
                                  grid_naive, block_naive, n, s, e);
    float t_tiled = bench_matmul("Tiled (32x32, bank-free)",
                                  matmul_tiled, d_a, d_b, d_c,
                                  grid_tile, block_tile, n, s, e);
    float t_v2    = bench_matmul("Thread coarsen (WPT=4, 8x8)",
                                  matmul_tiled_v2, d_a, d_b, d_c,
                                  grid_v2, block_v2, n, s, e);
    float t_v3    = bench_matmul("Double-buffered (32x32)",
                                  matmul_tiled_v3, d_a, d_b, d_c,
                                  grid_tile, block_tile, n, s, e);

    printf("\n  Speedup vs Naive:\n");
    printf("    Tiled:             %.2fx\n", t_naive / t_tiled);
    printf("    Thread coarsened:  %.2fx\n", t_naive / t_v2);
    printf("    Double-buffered:   %.2fx\n", t_naive / t_v3);

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b);
}

// ============================================================================
// MAIN
// ============================================================================
int main()
{
    srand((unsigned)time(NULL));

    printf("\n");
    printf("=========================================================================\n");
    printf("  CUDA Benchmark Suite — Optimized (float4 + bitonic + register tile)\n");
    printf("  Target: Jetson Orin Nano (sm_87, ~51.2 GB/s, ~1.6 TFLOPS FP32)\n");
    printf("=========================================================================\n");

    printf("\n\n===== MEMORY ACCESS PATTERNS (Cache Hierarchy Sweep) =====\n");

    int sizes[] = {
        1 << 16,   //  64K =  256 KB
        1 << 18,   // 256K =    1 MB
        1 << 20,   //   1M =    4 MB
        1 << 22,   //   4M =   16 MB
        1 << 24,   //  16M =   64 MB
        1 << 26,   //  64M =  256 MB
    };

    for (int i = 0; i < (int)(sizeof(sizes)/sizeof(sizes[0])); i++)
        run_memory_size(sizes[i]);

    printf("\n\n===== MATRIX MULTIPLICATION =====\n");
    run_matmul();

    printf("\n=========================================================================\n");
    printf("  Benchmark Complete\n");
    printf("=========================================================================\n\n");

    return 0;
}