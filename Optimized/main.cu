#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <climits>

#include "kernels.cuh"
#include "utils.cuh"

#define RUNS       10
#define PIPE_DEPTH  4   // must match memory_kernels.cu

static inline float bandwidth_GBs(float ms, size_t bytes) {
    return (float)bytes / (ms * 1e-3f) / 1e9f;
}
#define PEAK_MEM_BW 51.2f

// ============================================================================
// GENERIC TIMED LAUNCH — works for any kernel + smem size
// ============================================================================
static float timed_runs(const char *name,
                        std::function<void()> launch,
                        size_t bytes, float peak_bw)
{
    // warmup
    launch(); cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        launch();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        total += ms;
    }

    float avg = total / RUNS;
    float bw  = bandwidth_GBs(avg, bytes);
    printf("  %-32s  %8.3f ms   %6.2f GB/s  (%5.1f%% peak)\n",
           name, avg, bw, bw / peak_bw * 100.f);

    cudaEventDestroy(s); cudaEventDestroy(e);
    return avg;
}

// ============================================================================
// MEMORY SIZE BENCHMARK
// ============================================================================
void run_memory_size(int N)
{
    size_t size = (size_t)N * sizeof(float);

    float *h_a   = (float*)malloc(size);
    int   *h_idx = (int*)  malloc((size_t)N * sizeof(int));

    for (int i = 0; i < N; i++) h_a[i] = (float)(rand() % 100);
    for (int i = 0; i < N; i++) h_idx[i] = i;
    for (int i = N-1; i > 0; i--) {
        int j = rand() % (i+1);
        int t = h_idx[i]; h_idx[i] = h_idx[j]; h_idx[j] = t;
    }

    float *d_a, *d_c; int *d_idx;
    CHECK(cudaMalloc(&d_a,   size));
    CHECK(cudaMalloc(&d_c,   size));
    CHECK(cudaMalloc(&d_idx, (size_t)N * sizeof(int)));
    CHECK(cudaMemcpy(d_a,   h_a,   size,                   cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_idx, h_idx, (size_t)N*sizeof(int),  cudaMemcpyHostToDevice));

    int threads = 256;

    // sequential / strided: each block processes PIPE_DEPTH * threads elements
    int blocks_pipe = (N + threads * PIPE_DEPTH - 1) / (threads * PIPE_DEPTH);
    size_t smem_pipe = (size_t)threads * PIPE_DEPTH * sizeof(float);

    // random_access: one thread per element, smem = one int per thread
    int blocks_rnd  = (N + threads - 1) / threads;
    size_t smem_rnd = (size_t)threads * sizeof(int);

    size_t bytes_rw  = 2 * size;
    size_t bytes_rnd = 2 * size + (size_t)N * sizeof(int);

    printf("\n  N = %d  (~%.0f MB)  [Threads/block: %d, pipe_depth: %d]\n",
           N, (double)size/(1024.0*1024.0), threads, PIPE_DEPTH);
    printf("  %-32s  %8s     %s         %s\n",
           "Kernel", "Avg ms", "Bandwidth", "Util");
    printf("  %s\n",
           "----------------------------------------------------------------------");

    float ms_seq = timed_runs("Sequential (cp.async x4)",
        [&]{ sequential<<<blocks_pipe, threads, smem_pipe>>>(d_a, d_c, N); },
        bytes_rw, PEAK_MEM_BW);

    float ms_str = timed_runs("Strided x4 (cp.async)",
        [&]{ strided<<<blocks_pipe, threads, smem_pipe>>>(d_a, d_c, 4, N); },
        bytes_rw, PEAK_MEM_BW);

    float ms_rnd = timed_runs("Random (bitonic+cp.async)",
        [&]{ random_access<<<blocks_rnd, threads, smem_rnd>>>(d_a, d_c, d_idx, N); },
        bytes_rnd, PEAK_MEM_BW);

    printf("\n  Slowdown vs sequential:\n");
    printf("    Strided:       %.2fx\n", ms_str / ms_seq);
    printf("    Random access: %.2fx\n", ms_rnd / ms_seq);

    cudaFree(d_a); cudaFree(d_c); cudaFree(d_idx);
    free(h_a); free(h_idx);
}

// ============================================================================
// MATMUL BENCHMARK
// ============================================================================
static float bench_matmul(const char *name,
                           void (*kernel)(float*, float*, float*, int),
                           float *d_a, float *d_b, float *d_c,
                           dim3 grid, dim3 block, int n)
{
    double flops = 2.0 * (double)n * n * n;

    // warmup
    kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        kernel<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        total += ms;
    }

    float avg = total / RUNS;
    float gf  = (float)(flops / (avg * 1e-3) / 1e9);
    printf("  %-32s  %8.3f ms   %6.2f GFLOPS\n", name, avg, gf);

    cudaEventDestroy(s); cudaEventDestroy(e);
    return avg;
}

void run_matmul()
{
    int n = 2048;
    size_t sz = (size_t)n * n * sizeof(float);

    float *h_a = (float*)malloc(sz), *h_b = (float*)malloc(sz);
    for (int i = 0; i < n*n; i++) { h_a[i] = rand()%100; h_b[i] = rand()%100; }

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, sz)); CHECK(cudaMalloc(&d_b, sz)); CHECK(cudaMalloc(&d_c, sz));
    CHECK(cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice));

    // Naive: 16x16 block — smaller block = more blocks = higher occupancy
    dim3 blk_naive(16, 16);
    dim3 grd_naive(n/16, n/16);

    // Tiled v1 & v3: TILE_DIM x TILE_DIM block (32x32 = 1024 threads)
    dim3 blk_tile(TILE_DIM, TILE_DIM);
    dim3 grd_tile(n/TILE_DIM, n/TILE_DIM);

    // Tiled v2 & v3 (thread coarsening WPT=4): RTS x RTS block (8x8 = 64 threads)
    // Fewer threads per block → more blocks in flight → better latency hiding
    dim3 blk_v2(RTS, RTS);
    dim3 grd_v2(n/TILE_DIM, n/TILE_DIM);

    printf("\n  Matrix size: %dx%d\n", n, n);
    printf("  %-32s  %8s     %s\n", "Kernel", "Avg ms", "GFLOPS");
    printf("  %s\n",
           "----------------------------------------------------------------------");

    float t_naive = bench_matmul("Naive (16x16, unroll8)",
                                  matmul_naive, d_a, d_b, d_c,
                                  grd_naive, blk_naive, n);
    float t_tiled = bench_matmul("Tiled (32x32, no-bankconflict)",
                                  matmul_tiled, d_a, d_b, d_c,
                                  grd_tile, blk_tile, n);
    float t_v2    = bench_matmul("Thread coarsen WPT=4 (8x8)",
                                  matmul_tiled_v2, d_a, d_b, d_c,
                                  grd_v2, blk_v2, n);
    float t_v3    = bench_matmul("cp.async pipeline WPT=4 (8x8)",
                                  matmul_tiled_v3, d_a, d_b, d_c,
                                  grd_v2, blk_v2, n);

    printf("\n  Speedup vs Naive:\n");
    printf("    Tiled v1:              %.2fx\n", t_naive / t_tiled);
    printf("    Thread coarsened v2:   %.2fx\n", t_naive / t_v2);
    printf("    cp.async pipeline v3:  %.2fx\n", t_naive / t_v3);

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
    printf("  CUDA Benchmark — cp.async pipeline + register blocking\n");
    printf("  Target: Jetson Orin Nano (sm_87, ~51.2 GB/s, ~1.6 TFLOPS FP32)\n");
    printf("=========================================================================\n");

    printf("\n\n===== MEMORY ACCESS PATTERNS =====\n");
    int sizes[] = { 1<<16, 1<<18, 1<<20, 1<<22, 1<<24, 1<<26 };
    for (int i = 0; i < (int)(sizeof(sizes)/sizeof(sizes[0])); i++)
        run_memory_size(sizes[i]);

    printf("\n\n===== MATRIX MULTIPLICATION =====\n");
    run_matmul();

    printf("\n=========================================================================\n");
    printf("  Benchmark Complete\n");
    printf("=========================================================================\n\n");
    return 0;
}
