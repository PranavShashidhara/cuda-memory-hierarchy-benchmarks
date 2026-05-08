// src/main.cu
// CUDA Benchmark Suite — Jetson Orin Nano
// Phases 1–5: memory hierarchy + matmul progression + analysis engine
//
// Matmul ladder:
//   Naive (fp32, global)  →  Tiled (fp32, shared)
//   →  TC Basic (fp16, WMMA 64×64)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "kernels.cuh"
#include "utils.cuh"

#define RUNS      10
#define CSV_PATH  "baseline/results/benchmark_results.csv"

// ─────────────────────────────────────────────────────────────
//  CSV handle
// ─────────────────────────────────────────────────────────────
static FILE *g_csv = NULL;

static void csv_init() {
    g_csv = fopen(CSV_PATH, "w");
    if (!g_csv) {
        fprintf(stderr, "[WARN] Cannot open %s — CSV disabled\n", CSV_PATH);
        return;
    }
    fprintf(g_csv, "benchmark,variant,n_or_size,avg_ms,metric_value,metric_unit\n");
}

static void csv_write(const char *bench, const char *variant,
                      long long n, float avg_ms,
                      float metric, const char *unit) {
    if (g_csv)
        fprintf(g_csv, "%s,%s,%lld,%.6f,%.6f,%s\n",
                bench, variant, n, avg_ms, metric, unit);
}

// ─────────────────────────────────────────────────────────────
//  Bandwidth helper
// ─────────────────────────────────────────────────────────────
static inline float bandwidth_GBs(float ms, size_t bytes) {
    return (float)bytes / (ms * 1e-3f) / 1e9f;
}

// ─────────────────────────────────────────────────────────────
//  1D benchmark
// ─────────────────────────────────────────────────────────────
float benchmark_1D(const char *name,
                   void (*kernel)(float*, float*, int),
                   float *d_a, float *d_c,
                   int N, int blocks, int threads,
                   size_t bytes)
{
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    kernel<<<blocks, threads>>>(d_a, d_c, N);
    cudaDeviceSynchronize();
    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        kernel<<<blocks, threads>>>(d_a, d_c, N);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e); total += ms;
    }
    float avg = total / RUNS;
    float bw  = bandwidth_GBs(avg, bytes);
    printf("  %-26s  %8.3f ms   %8.2f GB/s\n", name, avg, bw);
    csv_write("memory", name, (long long)N, avg, bw, "GB/s");
    cudaEventDestroy(s); cudaEventDestroy(e);
    return avg;
}

// ─────────────────────────────────────────────────────────────
//  Strided benchmark
// ─────────────────────────────────────────────────────────────
float benchmark_strided(const char *name,
                        void (*kernel)(float*, float*, int, int),
                        float *d_a, float *d_c,
                        int N, int stride,
                        int blocks, int threads, size_t bytes)
{
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    kernel<<<blocks, threads>>>(d_a, d_c, stride, N);
    cudaDeviceSynchronize();
    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        kernel<<<blocks, threads>>>(d_a, d_c, stride, N);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e); total += ms;
    }
    float avg = total / RUNS;
    float bw  = bandwidth_GBs(avg, bytes);
    printf("  %-26s  %8.3f ms   %8.2f GB/s\n", name, avg, bw);
    csv_write("memory", name, (long long)N, avg, bw, "GB/s");
    cudaEventDestroy(s); cudaEventDestroy(e);
    return avg;
}

// ─────────────────────────────────────────────────────────────
//  Random access benchmark
// ─────────────────────────────────────────────────────────────
float benchmark_random(const char *name,
                       float *d_a, float *d_c, int *d_idx,
                       int N, int blocks, int threads, size_t bytes)
{
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    random_access<<<blocks, threads>>>(d_a, d_c, d_idx, N);
    cudaDeviceSynchronize();
    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        random_access<<<blocks, threads>>>(d_a, d_c, d_idx, N);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e); total += ms;
    }
    float avg = total / RUNS;
    float bw  = bandwidth_GBs(avg, bytes);
    printf("  %-26s  %8.3f ms   %8.2f GB/s\n", name, avg, bw);
    csv_write("memory", name, (long long)N, avg, bw, "GB/s");
    cudaEventDestroy(s); cudaEventDestroy(e);
    return avg;
}

// ─────────────────────────────────────────────────────────────
//  Memory sweep  (Phase 3)
// ─────────────────────────────────────────────────────────────
void run_memory_size(int N)
{
    size_t size = N * sizeof(float);
    float *h_a   = (float*)malloc(size);
    int   *h_idx = (int*)  malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_a[i] = (float)(rand() % 100);
    for (int i = 0; i < N; i++) h_idx[i] = i;
    for (int i = N-1; i > 0; i--) {
        int j = rand() % (i+1);
        int t = h_idx[i]; h_idx[i] = h_idx[j]; h_idx[j] = t;
    }

    float *d_a, *d_c; int *d_idx;
    CHECK(cudaMalloc(&d_a,   size));
    CHECK(cudaMalloc(&d_c,   size));
    CHECK(cudaMalloc(&d_idx, N * sizeof(int)));
    CHECK(cudaMemcpy(d_a,   h_a,   size,            cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_idx, h_idx, N * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256, blocks = (N + threads - 1) / threads;
    size_t bytes_rw  = 2 * size;
    size_t bytes_rnd = 2 * size + N * sizeof(int);

    printf("\n  N = %d  (~%.1f MB)\n", N, (double)size / (1<<20));
    printf("  %-26s  %8s   %10s\n", "Kernel", "Avg ms", "Bandwidth");
    printf("  %s\n", "----------------------------------------------------------------");

    float ms_seq    = benchmark_1D     ("Sequential",    sequential, d_a, d_c, N, blocks, threads, bytes_rw);
    float ms_stride = benchmark_strided("Strided (x4)",  strided,    d_a, d_c, N, 4, blocks, threads, bytes_rw);
    float ms_rand   = benchmark_random ("Random access", d_a, d_c, d_idx, N, blocks, threads, bytes_rnd);

    printf("  Strided slowdown vs sequential : %.2fx\n", ms_stride / ms_seq);
    printf("  Random  slowdown vs sequential : %.2fx\n", ms_rand   / ms_seq);

    cudaFree(d_a); cudaFree(d_c); cudaFree(d_idx);
    free(h_a); free(h_idx);
}

// ─────────────────────────────────────────────────────────────
//  Matmul helper: time a single kernel N times, return avg ms
// ─────────────────────────────────────────────────────────────
template<typename LaunchFn>
static float time_kernel(LaunchFn launch, cudaEvent_t s, cudaEvent_t e)
{
    // warmup
    launch();
    cudaDeviceSynchronize();

    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(s);
        launch();
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        total += ms;
    }
    return total / RUNS;
}

// ─────────────────────────────────────────────────────────────
//  Matrix multiply  (Phases 1/2/5)
//  Four-step ladder: naive → tiled → TC basic 
// ─────────────────────────────────────────────────────────────
void run_matmul()
{
    // n must be a multiple of 128 for the optimized WMMA kernel
    // (block tile = 128×128)
    int n = 2048;

    size_t sz32 = (size_t)n * n * sizeof(float);
    size_t sz16 = (size_t)n * n * sizeof(__half);

    // ── Host data ──────────────────────────────────────────────
    float *h_a = (float*)malloc(sz32);
    float *h_b = (float*)malloc(sz32);
    for (int i = 0; i < n*n; i++) { h_a[i] = rand()%100; h_b[i] = rand()%100; }

    // ── Device fp32 ────────────────────────────────────────────
    float *d_af, *d_bf, *d_cf;
    CHECK(cudaMalloc(&d_af, sz32));
    CHECK(cudaMalloc(&d_bf, sz32));
    CHECK(cudaMalloc(&d_cf, sz32));
    CHECK(cudaMemcpy(d_af, h_a, sz32, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bf, h_b, sz32, cudaMemcpyHostToDevice));

    // ── Device fp16 ────────────────────────────────────────────
    __half *d_ah, *d_bh;
    float  *d_ch_basic;
    CHECK(cudaMalloc(&d_ah,      sz16));
    CHECK(cudaMalloc(&d_bh,      sz16));
    CHECK(cudaMalloc(&d_ch_basic, sz32));
    convert_to_half(d_af, d_ah, n);
    convert_to_half(d_bf, d_bh, n);
    cudaDeviceSynchronize();

    // ── Launch configs ─────────────────────────────────────────
    // Naive / Tiled: 16×16 thread block
    dim3 blk16(16, 16);
    dim3 grd16(n/16, n/16);

    // TC Basic: 512 threads (16 warps, 4×4 layout), 64×64 block tile
    dim3 blkTCB(512);
    dim3 grdTCB((n+63)/64, (n+63)/64);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // ── Time each kernel ───────────────────────────────────────
    float avg_naive = time_kernel([&]{ matmul_naive<<<grd16, blk16>>>(d_af, d_bf, d_cf, n); }, s, e);
    float avg_tiled = time_kernel([&]{ matmul_tiled<<<grd16, blk16>>>(d_af, d_bf, d_cf, n); }, s, e);
    float avg_tcb   = time_kernel([&]{ matmul_wmma    <<<grdTCB, blkTCB>>>(d_ah, d_bh, d_ch_basic, n); }, s, e);

    // ── GFLOPS: 2×n³ ──────────────────────────────────────────
    double flops = 2.0 * (double)n * n * n;
    auto gf = [&](float ms) { return (float)(flops / (ms * 1e-3) / 1e9); };

    // ── Print ──────────────────────────────────────────────────
    printf("\n  Matrix size: %d × %d\n", n, n);
    printf("  %-34s  %8s   %10s\n", "Kernel", "Avg ms", "GFLOPS");
    printf("  %s\n", "--------------------------------------------------------------------");
    printf("  %-34s  %8.3f ms  %8.2f GFLOPS\n", "1. Naive  (fp32, global)",          avg_naive, gf(avg_naive));
    printf("  %-34s  %8.3f ms  %8.2f GFLOPS\n", "2. Tiled  (fp32, shared 32×32)",    avg_tiled, gf(avg_tiled));
    printf("  %-34s  %8.3f ms  %8.2f GFLOPS\n", "3. TC Basic  (fp16, WMMA 64×64)",   avg_tcb,   gf(avg_tcb));

    printf("\n  --- Speedups vs Naive ---\n");
    printf("  Tiled        : %.2fx\n", avg_naive / avg_tiled);
    printf("  TC Basic     : %.2fx\n", avg_naive / avg_tcb);


    // ── CSV ────────────────────────────────────────────────────
    csv_write("matmul", "naive",        (long long)n, avg_naive, gf(avg_naive), "GFLOPS");
    csv_write("matmul", "tiled",        (long long)n, avg_tiled, gf(avg_tiled), "GFLOPS");
    csv_write("matmul", "tc_basic",     (long long)n, avg_tcb,   gf(avg_tcb),   "GFLOPS");

    // ── Cleanup ────────────────────────────────────────────────
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(d_af); cudaFree(d_bf); cudaFree(d_cf);
    cudaFree(d_ah); cudaFree(d_bh);
    free(h_a); free(h_b);
}

// ─────────────────────────────────────────────────────────────
//  Phase 5 summary
// ─────────────────────────────────────────────────────────────
void print_summary()
{
    printf("\n");
    printf("=========================================\n");
    printf("  Phase 5 — Comparative Analysis Summary \n");
    printf("=========================================\n");
    printf("\n");
    printf("  Memory Access Patterns\n");
    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  • Sequential (coalesced): warp accesses 128-byte-aligned\n");
    printf("    contiguous addresses → single memory transaction → peak BW.\n");
    printf("  • Strided (x4): each thread jumps 4 elements; consecutive\n");
    printf("    threads no longer map to the same cache line → 4× more\n");
    printf("    transactions, proportionally lower effective bandwidth.\n");
    printf("  • Random: near-zero cache hit rate; DRAM latency exposed\n");
    printf("    per access → bandwidth collapses to a few GB/s.\n");
    printf("\n");
    printf("  Matrix Multiplication Kernel Ladder\n");
    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  1. Naive:  every A/B element reloaded from DRAM per output.\n");
    printf("             Heavily memory-bound; low GFLOPS/W.\n");
    printf("  2. Tiled:  32×32 tiles in SRAM; each element reused 32×.\n");
    printf("             DRAM traffic drops ~32×; compute-bound regime.\n");
    printf("  3. TC Basic: WMMA 16×16×16 MMA instructions on Tensor Cores.\n");
    printf("             fp16 halves the bandwidth pressure vs fp32.\n");
    printf("             64×64 block tile; 1 fragment/warp.\n");
    printf("\n");
    printf("  CSV results : %s\n", CSV_PATH);
    printf("  Run plots   : python3 scripts/plot_results.py\n\n");
}

// ─────────────────────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────────────────────
int main()
{
    srand((unsigned)time(NULL));
    (void)system("mkdir -p results");
    csv_init();

    printf("=========================================\n");
    printf("  CUDA Benchmark Suite — Jetson Orin Nano\n");
    printf("=========================================\n");

    printf("\n\n===== PHASE 3 — MEMORY ACCESS BENCHMARK =====\n");
    int sizes[] = { 1<<16, 1<<18, 1<<20, 1<<22, 1<<24, 1<<26 };
    for (int i = 0; i < (int)(sizeof(sizes)/sizeof(sizes[0])); i++)
        run_memory_size(sizes[i]);

    printf("\n\n===== PHASES 1/2/5 — MATRIX MULTIPLY =====\n");
    run_matmul();

    print_summary();

    if (g_csv) fclose(g_csv);

    printf("=========================================\n");
    printf("  Done.\n");
    printf("=========================================\n");
    return 0;
}