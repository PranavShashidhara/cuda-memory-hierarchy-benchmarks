// Optimized/main.cu
// CUDA Benchmark: memory access patterns + matmul (fp32 ladder + Tensor Cores)
// Target: Jetson Orin Nano (sm_87, ~51.2 GB/s mem BW, ~1.6 TFLOPS fp32,
//                                   fp16 Tensor Cores via WMMA)
//
// Benchmark sections
// ──────────────────
//  A. Memory access patterns (sequential / strided×4 / random)
//     swept over 6 array sizes, cp.async pipeline throughout
//
//  B. Matrix multiply — 2048×2048
//      1. Naive fp32          (global memory, #pragma unroll 8)
//      2. Tiled fp32 v1       (shared mem, bank-conflict-free +1 pad)
//      3. Tiled fp32 v2       (thread coarsening, WPT=4 register blocking)
//      4. Tiled fp32 v3       (cp.async double-buffer + register blocking)
//      5. TC WMMA basic       (fp16→fp32, 64×64 block tile, single-buffer)
//      6. TC WMMA optimised   (fp16→fp32, 128×128 block tile, double-buffer,
//                              +8-half smem padding, 8 MMA frags/warp)
//
// CSV output  →  results/benchmark_results.csv  (consumed by plot_results.py)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <climits>
#include <functional>
#include <sys/stat.h>

#include "kernels.cuh"
#include "utils.cuh"

// ─── Benchmark config ─────────────────────────────────────────────────────────
#define RUNS        10
#define PIPE_DEPTH   4          // must match memory_kernels.cu

// ─── Hardware reference values (Jetson Orin Nano 8 GB) ────────────────────────
#define PEAK_MEM_BW  51.2f      // GB/s  (LPDDR5)
#define PEAK_FP32    1600.0f    // GFLOPS
#define PEAK_TC_FP16 40000.0f   // GFLOPS (fp16 accumulate, from spec sheet)

// ─── Global CSV handle ────────────────────────────────────────────────────────
static FILE *g_csv = nullptr;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static inline float bandwidth_GBs(float ms, size_t bytes) {
    return (float)bytes / (ms * 1e-3f) / 1e9f;
}

static inline void csv_write_memory(const char *variant, int n_elem,
                                    float avg_ms, float bw_GBs)
{
    if (g_csv)
        fprintf(g_csv, "memory,%s,%d,%.4f,%.4f\n",
                variant, n_elem, avg_ms, bw_GBs);
}

static inline void csv_write_matmul(const char *variant, int n,
                                    float avg_ms, float gflops)
{
    if (g_csv)
        fprintf(g_csv, "matmul,%s,%d,%.4f,%.4f\n",
                variant, n, avg_ms, gflops);
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic timed launch (std::function, works for lambdas with any capture)
// ─────────────────────────────────────────────────────────────────────────────
static float timed_run(const char *name,
                       std::function<void()> launch,
                       size_t bytes_moved, float peak_bw,
                       bool print_bw = true)
{
    // warmup
    launch(); CHECK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));

    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        CHECK(cudaEventRecord(s));
        launch();
        CHECK(cudaEventRecord(e));
        CHECK(cudaEventSynchronize(e));
        float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, s, e));
        total += ms;
    }
    float avg = total / RUNS;

    if (print_bw) {
        float bw = bandwidth_GBs(avg, bytes_moved);
        printf("  %-36s  %8.3f ms   %6.2f GB/s  (%5.1f%% peak)\n",
               name, avg, bw, bw / peak_bw * 100.f);
    }
    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    return avg;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section A: Memory access benchmarks
// ─────────────────────────────────────────────────────────────────────────────
static void run_memory_size(int N)
{
    size_t size = (size_t)N * sizeof(float);

    float *h_a   = (float*)malloc(size);
    int   *h_idx = (int*)  malloc((size_t)N * sizeof(int));

    for (int i = 0; i < N; i++) h_a[i]   = (float)(rand() % 100);
    for (int i = 0; i < N; i++) h_idx[i] = i;
    // Fisher-Yates shuffle to build a random permutation
    for (int i = N-1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = h_idx[i]; h_idx[i] = h_idx[j]; h_idx[j] = t;
    }

    float *d_a, *d_c; int *d_idx;
    CHECK(cudaMalloc(&d_a,   size));
    CHECK(cudaMalloc(&d_c,   size));
    CHECK(cudaMalloc(&d_idx, (size_t)N * sizeof(int)));
    CHECK(cudaMemcpy(d_a,   h_a,   size,                  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_idx, h_idx, (size_t)N*sizeof(int), cudaMemcpyHostToDevice));

    const int threads     = 256;
    const int blocks_pipe = (N + threads * PIPE_DEPTH - 1) / (threads * PIPE_DEPTH);
    const int blocks_rnd  = (N + threads - 1) / threads;
    const size_t smem_pipe = (size_t)threads * PIPE_DEPTH * sizeof(float);
    const size_t smem_rnd  = (size_t)threads * sizeof(int);
    const size_t bytes_rw  = 2 * size;
    const size_t bytes_rnd = 2 * size + (size_t)N * sizeof(int);

    printf("\n  N = %-10d (~%4.0f MB)  [threads/block=%d, pipe_depth=%d]\n",
           N, (double)size / (1024.0*1024.0), threads, PIPE_DEPTH);
    printf("  %-36s  %8s     %-12s  %s\n",
           "Kernel", "Avg ms", "Bandwidth", "Util");
    printf("  %s\n",
           "------------------------------------------------------------------------");

    float ms_seq = timed_run("Sequential (cp.async ×4)",
        [&]{ sequential<<<blocks_pipe, threads, smem_pipe>>>(d_a, d_c, N); },
        bytes_rw, PEAK_MEM_BW);
    csv_write_memory("Sequential", N, ms_seq, bandwidth_GBs(ms_seq, bytes_rw));

    float ms_str = timed_run("Strided ×4 (cp.async)",
        [&]{ strided<<<blocks_pipe, threads, smem_pipe>>>(d_a, d_c, 4, N); },
        bytes_rw, PEAK_MEM_BW);
    csv_write_memory("Strided (x4)", N, ms_str, bandwidth_GBs(ms_str, bytes_rw));

    float ms_rnd = timed_run("Random (bitonic sort + cp.async)",
        [&]{ random_access<<<blocks_rnd, threads, smem_rnd>>>(d_a, d_c, d_idx, N); },
        bytes_rnd, PEAK_MEM_BW);
    csv_write_memory("Random access", N, ms_rnd, bandwidth_GBs(ms_rnd, bytes_rnd));

    printf("\n  Slowdown vs sequential:\n");
    printf("    Strided ×4   : %.2f×\n", ms_str / ms_seq);
    printf("    Random access: %.2f×\n", ms_rnd / ms_seq);

    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_c)); CHECK(cudaFree(d_idx));
    free(h_a); free(h_idx);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section B: Matrix multiply benchmarks
// ─────────────────────────────────────────────────────────────────────────────

// ── fp32 kernel wrapper (uniform signature) ──────────────────────────────────
static float bench_fp32(const char *name,
                         void (*kernel)(float*, float*, float*, int),
                         float *d_a, float *d_b, float *d_c,
                         dim3 grid, dim3 block, int n,
                         float naive_ms = 0.f)
{
    const double flops = 2.0 * (double)n * n * n;

    // warmup
    kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));

    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        CHECK(cudaEventRecord(s));
        kernel<<<grid, block>>>(d_a, d_b, d_c, n);
        CHECK(cudaEventRecord(e));
        CHECK(cudaEventSynchronize(e));
        float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, s, e));
        total += ms;
    }
    float avg = total / RUNS;
    float gf  = (float)(flops / (avg * 1e-3) / 1e9);

    if (naive_ms > 0.f)
        printf("  %-36s  %8.3f ms   %7.2f GFLOPS  (%5.1f%% fp32 peak)  "
               "[%.2f× naive]\n",
               name, avg, gf, gf / PEAK_FP32 * 100.f, naive_ms / avg);
    else
        printf("  %-36s  %8.3f ms   %7.2f GFLOPS  (%5.1f%% fp32 peak)\n",
               name, avg, gf, gf / PEAK_FP32 * 100.f);

    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    return avg;
}

// ── TC (WMMA) kernel wrapper ──────────────────────────────────────────────────
static float bench_tc(const char *name,
                       void (*kernel)(const __half*, const __half*, float*, int),
                       const __half *d_a16, const __half *d_b16, float *d_c,
                       dim3 grid, dim3 block, int n,
                       float naive_ms)
{
    const double flops = 2.0 * (double)n * n * n;

    // warmup
    kernel<<<grid, block>>>(d_a16, d_b16, d_c, n);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));

    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        CHECK(cudaEventRecord(s));
        kernel<<<grid, block>>>(d_a16, d_b16, d_c, n);
        CHECK(cudaEventRecord(e));
        CHECK(cudaEventSynchronize(e));
        float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, s, e));
        total += ms;
    }
    float avg = total / RUNS;
    float gf  = (float)(flops / (avg * 1e-3) / 1e9);

    printf("  %-36s  %8.3f ms   %7.2f GFLOPS  (%5.1f%% TC fp16 peak) "
           "[%.2f× naive]\n",
           name, avg, gf, gf / PEAK_TC_FP16 * 100.f, naive_ms / avg);

    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    return avg;
}

// ── Main matmul section ───────────────────────────────────────────────────────
static void run_matmul()
{
    const int n   = 2048;
    const size_t sz_fp32 = (size_t)n * n * sizeof(float);
    const size_t sz_fp16 = (size_t)n * n * sizeof(__half);

    // ── Allocate & fill host buffers ─────────────────────────────────────────
    float *h_a = (float*)malloc(sz_fp32);
    float *h_b = (float*)malloc(sz_fp32);
    for (int i = 0; i < n * n; i++) {
        h_a[i] = (float)(rand() % 10);   // small ints → stable fp16 repr
        h_b[i] = (float)(rand() % 10);
    }

    // ── Device buffers — fp32 ─────────────────────────────────────────────────
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, sz_fp32));
    CHECK(cudaMalloc(&d_b, sz_fp32));
    CHECK(cudaMalloc(&d_c, sz_fp32));
    CHECK(cudaMemcpy(d_a, h_a, sz_fp32, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sz_fp32, cudaMemcpyHostToDevice));

    // ── Device buffers — fp16 (converted from fp32 in-place on device) ────────
    __half *d_a16, *d_b16;
    CHECK(cudaMalloc(&d_a16, sz_fp16));
    CHECK(cudaMalloc(&d_b16, sz_fp16));
    convert_to_half(d_a, d_a16, n);
    convert_to_half(d_b, d_b16, n);
    CHECK(cudaDeviceSynchronize());

    // ── Launch configs ────────────────────────────────────────────────────────
    // fp32 naive (16×16 blocks)
    dim3 blk_naive(16, 16),   grd_naive(n/16, n/16);
    // fp32 tiled v1 (32×32 blocks)
    dim3 blk_tile(TILE_DIM, TILE_DIM), grd_tile(n/TILE_DIM, n/TILE_DIM);
    // fp32 v2 / v3 (8×8 blocks, RTS=TILE_DIM/WPT)
    dim3 blk_v2(RTS, RTS),   grd_v2(n/TILE_DIM, n/TILE_DIM);

    // TC basic: 512 threads/block (16 warps), grid = (n/64)×(n/64)
    const int BM_B = 64, BN_B = 64;
    dim3 blk_tc_b(512),            grd_tc_b(n/BN_B, n/BM_B);
    // TC optimised: 256 threads/block (8 warps), grid = (n/128)×(n/128)
    const int BM_O = 128, BN_O = 128;
    dim3 blk_tc_o(256),            grd_tc_o(n/BN_O, n/BM_O);

    // ── Print header ──────────────────────────────────────────────────────────
    printf("\n  Matrix size: %d×%d  (fp32 inputs for fp32 kernels; "
           "fp16 inputs for TC kernels)\n", n, n);
    printf("  %-36s  %8s   %14s  %s\n",
           "Kernel", "Avg ms", "Throughput", "vs fp32 peak");
    printf("  %s\n",
           "------------------------------------------------------------------------"
           "----");

    // ── fp32 ladder ───────────────────────────────────────────────────────────
    float t_naive = bench_fp32("Naive fp32  (16×16, unroll8)",
                                matmul_naive, d_a, d_b, d_c,
                                grd_naive, blk_naive, n);

    float t_tiled = bench_fp32("Tiled fp32 v1 (32×32, pad+1)",
                                matmul_tiled, d_a, d_b, d_c,
                                grd_tile, blk_tile, n, t_naive);

    float t_v2    = bench_fp32("Tiled fp32 v2 (WPT=4, 8×8)",
                                matmul_tiled_v2, d_a, d_b, d_c,
                                grd_v2, blk_v2, n, t_naive);

    float t_v3    = bench_fp32("Tiled fp32 v3 (cp.async+WPT=4)",
                                matmul_tiled_v3, d_a, d_b, d_c,
                                grd_v2, blk_v2, n, t_naive);

    // ── Tensor Core kernels ───────────────────────────────────────────────────
    printf("\n  -- Tensor Core (fp16 input, fp32 accumulate) --\n");

    float t_tc_b  = bench_tc  ("TC WMMA basic  (64×64, 1 frag/warp)",
                                matmul_wmma, d_a16, d_b16, d_c,
                                grd_tc_b, blk_tc_b, n, t_naive);

    float t_tc_o  = bench_tc  ("TC WMMA opt    (128×128, dbl-buf, pad)",
                                matmul_wmma_opt, d_a16, d_b16, d_c,
                                grd_tc_o, blk_tc_o, n, t_naive);

    // ── Summary table ─────────────────────────────────────────────────────────
    printf("\n  ── Speedup ladder vs Naive fp32 ──\n");
    printf("  %-36s  %.2f×\n", "Tiled v1",               t_naive / t_tiled);
    printf("  %-36s  %.2f×\n", "Tiled v2 (WPT=4)",       t_naive / t_v2);
    printf("  %-36s  %.2f×\n", "Tiled v3 (cp.async)",    t_naive / t_v3);
    printf("  %-36s  %.2f×  ← Tensor Core basic\n",
           "TC WMMA basic",  t_naive / t_tc_b);
    printf("  %-36s  %.2f×  ← Tensor Core optimised\n",
           "TC WMMA optimised", t_naive / t_tc_o);
    printf("\n  TC optimised vs best fp32 (v3):  %.2f×\n",
           t_v3 / t_tc_o);

    // ── Write CSV ─────────────────────────────────────────────────────────────
    const double flops = 2.0 * (double)n * n * n;
    auto gf = [&](float ms){ return (float)(flops / (ms*1e-3) / 1e9); };
    csv_write_matmul("naive",        n, t_naive, gf(t_naive));
    csv_write_matmul("tiled",        n, t_tiled, gf(t_tiled));
    csv_write_matmul("tiled_v2",     n, t_v2,    gf(t_v2));
    csv_write_matmul("tiled_v3",     n, t_v3,    gf(t_v3));
    csv_write_matmul("tc_basic",     n, t_tc_b,  gf(t_tc_b));
    csv_write_matmul("tc_optimized", n, t_tc_o,  gf(t_tc_o));

    // ── Cleanup ───────────────────────────────────────────────────────────────
    CHECK(cudaFree(d_a));   CHECK(cudaFree(d_b));   CHECK(cudaFree(d_c));
    CHECK(cudaFree(d_a16)); CHECK(cudaFree(d_b16));
    free(h_a); free(h_b);
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    srand((unsigned)time(NULL));

    // ── GPU info ──────────────────────────────────────────────────────────────
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("\n");
    printf("=========================================================================\n");
    printf("  CUDA Benchmark — fp32 ladder + Tensor Core WMMA\n");
    printf("  Device : %s  (sm_%d%d)\n",
           prop.name, prop.major, prop.minor);
    printf("  SMs    : %d   Warp size : %d   Max threads/SM : %d\n",
           prop.multiProcessorCount, prop.warpSize,
           prop.maxThreadsPerMultiProcessor);
    printf("  Shared mem/block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("=========================================================================\n");

    // ── Open CSV ──────────────────────────────────────────────────────────────
    mkdir("results", 0755);
    g_csv = fopen("results/benchmark_results.csv", "w");
    if (g_csv) {
        fprintf(g_csv, "benchmark,variant,n_or_size,avg_ms,metric_value\n");
        printf("\nCSV output → results/benchmark_results.csv\n");
    } else {
        fprintf(stderr, "[WARN] Cannot open results/benchmark_results.csv"
                        " — continuing without CSV.\n");
    }

    // ── Section A: memory ─────────────────────────────────────────────────────
    printf("\n\n===== A. MEMORY ACCESS PATTERNS =====\n");
    int mem_sizes[] = { 1<<16, 1<<18, 1<<20, 1<<22, 1<<24, 1<<26 };
    for (int i = 0; i < (int)(sizeof(mem_sizes)/sizeof(mem_sizes[0])); i++)
        run_memory_size(mem_sizes[i]);

    // ── Section B: matmul ─────────────────────────────────────────────────────
    printf("\n\n===== B. MATRIX MULTIPLICATION =====\n");
    run_matmul();

    // ── Done ──────────────────────────────────────────────────────────────────
    if (g_csv) fclose(g_csv);

    printf("\n=========================================================================\n");
    printf("  Benchmark complete.  Run  python3 scripts/plot_results.py\n");
    printf("  to generate the comparison charts.\n");
    printf("=========================================================================\n\n");
    return 0;
}
