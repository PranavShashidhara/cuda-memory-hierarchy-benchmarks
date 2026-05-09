// cublass_and_cutlass/cublas_baseline.cu
// cuBLAS GEMM baseline — hardware-ceiling reference for the GEMM ladder.
//
// Purpose
// ───────
// cuBLAS sgemm (fp32) and cublasGemmEx (fp16 TC) establish the upper bound
// that hand-written kernels are measured against.  Results are reported in the
// same format as bench_fp32 / bench_tc in main.cu and written to the shared
// CSV so plot_results.py can include them in the roofline chart.
//
// API surface used
// ────────────────
//   cublasCreate / cublasDestroy               — handle lifecycle
//   cublasSgemm                                — fp32 GEMM  (CUBLAS_OP_N, row-major
//                                                via the standard A^T B^T = (BA)^T
//                                                transposition trick)
//   cublasGemmEx + CUBLAS_COMPUTE_32F_FAST_16F — fp16 inputs, fp32 accumulate via TC
//
// Jetson Orin (sm_87, iGPU / unified memory) note
// ─────────────────────────────────────────────────
// CUBLAS_COMPUTE_16F (pure fp16 accumulation) is NOT supported on sm_87.
// The correct compute type for Tensor Core fp16→fp32 on Ampere iGPU is:
//   CUBLAS_COMPUTE_32F_FAST_16F
// This uses fp16 multiply + fp32 accumulate, matching what WMMA does.
//
// Build note
// ──────────
// Link with: -lcublas
// No extra headers beyond cuda_runtime.h and cublas_v2.h are required.
//
// Target: Jetson Orin Nano (sm_87, ~51.2 GB/s mem BW,
//         ~1.6 TFLOPS fp32, ~5.2 TFLOPS fp16 TC)

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "utils.cuh"   // CHECK macro

// ─── Benchmark config (must match main.cu) ────────────────────────────────────
#define RUNS          10
#define MATRIX_N    2048

// ─── Hardware reference values (Jetson Orin Nano 8 GB) ────────────────────────
#define PEAK_FP32   1600.0f    // GFLOPS
#define PEAK_TC_FP16 5200.0f   // GFLOPS (fp16 accumulate)
#define CSV_PATH "cublass_and_cutlass/results/benchmark_results_cublas.csv"

// ─── cuBLAS error-check macro ─────────────────────────────────────────────────
#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t _st = (call);                                            \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                     \
            printf("cuBLAS Error at %s:%d — status %d\n",                      \
                   __FILE__, __LINE__, (int)_st);                               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static FILE *g_csv = NULL;

static void csv_init() {
    g_csv = fopen(CSV_PATH, "w");
    if (!g_csv) {
        fprintf(stderr, "[WARN] Cannot open %s\n", CSV_PATH);
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

// ─── fp32 → fp16 conversion kernel (same as wmma_matmul_kernels.cu) ──────────
__global__ static void float2half_kernel(const float * __restrict__ src,
                                          __half      * __restrict__ dst,
                                          int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __float2half(src[i]);
}

static void convert_to_half_local(const float *d_fp32, __half *d_fp16, int n)
{
    const int total   = n * n;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    float2half_kernel<<<blocks, threads>>>(d_fp32, d_fp16, total);
    CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// bench_cublas_sgemm
//
// Measures cublasSgemm (fp32).
//
// cuBLAS assumes column-major storage.  For row-major matrices A, B (as used
// throughout this project) we exploit the identity:
//
//     C = A  × B   (row-major)
//   ↔ C^T = B^T × A^T   (column-major, swapped operands)
//
// By passing (B, A) instead of (A, B) and keeping CUBLAS_OP_N for both,
// cuBLAS computes the correct row-major result into C with zero data copies.
//
// Returns: average time in ms.
// ─────────────────────────────────────────────────────────────────────────────
static float bench_cublas_sgemm(cublasHandle_t handle,
                                  float *d_a, float *d_b, float *d_c,
                                  int n, float naive_ms)
{
    const double flops = 2.0 * (double)n * n * n;
    const float  alpha = 1.f, beta = 0.f;

    // Warmup
    CHECK_CUBLAS(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_b, n,   // B is "A" in col-major view
        d_a, n,   // A is "B" in col-major view
        &beta,
        d_c, n));
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));

    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        CHECK(cudaEventRecord(s));
        CHECK_CUBLAS(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &alpha, d_b, n, d_a, n, &beta, d_c, n));
        CHECK(cudaEventRecord(e));
        CHECK(cudaEventSynchronize(e));
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, s, e));
        total += ms;
    }
    float avg = total / RUNS;
    float gf  = (float)(flops / (avg * 1e-3) / 1e9);

    if (naive_ms > 0.f) {
    printf("  %-36s  %8.3f ms   %7.2f GFLOPS  (%5.1f%% fp32 peak) "
           "[%.2f× naive]\n",
           "cuBLAS Sgemm (fp32)", avg, gf,
           gf / PEAK_FP32 * 100.f, naive_ms / avg);
    } else {
        printf("  %-36s  %8.3f ms   %7.2f GFLOPS\n",
            "cuBLAS Sgemm (fp32)", avg, gf);
    }

    csv_write("matmul", "cublas_sgemm", n, avg, gf, "GFLOPS");

    CHECK(cudaEventDestroy(s));
    CHECK(cudaEventDestroy(e));
    return avg;
}

// ─────────────────────────────────────────────────────────────────────────────
// bench_cublas_gemmex_fp16
//
// Measures cublasGemmEx with:
//   CUDA_R_16F inputs + CUDA_R_32F output + CUBLAS_COMPUTE_32F_FAST_16F
//
// CUBLAS_COMPUTE_32F_FAST_16F  — fp16 multiply, fp32 accumulate via Tensor Cores.
// This is the correct mode for sm_87 (Jetson Orin, Ampere iGPU).
// CUBLAS_COMPUTE_16F (pure fp16 accumulate) returns status 15 (NOT_SUPPORTED)
// on integrated Ampere GPUs — do not use it here.
//
// Same column-major swap trick as sgemm above.
// Returns: average time in ms.
// ─────────────────────────────────────────────────────────────────────────────
static float bench_cublas_gemmex_fp16(cublasHandle_t handle,
                                        const __half *d_a16,
                                        const __half *d_b16,
                                        float        *d_c,
                                        int n, float naive_ms)
{
    const double flops = 2.0 * (double)n * n * n;
    const float  alpha = 1.f, beta = 0.f;

    // Warmup
    CHECK_CUBLAS(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_b16, CUDA_R_16F, n,   // B (col-major "A")
        d_a16, CUDA_R_16F, n,   // A (col-major "B")
        &beta,
        d_c,   CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_16F,   // fp16 TC, fp32 accumulate — works on sm_87
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));

    float total = 0.f;
    for (int i = 0; i < RUNS; i++) {
        CHECK(cudaEventRecord(s));
        CHECK_CUBLAS(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &alpha,
            d_b16, CUDA_R_16F, n,
            d_a16, CUDA_R_16F, n,
            &beta,
            d_c,   CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK(cudaEventRecord(e));
        CHECK(cudaEventSynchronize(e));
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, s, e));
        total += ms;
    }
    float avg = total / RUNS;
    float gf  = (float)(flops / (avg * 1e-3) / 1e9);

    if (naive_ms > 0.f) {
    printf("  %-36s  %8.3f ms   %7.2f GFLOPS  (%5.1f%% TC fp16 peak) "
           "[%.2f× naive]\n",
           "cuBLAS GemmEx (fp16 TC)", avg, gf,
           gf / PEAK_TC_FP16 * 100.f, naive_ms / avg);
    } else {
        printf("  %-36s  %8.3f ms   %7.2f GFLOPS\n",
            "cuBLAS GemmEx (fp16 TC)", avg, gf);
    }

    csv_write("matmul", "cublas_gemmex_fp16", n, avg, gf, "GFLOPS");

    CHECK(cudaEventDestroy(s));
    CHECK(cudaEventDestroy(e));
    return avg;
}

// ─────────────────────────────────────────────────────────────────────────────
// run_cublas_baseline
//
// Entry point — call this from main.cu after run_matmul(), passing:
//   naive_ms  — timing from bench_fp32("Naive ...") so speedup column matches
//   csv       — the already-open FILE* used by the rest of the benchmark
// ─────────────────────────────────────────────────────────────────────────────
void run_cublas_baseline(float naive_ms)
{   
    const int    n       = MATRIX_N;
    const size_t sz_fp32 = (size_t)n * n * sizeof(float);
    const size_t sz_fp16 = (size_t)n * n * sizeof(__half);

    // ── Host fill ─────────────────────────────────────────────────────────────
    float *h_a = (float*)malloc(sz_fp32);
    float *h_b = (float*)malloc(sz_fp32);
    for (int i = 0; i < n * n; i++) {
        h_a[i] = (float)(rand() % 10);
        h_b[i] = (float)(rand() % 10);
    }

    // ── Device fp32 ───────────────────────────────────────────────────────────
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, sz_fp32));
    CHECK(cudaMalloc(&d_b, sz_fp32));
    CHECK(cudaMalloc(&d_c, sz_fp32));
    CHECK(cudaMemcpy(d_a, h_a, sz_fp32, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sz_fp32, cudaMemcpyHostToDevice));

    // ── Device fp16 ───────────────────────────────────────────────────────────
    __half *d_a16, *d_b16;
    CHECK(cudaMalloc(&d_a16, sz_fp16));
    CHECK(cudaMalloc(&d_b16, sz_fp16));
    convert_to_half_local(d_a, d_a16, n);
    convert_to_half_local(d_b, d_b16, n);
    CHECK(cudaDeviceSynchronize());

    // ── cuBLAS handle ─────────────────────────────────────────────────────────
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Enable TF32 on Ampere (no-op on chips that don't support it)
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    
    if (!g_csv) csv_init();

    // ── Run benchmarks ────────────────────────────────────────────────────────
    printf("\n  -- cuBLAS hardware ceiling (n=%d) --\n", n);
    printf("  %-36s  %8s   %14s  %s\n",
           "Kernel", "Avg ms", "Throughput", "vs peak");
    printf("  %s\n",
           "------------------------------------------------------------------------"
           "----");

    float t_sgemm = bench_cublas_sgemm(handle, d_a, d_b, d_c,
                                        n, naive_ms);
    float t_fp16  = bench_cublas_gemmex_fp16(handle, d_a16, d_b16, d_c,
                                              n, naive_ms);

    printf("\n  cuBLAS GemmEx fp16 vs cuBLAS Sgemm fp32:  %.2f×\n",
           t_sgemm / t_fp16);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK(cudaFree(d_a));   CHECK(cudaFree(d_b));   CHECK(cudaFree(d_c));
    CHECK(cudaFree(d_a16)); CHECK(cudaFree(d_b16));
    free(h_a); free(h_b);
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone main (compile with -DCUBLAS_STANDALONE to run independently)
// Remove or ifdef-out when integrating with the main benchmark binary.
// ─────────────────────────────────────────────────────────────────────────────
#ifdef CUBLAS_STANDALONE
int main()
{
    srand((unsigned)time(NULL));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("\n");
    printf("=========================================================================\n");
    printf("  cuBLAS Baseline — hardware GEMM ceiling\n");
    printf("  Device : %s  (sm_%d%d)\n",
           prop.name, prop.major, prop.minor);
    printf("=========================================================================\n");

    // naive_ms = 0 → speedup column will show 0× (not meaningful standalone)
    run_cublas_baseline(/*naive_ms=*/0.f);

    printf("\n=========================================================================\n");
    printf("  Done.  Integrate run_cublas_baseline() into main.cu for full context.\n");
    printf("=========================================================================\n\n");

    if (g_csv) fclose(g_csv);

    return 0;
}
#endif // CUBLAS_STANDALONE