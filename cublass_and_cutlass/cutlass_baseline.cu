// Optimized/cutlass_baseline.cu
// CUTLASS GEMM baseline — structured template library reference for the GEMM ladder.
//
// Purpose
// ───────
// CUTLASS provides hand-optimised GEMM implementations that sit between
// hand-written WMMA kernels and cuBLAS on the performance ladder.  This file
// benchmarks two CUTLASS kernels:
//
//   1. cutlass_sgemm_default   — fp32 GEMM, default (Simt) threadblock shape
//   2. cutlass_gemm_fp16_tc    — fp16 Tensor Core GEMM (Ampere sm_87)
//
// Both use the CUTLASS 3.x "device" API (cutlass::gemm::device::Gemm).
//
// API surface used
// ────────────────
//   cutlass::gemm::device::Gemm<...>    — templated GEMM operator
//   GemmOperator::Arguments             — problem size + pointer binding
//   GemmOperator::can_implement()       — runtime feasibility check
//   operator()                          — launches the kernel
//
// Build note
// ──────────
// Requires CUTLASS headers on the include path.  Typical setup:
//   git clone https://github.com/NVIDIA/cutlass  (v3.x tag)
//   nvcc ... -I/path/to/cutlass/include -I/path/to/cutlass/tools/util/include
// No additional link libraries needed (header-only on the CUDA side).
//
// CUTLASS column-major convention
// ────────────────────────────────
// CUTLASS defaults to column-major layout.  We match the rest of the project
// (row-major C arrays) by declaring both A and B as RowMajor in the template.
// CUTLASS then maps them correctly to the GPU without data transposition.
//
// Target: Jetson Orin Nano (sm_87, ~51.2 GB/s mem BW,
//         ~1.6 TFLOPS fp32, ~5.2 TFLOPS fp16 TC)
// cublass_and_cutlass/cutlass_baseline.cu
// CUTLASS GEMM baseline — structured reference implementation

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/numeric_types.h>

#include "utils.cuh"

// ─────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────
#define RUNS       10
#define MATRIX_N   2048

#define CSV_PATH "cublass_and_cutlass/results/benchmark_results_cutlass.csv"

// ─────────────────────────────────────────────────────────────
// CSV
// ─────────────────────────────────────────────────────────────
static FILE *g_csv = NULL;

static void csv_init() {

    g_csv = fopen(CSV_PATH, "w");
    if (!g_csv) {
        fprintf(stderr, "[WARN] Cannot open %s\n", CSV_PATH);
        return;
    }

    fprintf(g_csv,
        "benchmark,variant,n_or_size,avg_ms,metric_value,metric_unit\n");
}

static void csv_write(const char *bench, const char *variant,
                      long long n, float avg_ms,
                      float metric, const char *unit)
{
    if (!g_csv) return;

    fprintf(g_csv, "%s,%s,%lld,%.6f,%.6f,%s\n",
            bench, variant, n, avg_ms, metric, unit);
}

// ─────────────────────────────────────────────────────────────
// CUTLASS status checker (FIXED)
// ─────────────────────────────────────────────────────────────
#define CHECK_CUTLASS(status)                                  \
    do {                                                       \
        cutlass::Status _s = (status);                         \
        if (_s != cutlass::Status::kSuccess) {                 \
            printf("CUTLASS error at %s:%d\n",                 \
                   __FILE__, __LINE__);                        \
            exit(1);                                           \
        }                                                      \
    } while (0)

// ─────────────────────────────────────────────────────────────
// GEMM definitions
// ─────────────────────────────────────────────────────────────
using CutlassSgemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128,128,8>,
    cutlass::gemm::GemmShape<64,64,8>,
    cutlass::gemm::GemmShape<1,1,1>,
    cutlass::epilogue::thread::LinearCombination<float,1,float,float>
>;

using CutlassGemmFp16TC = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128,128,32>,
    cutlass::gemm::GemmShape<64,64,32>,
    cutlass::gemm::GemmShape<16,8,16>,
    cutlass::epilogue::thread::LinearCombination<float,4,float,float>
>;

// ─────────────────────────────────────────────────────────────
// FP32 → FP16 conversion
// ─────────────────────────────────────────────────────────────
__global__ void fp32_to_fp16(const float *src, cutlass::half_t *dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = cutlass::half_t(__float2half(src[i]));
}

static void convert(const float *d_in, cutlass::half_t *d_out, int n) {
    int total = n * n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fp32_to_fp16<<<blocks, threads>>>(d_in, d_out, total);
}

// ─────────────────────────────────────────────────────────────
// CUTLASS FP32
// ─────────────────────────────────────────────────────────────
static float bench_fp32(float *a, float *b, float *c, int n)
{
    cutlass::gemm::GemmCoord prob(n,n,n);
    float alpha=1.f, beta=0.f;

    CutlassSgemm gemm;

    CutlassSgemm::Arguments args(
        prob,
        {a,n},{b,n},{c,n},{c,n},
        {alpha,beta}
    );

    CHECK_CUTLASS(gemm.can_implement(args));

    size_t ws = CutlassSgemm::get_workspace_size(args);
    void *workspace = nullptr;
    if (ws) cudaMalloc(&workspace, ws);

    gemm(args, workspace);
    cudaDeviceSynchronize();

    cudaEvent_t s,e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    float total=0;

    for(int i=0;i<RUNS;i++){
        cudaEventRecord(s);
        gemm(args, workspace);
        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms;
        cudaEventElapsedTime(&ms,s,e);
        total += ms;
    }

    float avg = total / RUNS;
    float flops = 2.0*n*n*n;
    float gflops = flops / (avg*1e-3) / 1e9;

    printf("CUTLASS FP32: %.3f ms %.2f GFLOPS\n", avg, gflops);

    csv_write("matmul","cutlass_fp32",n,avg,gflops,"GFLOPS");

    if(workspace) cudaFree(workspace);
    return avg;
}

// ─────────────────────────────────────────────────────────────
// CUTLASS FP16 TC
// ─────────────────────────────────────────────────────────────
static float bench_fp16(cutlass::half_t *a,
                        cutlass::half_t *b,
                        float *c, int n)
{
    CutlassGemmFp16TC gemm;

    cutlass::gemm::GemmCoord prob(n,n,n);
    float alpha=1.f,beta=0.f;

    CutlassGemmFp16TC::Arguments args(
        prob,
        {a,n},{b,n},{c,n},{c,n},
        {alpha,beta}
    );

    CHECK_CUTLASS(gemm.can_implement(args));

    size_t ws = CutlassGemmFp16TC::get_workspace_size(args);
    void *workspace = nullptr;
    if(ws) cudaMalloc(&workspace, ws);

    cudaEvent_t s,e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    float total=0;

    for(int i=0;i<RUNS;i++){
        cudaEventRecord(s);
        gemm(args, workspace);
        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms;
        cudaEventElapsedTime(&ms,s,e);
        total += ms;
    }

    float avg = total / RUNS;
    float flops = 2.0*n*n*n;
    float gflops = flops / (avg*1e-3) / 1e9;

    printf("CUTLASS FP16 TC: %.3f ms %.2f GFLOPS\n", avg, gflops);

    csv_write("matmul","cutlass_fp16_tc",n,avg,gflops,"GFLOPS");

    if(workspace) cudaFree(workspace);
    return avg;
}

// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────
int main()
{
    srand(time(NULL));
    csv_init();

    int n = MATRIX_N;
    size_t sz = n*n*sizeof(float);

    float *h_a=(float*)malloc(sz);
    float *h_b=(float*)malloc(sz);

    for(int i=0;i<n*n;i++){
        h_a[i]=rand()%10;
        h_b[i]=rand()%10;
    }

    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,sz);
    cudaMalloc(&d_b,sz);
    cudaMalloc(&d_c,sz);

    cudaMemcpy(d_a,h_a,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,sz,cudaMemcpyHostToDevice);

    cutlass::half_t *d_a16,*d_b16;
    cudaMalloc(&d_a16,n*n*sizeof(cutlass::half_t));
    cudaMalloc(&d_b16,n*n*sizeof(cutlass::half_t));

    convert(d_a,d_a16,n);
    convert(d_b,d_b16,n);

    cudaDeviceSynchronize();

    printf("\nCUTLASS BENCHMARK\n");

    float naive = bench_fp32(d_a,d_b,d_c,n);
    float tc    = bench_fp16(d_a16,d_b16,d_c,n);

    printf("Speedup: %.2fx\n", naive/tc);

    if(g_csv) fclose(g_csv);

    return 0;
}