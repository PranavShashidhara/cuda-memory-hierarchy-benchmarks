# CUDA GPU Systems Benchmark Suite (Jetson Orin Nano)

A CUDA-based microbenchmarking framework for analyzing **GPU memory hierarchy behavior**, **kernel optimization techniques**, and **shared memory tiling efficiency** on edge-class GPUs (Jetson Orin Nano).

## Overview

This project is a **systems-level CUDA benchmarking suite** designed to study how GPU kernel design decisions impact performance at the hardware level. It isolates **core GPU execution behavior** without model or inference dependencies.

**Key focuses:**
- Memory hierarchy behavior (global, cache, shared memory)
- Kernel optimization techniques (naive → tiled → Tensor Core)
- Access pattern sensitivity (sequential, strided, random)
- Compute vs memory-bound workload characterization
- Library ceiling comparison (CUTLASS structured baselines, cuBLAS hardware ceiling)

## Results

![GEMM GFLOPS Ladder](results/matmul_gflops_combined.png)
*Full GEMM ladder — throughput (higher is better)*

![GEMM Latency Ladder](results/matmul_latency_combined.png)
*Full GEMM ladder — latency (lower is better)*

## ⚠️ Important: Run Setup First

Before running any make commands, you **must** initialize the repository structure:

```bash
./setup.sh
```

This creates all required directories (`baseline/`, `Optimized/`, `build/`, `results/`, etc.)

## Quick Start

```bash
# 1. Setup directory structure (REQUIRED — do this first!)
./setup.sh

# 2. Build both baseline and optimized versions
make all

# 3. Run benchmarks and generate plots
make bench-all
```

## Make Commands Reference

| Command | Description |
|---------|-------------|
| `make all` | Build all four targets (baseline, optimized, cuBLAS, CUTLASS) |
| `make original` | Build baseline version only |
| `make optimized` | Build optimized version only |
| `make cublas` | Build cuBLAS hardware-ceiling binary |
| `make cutlass` | Build CUTLASS structured-baseline binary |
| `make clone-cutlass` | Clone CUTLASS headers then build the CUTLASS target |
| `make run` | Run all four benchmarks |
| `make run-original` | Run baseline benchmark, save results to `baseline/results/run.log` |
| `make run-optimized` | Run optimized benchmark, save results to `Optimized/results/run.log` |
| `make run-cublas` | Run cuBLAS ceiling benchmark |
| `make run-cutlass` | Run CUTLASS structured-baseline benchmark |
| `make bench` | Run optimized benchmark + generate plots (recommended single command) |
| `make bench-all` | Run all four benchmarks + generate plots |
| `make plot` | Generate performance plots from CSV results |
| `make compare` | Run all benchmarks and generate comparison plots |
| `make nsys` | Profile optimized binary with Nsight Systems |
| `make ncu` | Profile optimized binary with Nsight Compute |
| `make clean` | Remove binaries and result files |

## Workflow

1. **Setup**: `./setup.sh` creates the directory structure
2. **Build**: `make all` compiles all four targets
3. **Run**: `make bench-all` executes all benchmarks and generates plots
4. **Results**: CSV files in `baseline/results/`, `Optimized/results/`, `cublass_and_cutlass/results/`; plots alongside each CSV and a combined chart in `results/`

## Repository Structure

```
.
├── baseline/                        # Baseline CUDA kernels (naive + WMMA)
│   ├── main.cu
│   ├── kernels.cuh / utils.cuh
│   ├── matmul_kernels.cu
│   ├── memory_kernels.cu
│   ├── wmma_matmul_kernels.cu
│   └── results/
│       ├── benchmark_results.csv
│       ├── matmul_gflops.png
│       ├── matmul_latency.png
│       ├── matmul_speedup.png
│       ├── memory_bandwidth.png
│       ├── memory_slowdown_heatmap.png
│       └── run.log
├── Optimized/                       # Optimized kernels (cp.async, WPT, dbl-buf TC)
│   ├── main.cu
│   ├── kernels.cuh / utils.cuh
│   ├── matmul_kernels.cu
│   ├── memory_kernels.cu
│   ├── wmma_matmul_kernels.cu
│   └── results/                     # same layout as baseline/results/
├── cublass_and_cutlass/             # Library ceiling benchmarks
│   ├── cublas_baseline.cu
│   ├── cutlass_baseline.cu
│   └── results/
│       ├── benchmark_results_cublas.csv
│       ├── benchmark_results_cutlass.csv
│       ├── matmul_gflops_cublas.png
│       ├── matmul_gflops_cutlass.png
│       ├── matmul_latency_cublas.png
│       └── matmul_latency_cutlass.png
├── build/                           # Compiled binaries (generated)
│   ├── cuda_bench
│   ├── cuda_bench_optimized
│   ├── cuda_bench_cublas
│   └── cuda_bench_cutlass
├── results/                         # Combined cross-build charts
│   ├── matmul_gflops_combined.png
│   └── matmul_latency_combined.png
├── plot_results.py                  # Unified plotting script
├── Makefile
├── setup.sh
├── report.nsys-rep                  # Nsight Systems profile
├── report_clean.ncu-rep             # Nsight Compute profile (clean)
└── report_full_app.ncu-rep          # Nsight Compute profile (full app)
```

## Architecture

```
Synthetic Kernels (Memory, MatMul, Compute)
           ↓
    CUDA Kernel Layer
    ├── Naive (global memory only)
    ├── Tiled v1/v2/v3 (shared memory + cp.async)
    ├── Tensor Core WMMA (fp16 input, fp32 accumulate)
    ├── CUTLASS structured baseline (Simt fp32 + TensorOp fp16)
    └── cuBLAS hardware ceiling (Sgemm fp32 + GemmEx fp16 TC)
           ↓
    Execution on Jetson Orin Nano (sm_87, 8 SMs)
           ↓
    CUDA Event Timing & Nsight Profiling
           ↓
    CSV Results → plot_results.py → PNG charts
           ↓
    Performance Analysis
```

## Core Components

### Memory Access Benchmarks
- **Sequential Access**: Coalesced memory reads via `cp.async` pipeline (best-case bandwidth)
- **Strided Access**: Non-coalesced patterns (cache-inefficient)
- **Random Access**: Cache-unfriendly patterns with bitonic sort + `cp.async` (latency stress)

### Matrix Multiplication Kernels

| Kernel | Description |
|--------|-------------|
| `naive` | Direct global memory access, high DRAM traffic |
| `tiled v1` | 32×32 shared memory tiling, padded to avoid bank conflicts |
| `tiled v2` | Work-Per-Thread=4, 8×8 register tile — best fp32 hand-written kernel |
| `tiled v3` | `cp.async` software pipelining + WPT=4 |
| `tc_basic` | WMMA 16×16×16 MMA on Tensor Cores, 64×64 block tile |
| `tc_optimized` | WMMA with 128×128 block tile, double-buffered shared memory |
| CUTLASS fp32 | Simt GEMM via CUTLASS structured templates |
| CUTLASS fp16 TC | TensorOp fp16 GEMM via CUTLASS multistage pipeline |
| cuBLAS Sgemm | fp32 hardware ceiling via cuBLAS |
| cuBLAS GemmEx | fp16 Tensor Core hardware ceiling via cuBLAS |

### Metrics Collected
- Kernel execution time (ms)
- Memory bandwidth (GB/s) with utilization vs theoretical peak
- Throughput (GFLOPS)
- Speedup vs naive baseline
- Occupancy estimates

## Benchmark Results (Jetson Orin Nano, sm_87)

### Memory Access — Key Findings

| Pattern | Peak BW (Optimized) | Notes |
|---------|---------------------|-------|
| Sequential (`cp.async ×4`) | ~64 GB/s | Exceeds DRAM peak at mid-sizes due to cache effects |
| Strided ×4 (`cp.async`) | ~59 GB/s | Near-sequential at small sizes; diverges at 256 MB |
| Random (bitonic + `cp.async`) | ~12 GB/s | Cache pressure collapses bandwidth; up to 24× slowdown |

Random access slowdown peaks at **23.8× vs sequential** at 256 MB — demonstrating how severely cache-unfriendly patterns degrade effective bandwidth.

### Matrix Multiplication — GFLOPS Ladder (2048×2048)

| Kernel | GFLOPS | Speedup vs Naive |
|--------|--------|-----------------|
| Naive fp32 | ~172 | 1.0× |
| Tiled v1 (fp32, 32×32) | ~157 | 0.92× *(register pressure regression)* |
| Tiled v2 (WPT=4, 8×8) | **968** | **5.64×** |
| Tiled v3 (cp.async + WPT=4) | 964 | 5.62× |
| TC WMMA Optimized (fp16, 128×128) | **1,775** | **10.34×** |
| CUTLASS fp32 (Simt) | 1,310 | — |
| CUTLASS fp16 TC | **6,934** | — |
| cuBLAS Sgemm (fp32) | 1,149 | — |
| cuBLAS GemmEx (fp16 TC) | **4,916** | — |

> Note: CUTLASS fp16 TC achieves the highest throughput, reflecting its highly optimized multistage pipelining vs the hand-written WMMA kernel. cuBLAS GemmEx is the practical fp16 ceiling for drop-in usage.

![GEMM Ladder — Full Comparison](results/matmul_gflops_combined.png)
*Full GEMM ladder: hand-written kernels vs CUTLASS vs cuBLAS on Jetson Orin Nano (sm_87)*

## Expected Outcomes

- Tiled kernels significantly reduce global memory traffic; WPT=4 delivers the largest fp32 gain (~5.6×)
- `cp.async` pipelining (`tiled_v3`) shows marginal gain over `tiled_v2` at this problem size
- Tensor Core kernels deliver 10× over naive at 2048×2048 (hand-written) and 40× (CUTLASS)
- Coalesced access approaches or exceeds stated DRAM bandwidth with pipelining
- Random access exposes raw DRAM latency — bandwidth collapses to 3–12 GB/s
- Edge GPUs (Jetson Orin Nano, 8 SMs) are heavily memory-bound at small to mid-sizes

## Key Insights

- GPU performance is dominated by **memory hierarchy behavior**
- Shared memory tiling is critical, but register tiling (WPT) matters equally at scale
- `cp.async` helps hide latency but doesn't change the memory-bound ceiling
- Tensor Cores operate in a qualitatively different performance tier
- Library baselines (CUTLASS, cuBLAS) reveal how much performance headroom custom kernels leave
- Edge GPUs exhibit different bottlenecks than datacenter GPUs: fewer SMs mean less latency hiding

## Plotting

`plot_results.py` reads all four CSVs and generates per-build and combined charts:

```bash
python3 plot_results.py              # all four builds
python3 plot_results.py --baseline-only
python3 plot_results.py --optimized-only
python3 plot_results.py --cublas-only
python3 plot_results.py --cutlass-only
```

**Charts produced per build:**
- `memory_bandwidth.png` — bandwidth vs array size
- `memory_slowdown_heatmap.png` — strided/random slowdown grid
- `matmul_gflops.png` — GFLOPS bar chart
- `matmul_latency.png` — latency bar chart
- `matmul_speedup.png` — speedup ladder (skipped for library-only CSVs)

**Combined chart** (when multiple builds present):
- `results/matmul_gflops_combined.png` — full GEMM ladder across all four sources

### CSV Format

```
benchmark,variant,n_or_size,avg_ms,metric_value,metric_unit
memory,Sequential,1048576,0.139,60.35,GB/s
matmul,naive,2048,100.119,171.59,GFLOPS
matmul,cublas_sgemm,2048,14.954,1148.87,GFLOPS
```

Recognised `variant` names for matmul: `naive`, `tiled`, `tiled_v2`, `tiled_v3`, `tc_basic`, `tc_optimized`, `cutlass_sgemm`, `cutlass_fp32`, `cutlass_gemm_fp16_tc`, `cutlass_fp16_tc`, `cublas_sgemm`, `cublas_gemmex_fp16`.
