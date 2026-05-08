# CUDA GPU Systems Benchmark Suite (Jetson Orin Nano)

A CUDA-based microbenchmarking framework for analyzing **GPU memory hierarchy behavior**, **kernel optimization techniques**, and **shared memory tiling efficiency** on edge-class GPUs (Jetson Orin Nano).

## Overview

This project is a **systems-level CUDA benchmarking suite** designed to study how GPU kernel design decisions impact performance at the hardware level. It isolates **core GPU execution behavior** without model or inference dependencies.

**Key focuses:**
- Memory hierarchy behavior (global, cache, shared memory)
- Kernel optimization techniques (naive vs tiled implementations)
- Access pattern sensitivity (sequential, strided, random)
- Compute vs memory-bound workload characterization

## ⚠️ Important: Run Setup First

Before running any make commands, you **must** initialize the repository structure:

```bash
./setup.sh
```

This creates all required directories (`baseline/`, `Optimized/`, `build/`, `results/`, etc.)

## Quick Start

```bash
# 1. Setup directory structure (REQUIRED - do this first!)
./setup.sh

# 2. Build both baseline and optimized versions
make all

# 3. Run benchmarks and generate plots
make bench
```

## Make Commands Reference

| Command | Description |
|---------|-------------|
| `make all` | Build both baseline and optimized binaries |
| `make original` | Build baseline version only |
| `make optimized` | Build optimized version only |
| `make run` | Run both baseline and optimized benchmarks |
| `make run-original` | Run baseline benchmark, save results to `baseline/run.log` |
| `make run-optimized` | Run optimized benchmark, save results to `Optimized/results/run.log` |
| `make bench` | Run optimized benchmark + generate plots (recommended single command) |
| `make plot` | Generate performance plots from CSV results |
| `make compare` | Run all benchmarks and generate comparison plots |
| `make nsys` | Profile optimized binary with Nsight Systems |
| `make ncu` | Profile optimized binary with Nsight Compute |
| `make clean` | Remove binaries and result files |

## Workflow

1. **Setup**: `./setup.sh` creates the directory structure
2. **Build**: `make all` compiles baseline and optimized kernels
3. **Run**: `make bench` executes benchmarks and generates plots
4. **Results**: CSV files stored in `baseline/` and `Optimized/results/`, plots in respective directories

## Architecture

```
Synthetic Kernels (Memory, MatMul, Compute)
           ↓
    CUDA Kernel Layer (Naive vs Optimized)
           ↓
    Execution on Jetson Orin Nano (sm_87)
           ↓
    CUDA Event Timing & Nsight Profiling
           ↓
    CSV Results → Python Visualization
           ↓
    Performance Analysis
```

## Core Components

### Memory Access Benchmarks
- **Sequential Access**: Coalesced memory reads (best-case bandwidth)
- **Strided Access**: Non-coalesced patterns (cache-inefficient)
- **Random Access**: Cache-unfriendly patterns (latency stress)

### Matrix Multiplication Kernels
- **Naive**: Direct global memory access, high DRAM traffic
- **Tiled (Optimized)**: Shared memory blocking, data reuse, reduced transactions

### Metrics Collected
- Kernel execution time (ms)
- Memory bandwidth (GB/s)
- Speedup vs baseline
- Occupancy estimates

## Expected Outcomes

- Tiled kernels significantly reduce global memory traffic
- Coalesced access improves bandwidth utilization
- Strided access exposes cache inefficiencies
- Many GPU workloads are memory-bound at small scales
- Performance divergence increases with input size

## Key Insights

- GPU performance is dominated by **memory hierarchy behavior**
- Shared memory tiling is a critical optimization technique
- Kernel performance depends heavily on memory access patterns
- Edge GPUs (Jetson Orin Nano) exhibit different bottlenecks than datacenter GPUs
- Profiling is essential for understanding true bottlenecks

## Scope

**Includes:**
- CUDA kernel performance analysis
- GPU architecture behavior characterization
- Memory hierarchy studies
- Kernel optimization techniques

**Excludes:**
- LLM inference systems
- ML model training
- Distributed GPU workloads
- High-level application logic