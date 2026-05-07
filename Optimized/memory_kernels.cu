#include "kernels.cuh"
#include <cuda_pipeline.h>

// ============================================================================
// SEQUENTIAL: cp.async pipeline — thread never stalls on LDG
// Uses __pipeline_memcpy_async to copy global → L1/smem directly,
// bypassing the register file entirely (no LDG.E stall).
// Processes PIPE_DEPTH tiles ahead so memory latency is fully hidden.
// ============================================================================
#define PIPE_DEPTH 4   // how many async copies to keep in flight

__global__ void sequential(float *a, float *out, int N) {
    // Shared memory staging buffer: PIPE_DEPTH slots x blockDim elements
    extern __shared__ float smem[];  // blockDim.x * PIPE_DEPTH floats

    int tid = threadIdx.x;
    int bsz = blockDim.x;

    // Each iteration of the outer loop processes one block-worth of elements
    // from the staging buffer at offset [slot * bsz]
    int base = blockIdx.x * bsz * PIPE_DEPTH;

    // ── Prologue: fill the pipeline ────────────────────────────────────────
    #pragma unroll
    for (int d = 0; d < PIPE_DEPTH; d++) {
        int gid = base + d * bsz + tid;
        __pipeline_memcpy_async(
            &smem[d * bsz + tid],
            (gid < N) ? &a[gid] : &a[0],   // safe dummy addr for OOB
            sizeof(float));
        __pipeline_commit();
    }

    // ── Steady state: wait for slot, write out, refill ─────────────────────
    // (for large N the block loops over multiple PIPE_DEPTH-sized chunks)
    // Here each block handles exactly one chunk (PIPE_DEPTH * bsz elements)
    #pragma unroll
    for (int d = 0; d < PIPE_DEPTH; d++) {
        __pipeline_wait_prior(PIPE_DEPTH - 1 - d);  // wait for oldest in-flight
        __syncthreads();
        int gid = base + d * bsz + tid;
        if (gid < N)
            out[gid] = smem[d * bsz + tid];
    }
}

// ============================================================================
// STRIDED: cp.async load into smem, then scattered write from smem
// L1 hit on the read side (smem), write side is always scattered.
// ============================================================================
__global__ void strided(float *a, float *out, int stride, int N) {
    extern __shared__ float smem[];

    int tid  = threadIdx.x;
    int bsz  = blockDim.x;
    int base = blockIdx.x * bsz * PIPE_DEPTH;

    #pragma unroll
    for (int d = 0; d < PIPE_DEPTH; d++) {
        int gid = base + d * bsz + tid;
        __pipeline_memcpy_async(
            &smem[d * bsz + tid],
            (gid < N) ? &a[gid] : &a[0],
            sizeof(float));
        __pipeline_commit();
    }

    #pragma unroll
    for (int d = 0; d < PIPE_DEPTH; d++) {
        __pipeline_wait_prior(PIPE_DEPTH - 1 - d);
        __syncthreads();
        int gid     = base + d * bsz + tid;
        int out_idx = gid * stride;
        if (gid < N && out_idx < N)
            out[out_idx] = smem[d * bsz + tid];
    }
}

// ============================================================================
// RANDOM ACCESS: bitonic sort + cp.async gather
// Step 1: Sort indices within the block (coalesces accesses as much as
//         possible for random data).
// Step 2: Use cp.async to load the sorted-index elements into smem so the
//         gather itself doesn't stall on LDG.
// ============================================================================
__global__ void random_access(float *a, float *out, int *idx, int N) {
    // smem layout: [0 .. bsz-1] = sort buffer (int), reused as [float] after sort
    extern __shared__ int s_idx[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // ── Load indices into smem ──────────────────────────────────────────────
    s_idx[tid] = (gid < N) ? idx[gid] : INT_MAX;
    __syncthreads();

    // ── Bitonic sort (ascending) — coalesces the gather below ──────────────
    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool asc = ((tid & k) == 0);
                if ((s_idx[tid] > s_idx[ixj]) == asc) {
                    int tmp   = s_idx[tid];
                    s_idx[tid] = s_idx[ixj];
                    s_idx[ixj] = tmp;
                }
            }
            __syncthreads();
        }
    }

    // ── Async gather: sorted index gather via cp.async ─────────────────────
    // Reuse smem as float staging (safe: sort is done, ints and floats are 4B)
    float *s_val = reinterpret_cast<float*>(s_idx);
    int sorted = s_idx[tid];   // save before smem is overwritten

    __pipeline_memcpy_async(
        &s_val[tid],
        (sorted < N) ? &a[sorted] : &a[0],
        sizeof(float));
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    if (gid < N)
        out[gid] = s_val[tid];
}

// ============================================================================
// SEQUENTIAL SHARED: baseline reference (unchanged)
// ============================================================================
__global__ void sequential_shared(float *a, float *out, int N) {
    extern __shared__ float sA[];
    int tid       = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    if (global_id < N) sA[tid] = a[global_id];
    __syncthreads();
    if (global_id < N) out[global_id] = sA[tid];
}
