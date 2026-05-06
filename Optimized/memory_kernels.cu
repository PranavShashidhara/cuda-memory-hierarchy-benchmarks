#include "kernels.cuh"

// ============================================================================
// SEQUENTIAL: float4 vectorized loads/stores
// Reads 16 bytes per transaction instead of 4 — 4x fewer memory requests,
// maximizes L2 and DRAM burst utilization.
// ============================================================================
__global__ void sequential(float *a, float *out, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float4 val = reinterpret_cast<float4*>(a)[i / 4];
        reinterpret_cast<float4*>(out)[i / 4] = val;
    } else {
        for (int j = i; j < N && j < i + 4; j++)
            out[j] = a[j];
    }
}

// ============================================================================
// STRIDED: Vectorized coalesced load, scattered write
// Reads are always coalesced via float4; scattered writes are unavoidable
// but we minimize their cost by batching the load side.
// ============================================================================
__global__ void strided(float *a, float *out, int stride, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (i + 3 < N) {
        float4 val = reinterpret_cast<float4*>(a)[i / 4];
        float elems[4] = {val.x, val.y, val.z, val.w};
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int out_idx = (i + k) * stride;
            if (out_idx < N)
                out[out_idx] = elems[k];
        }
    }
}

// ============================================================================
// RANDOM ACCESS: Bitonic sort of indices in shared memory per block
// Sorting indices within each block so accesses are as coalesced as
// possible, reducing L2 miss rate for large out-of-cache arrays.
// ============================================================================
__global__ void random_access(float *a, float *out, int *idx, int N) {
    extern __shared__ int s_idx[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    s_idx[tid] = (gid < N) ? idx[gid] : INT_MAX;
    __syncthreads();

    // Bitonic sort within block
    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool ascending = ((tid & k) == 0);
                if ((s_idx[tid] > s_idx[ixj]) == ascending) {
                    int tmp = s_idx[tid];
                    s_idx[tid] = s_idx[ixj];
                    s_idx[ixj] = tmp;
                }
            }
            __syncthreads();
        }
    }

    if (gid < N)
        out[gid] = __ldg(&a[s_idx[tid]]);
}

// ============================================================================
// SEQUENTIAL SHARED: Baseline reference
// ============================================================================
__global__ void sequential_shared(float *a, float *out, int N) {
    extern __shared__ float sA[];
    int tid       = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    if (global_id < N) sA[tid] = a[global_id];
    __syncthreads();
    if (global_id < N) out[global_id] = sA[tid];
}