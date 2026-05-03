#include <stdio.h>
#include <cuda_runtime.h>
#include "utils.cuh"
#include "kernels.cuh"

// Example timing wrapper for SEQUENTIAL kernel
float time_sequential(float *d_a, float *d_b, float *d_c, int N) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    sequential<<<256, 256>>>(d_a, d_c, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    return ms;
}