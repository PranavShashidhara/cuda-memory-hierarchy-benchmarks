#pragma once
// Optimized/utils.cuh — CUDA error-check macro
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { printf("CUDA Error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } } while (0)

