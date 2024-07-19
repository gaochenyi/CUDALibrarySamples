#pragma once

#define DEBUG

// CUDA runtime API error checking
#ifndef CUDA_RT_CALL
#if defined(DEBUG) || defined(_DEBUG)
#include <cuda_runtime_api.h>
#include <stdio.h>
#   define CUDA_RT_CALL(call)                            \
    do                                                   \
    {                                                    \
        auto status = static_cast<cudaError_t>(call);    \
        if (status != cudaSuccess)                       \
            fprintf(stderr,                              \
                    "ERROR: CUDA runtime call \"%s\""    \
                    " in line %d of file %s"             \
                    " failed with %s (%d).\n",           \
                    #call, __LINE__, __FILE__,           \
                    cudaGetErrorString(status), status); \
    } while (0)
#else
#   define CUDA_RT_CALL(call) do { call } while (0)
#endif
#endif // CUDA_RT_CALL

// cuFFT API error checking
#ifndef CUFFT_CALL
#include <cufft.h>
#include <stdio.h>
#   define CUFFT_CALL(call)                           \
    do                                                \
    {                                                 \
        auto status = static_cast<cufftResult>(call); \
        if (status != CUFFT_SUCCESS)                  \
            fprintf(stderr,                           \
                    "ERROR: cuFFT call \"%s\""        \
                    " in line %d of file %s"          \
                    " failed with code (%d).\n",      \
                    #call, __LINE__, __FILE__,        \
                    status);                          \
    } while (0)
#endif // CUFFT_CALL

__global__ void scaling_kernel(cufftComplex *data, int element_count, float scale)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (auto i = tid; i < element_count; i += stride)
    {
        data[tid].x *= scale;
        data[tid].y *= scale;
    }
}
