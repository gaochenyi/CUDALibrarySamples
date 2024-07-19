/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <array>
#include <complex>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cufft.h>
#include "cufft_utils.h"
#include <cassert>

using dim_t = std::array<int, 2>;

#define DEBUG

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

int main(int argc, char *argv[]) {
    cufftHandle planc2r, planr2c;
    cudaStream_t stream = NULL;

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    float time;

    const int nx = atoi(argv[1]);
    const int ny = atoi(argv[2]);
    dim_t fft_size = {nx, ny};
    const int batch_size = atoi(argv[3]);

    using scalar_type = float;
    using input_type = std::complex<scalar_type>;
    using output_type = scalar_type;

    std::vector<input_type> input_complex(batch_size * nx * (ny/2 + 1));
    std::vector<output_type> output_real(batch_size * nx * ny, 0);

    for (int i = 0; i < input_complex.size(); i++) {
        input_complex[i] = input_type(i, 0);
    }

#if 0
    std::printf("Input array:\n");
    for (auto &i : input_complex) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");
#endif

    cufftComplex *d_data = nullptr;

    CUFFT_CALL(cufftCreate(&planc2r));
    CUFFT_CALL(cufftCreate(&planr2c));
    // inembed/onembed being nullptr indicates contiguous data for each batch, then the stride and dist settings are ignored
    checkCuda( cudaEventRecord(startEvent, 0) );
    CUFFT_CALL(cufftPlanMany(&planc2r, fft_size.size(), fft_size.data(),
                             nullptr, 1, 0, // *inembed, istride, idist
                             nullptr, 1, 0, // *onembed, ostride, odist
                             CUFFT_C2R, batch_size));
    checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("  cufftPlanManyC2R: %f ms\n", time);
    // const auto time_cufftPlanManyC2R = time;

    checkCuda( cudaEventRecord(startEvent, 0) );
    CUFFT_CALL(cufftPlanMany(&planr2c, fft_size.size(), fft_size.data(),
                             nullptr, 1, 0, // *inembed, istride, idist
                             nullptr, 1, 0, // *onembed, ostride, odist
                             CUFFT_R2C, batch_size));
    checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("  cufftPlanManyR2C: %f ms\n", time);
    const auto time_cufftPlanManyR2C = time;

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(planc2r, stream));
    CUFFT_CALL(cufftSetStream(planr2c, stream));

    // Create device arrays
    // For in-place r2c/c2r transforms, make sure the device array is always allocated to the size of complex array
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_data), sizeof(input_type) * input_complex.size()));

    checkCuda( cudaEventRecord(startEvent, 0) );
    CUDA_RT_CALL(cudaMemcpyAsync(d_data, (input_complex.data()), sizeof(input_type) * input_complex.size(),
                                 cudaMemcpyHostToDevice, stream));
    checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("  cudaMemcpyHostToDevice: %f ms\n", time);
    const auto time_cudaMemcpyHostToDevice = time;

    // C2R
    checkCuda( cudaEventRecord(startEvent, 0) );
    CUFFT_CALL(cufftExecC2R(planc2r, d_data, reinterpret_cast<scalar_type*>(d_data)));
    checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("  cufftExecC2R: %f ms\n", time);

    checkCuda( cudaEventRecord(startEvent, 0) );
    CUDA_RT_CALL(cudaMemcpyAsync(output_real.data(), reinterpret_cast<scalar_type*>(d_data), sizeof(output_type) * output_real.size(),
                                 cudaMemcpyDeviceToHost, stream));
    checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("  cudaMemcpyDeviceToHost: %f ms\n", time);
    const auto time_cudaMemcpyDeviceToHost = time;

    CUDA_RT_CALL(cudaStreamSynchronize(stream));

#if 0
    std::printf("Output array after C2R:\n");
    for (auto i = 0; i < output_real.size(); i++) {
        std::printf("%f\n", output_real[i]);
    }
    std::printf("=====\n");
#endif

    checkCuda( cudaEventRecord(startEvent, 0) );
    // Normalize the data
    scaling_kernel<<<1, 128, 0, stream>>>(d_data, input_complex.size(), 1.f/(nx * ny));
    checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("  normalization by scaling: %f ms\n", time);

    checkCuda( cudaEventRecord(startEvent, 0) );
    // R2C
    CUFFT_CALL(cufftExecR2C(planr2c, reinterpret_cast<scalar_type*>(d_data), d_data));
    checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("  cufftExecR2C: %f ms\n", time);
    const auto time_cufftExecR2C = time;

    checkCuda( cudaEventRecord(startEvent, 0) );
    CUDA_RT_CALL(cudaMemcpyAsync(input_complex.data(), d_data, sizeof(input_type) * input_complex.size(),
                                 cudaMemcpyDeviceToHost, stream));
    checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("  cudaMemcpyDeviceToHost: %f ms\n", time);

#if 0
    std::printf("Output array after C2R, Normalization, and R2C:\n");
    for (auto i = 0; i < input_complex.size(); i++) {
        std::printf("%f + %fj\n", input_complex[i].real(), input_complex[i].imag());
    }
    std::printf("=====\n");
#endif

    CUDA_RT_CALL(cudaEventDestroy(startEvent));
    CUDA_RT_CALL(cudaEventDestroy(stopEvent));
    /* free resources */
    CUDA_RT_CALL(cudaFree(d_data));

    CUFFT_CALL(cufftDestroy(planc2r));
    CUFFT_CALL(cufftDestroy(planr2c));

    CUDA_RT_CALL(cudaStreamDestroy(stream));

    CUDA_RT_CALL(cudaDeviceReset());

    std::cerr << nx
        << "\t" << ny
        << "\t" << time_cufftExecR2C
        << "\t" << time_cufftPlanManyR2C
        << "\t" << time_cudaMemcpyHostToDevice
        << "\t" << time_cudaMemcpyDeviceToHost
        << "\t" << batch_size
        << "\n";

    return EXIT_SUCCESS;
}
