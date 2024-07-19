#include <complex>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include "cufft_utils.h"

// #define printf(...) {}

#define COUT(cmd) {std::cout << #cmd "=" << (cmd) << "\n";}

int main(int argc, char *argv[]) {
    cufftHandle planr2c, planc2r;
    cudaStream_t stream = NULL;

    int cudaDevId;
    cudaGetDevice(&cudaDevId);
    cudaDeviceProp cdp;
    cudaGetDeviceProperties(&cdp, cudaDevId);
    COUT(cdp.deviceOverlap)

    int fft_size = atoi(argv[1]);
    int batch_size = atoi(argv[2]);
    int element_count = batch_size * fft_size;

    using scalar_type = float;
    using input_type = scalar_type;
    using output_type = std::complex<scalar_type>;

    std::vector<input_type> input(element_count, 0);
    std::vector<output_type> output((fft_size / 2 + 1) * batch_size);

    for (auto i = 0; i < element_count; i++) {
        input[i] = static_cast<input_type>(i);
    }

    printf("Input array:\n{");
    for (auto &i : input) {
        printf("%g  ", i);
    }
    printf("}\n=====\n");

    input_type *d_input = nullptr;
    cufftComplex *d_output = nullptr;

    // create plans
    CUFFT_CALL(cufftCreate(&planr2c));
    CUFFT_CALL(cufftCreate(&planc2r));
    CUFFT_CALL(cufftPlan1d(&planr2c, fft_size, CUFFT_R2C, batch_size));
    CUFFT_CALL(cufftPlan1d(&planc2r, fft_size, CUFFT_C2R, batch_size));

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); // Specifies that work running in the created stream may run concurrently with work in stream 0 (the NULL stream), and that the created stream should perform no implicit synchronization with stream 0.
    CUFFT_CALL(cufftSetStream(planr2c, stream)); // ?
    CUFFT_CALL(cufftSetStream(planc2r, stream)); // ?

    // Create device arrays
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(input_type) * input.size()));
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(output_type) * output.size()));
    CUDA_RT_CALL(cudaMemcpyAsync(d_input, input.data(), sizeof(input_type) * input.size(),
                                 cudaMemcpyHostToDevice, stream)); // ?

    // out-of-place Forward transform
    CUFFT_CALL(cufftExecR2C(planr2c, d_input, d_output));

    CUDA_RT_CALL(cudaMemcpyAsync(output.data(), d_output, sizeof(output_type) * output.size(),
                                 cudaMemcpyDeviceToHost, stream));

    CUDA_RT_CALL(cudaStreamSynchronize(stream)); // ?

    printf("Output array after Forward FFT:\n");
    for (auto &i : output) {
        printf("%f + %fj\n", i.real(), i.imag());
    }
    printf("=====\n");

    // Normalize the data
    scaling_kernel<<<1, 128, 0, stream>>>(d_output, element_count, 1./fft_size);

    // out-of-place Inverse transform
    CUFFT_CALL(cufftExecC2R(planc2r, d_output, d_input));

    CUDA_RT_CALL(cudaMemcpyAsync(output.data(), d_input, sizeof(input_type) * input.size(),
                                 cudaMemcpyDeviceToHost, stream));

    printf("Output array after Forward FFT, Normalization, and Inverse FFT:\n");
    for (auto i = 0; i < input.size()/2; i++) {
        printf("%f + %fj\n", output[i].real(), output[i].imag());
    }
    printf("=====\n");

    /* free resources */
    CUDA_RT_CALL(cudaFree(d_input));
    CUDA_RT_CALL(cudaFree(d_output));

    CUFFT_CALL(cufftDestroy(planr2c));
    CUFFT_CALL(cufftDestroy(planc2r));

    CUDA_RT_CALL(cudaStreamDestroy(stream));

    CUDA_RT_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}
