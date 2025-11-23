#include "vesper/ops/reduction.h"
#include "vesper/core/tensor.h"
#include <cuda_runtime.h>
#include <iostream>

namespace vesper {
namespace ops {

// Existing sum_cuda_dispatch ... (Assuming existing content)
// I will rewrite the whole file to be safe or just append if I could.
// Let's rewrite with the new kernels added.

// ... [Previous reduce_sum_kernel code] ...
// ... [Previous sum_cuda_dispatch code] ...

// Copied from previous `read_file` output of reduction.cu
__global__ void reduce_sum_kernel(const float* input, float* output, size_t n) {
    __shared__ float sdata[1024];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;
    float local_sum = 0.0f;
    while (i < n) { local_sum += input[i]; i += gridSize; }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sdata[tid] += sdata[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { output[blockIdx.x] = sdata[0]; }
}

void sum_cuda_dispatch(const Tensor& input, Tensor& output) {
    size_t n = input.numel();
    if (n == 0) return;
    const float* d_input = input.data_ptr<const float>();
    float* d_output = output.data_ptr<float>();
    int blockSize = 256;
    int minGridSize = (n + blockSize - 1) / blockSize;
    int gridSize = (minGridSize < 1024) ? minGridSize : 1024;
    float* d_partial_sums = nullptr;
    cudaMalloc(&d_partial_sums, gridSize * sizeof(float));
    reduce_sum_kernel<<<gridSize, blockSize>>>(d_input, d_partial_sums, n);
    reduce_sum_kernel<<<1, blockSize>>>(d_partial_sums, d_output, gridSize);
    cudaFree(d_partial_sums);
}

// ======================================================================================
// Sum Rows [M, N] -> [N]
// ======================================================================================

__global__ void sum_rows_kernel(const float* input, float* output, int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += input[i * N + j];
        }
        output[j] = sum;
    }
}

void sum_rows_cuda_dispatch(const Tensor& input, Tensor& output) {
    int64_t M = input.shape()[0];
    int64_t N = input.shape()[1];
    const float* d_input = input.data_ptr<const float>();
    float* d_output = output.data_ptr<float>();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sum_rows_kernel<<<blocks, threads>>>(d_input, d_output, M, N);
}

// ======================================================================================
// New: Sum Cols [M, N] -> [M, 1]
// ======================================================================================

__global__ void sum_cols_kernel(const float* input, float* output, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += input[i * N + j];
        }
        output[i] = sum;
    }
}

void sum_cols_cuda_dispatch(const Tensor& input, Tensor& output) {
    int64_t M = input.shape()[0];
    int64_t N = input.shape()[1];
    const float* d_input = input.data_ptr<const float>();
    float* d_output = output.data_ptr<float>();
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    sum_cols_kernel<<<blocks, threads>>>(d_input, d_output, M, N);
}

} // namespace ops
} // namespace vesper