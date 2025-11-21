#include "vesper/ops/reduction.h"
#include "vesper/core/tensor.h"
#include <hip/hip_runtime.h>
#include <iostream>

namespace vesper {
namespace ops {

// ======================================================================================
// Chapter 8.1: The Intra-Block Reduction Kernel
// ======================================================================================

// A simple block-reduce kernel.
// Each thread loads elements, adds them, then we reduce within shared memory.
//
// Grid-Stride Loop:
// If the array is larger than the grid, threads loop over the array.
//
// Shared Memory:
// We use a static size or dynamic size. For simplicity here, let's use a fixed size
// assuming block size is 256 or 512.
//
// Arguments:
// - input: pointer to input data
// - output: pointer to output data (array of partial sums, one per block)
// - n: total number of elements
__global__ void reduce_sum_kernel(const float* input, float* output, size_t n) {
    // Shared memory for the block
    // We'll assume blockDim.x is max 1024.
    __shared__ float sdata[1024];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    // 1. Grid-Stride Loop: Accumulate local sum
    float local_sum = 0.0f;
    while (i < n) {
        local_sum += input[i];
        i += gridSize;
    }

    // 2. Store in shared memory
    sdata[tid] = local_sum;
    __syncthreads();

    // 3. Parallel Reduction in Shared Memory
    // We do the standard tree-based reduction.
    // This assumes blockDim.x is a power of 2.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 4. Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ======================================================================================
// Chapter 8.2: Multi-Block Reduction Dispatch
// ======================================================================================

void sum_hip_dispatch(const Tensor& input, Tensor& output) {
    // 1. Get input size
    size_t n = input.numel();
    if (n == 0) {
        // Handle empty tensor? Set output to 0.
        // For now assume n > 0.
        return;
    }

    const float* d_input = input.data_ptr<const float>();
    float* d_output = output.data_ptr<float>();

    // 2. Configure Kernel
    int blockSize = 256;
    // Calculate grid size needed to cover the data, but cap it to avoid too many blocks.
    // A common heuristic is to use enough blocks to saturate the GPU, e.g., 1024 blocks.
    // Or just (n + blockSize - 1) / blockSize.
    // For very large N, we don't want gridDim to explode.
    // Let's pick a reasonable max grid size, say 1024.
    int minGridSize = (n + blockSize - 1) / blockSize;
    int gridSize = (minGridSize < 1024) ? minGridSize : 1024;

    // 3. Allocate Intermediate Buffer (Partial Sums)
    // Each block writes 1 partial sum. So we need 'gridSize' floats.
    float* d_partial_sums = nullptr;
    hipError_t err = hipMalloc(&d_partial_sums, gridSize * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "HIP Error allocating partial sums: " << hipGetErrorString(err) << std::endl;
        return;
    }

    // 4. Pass 1: Reduce Input -> Partial Sums
    reduce_sum_kernel<<<gridSize, blockSize>>>(d_input, d_partial_sums, n);
    
    // Check for errors
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "HIP Error in Pass 1: " << hipGetErrorString(err) << std::endl;
        hipFree(d_partial_sums);
        return;
    }

    // 5. Pass 2: Reduce Partial Sums -> Output
    // We have 'gridSize' elements in d_partial_sums.
    // We want to reduce them to 1 element in d_output.
    // Since gridSize is small (<= 1024), one block is enough.
    reduce_sum_kernel<<<1, blockSize>>>(d_partial_sums, d_output, gridSize);

    // Check for errors
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "HIP Error in Pass 2: " << hipGetErrorString(err) << std::endl;
    }

    // 6. Cleanup
    hipFree(d_partial_sums);
}

} // namespace ops
} // namespace vesper
