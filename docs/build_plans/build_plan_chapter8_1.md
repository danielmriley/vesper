
# Vesper Build Plan - Chapter 8.1: The Intra-Block Reduction Kernel

## 1. Goal

Implement a generic, reusable HIP kernel that can perform a reduction operation (like `sum`) on an array of data that fits within a single thread block. This kernel is the fundamental building block for reducing large tensors.

## 2. The Parallel Reduction Algorithm (Intra-Block)

To sum an array in parallel within a block, we use shared memory and a tree-like approach:
1.  Each thread in the block loads one element from global memory into a shared memory array.
2.  The threads then perform a sequence of synchronized steps. In each step, half the threads become inactive, and the other half add two values from the shared memory array.
3.  This continues until only the first thread (`threadIdx.x == 0`) is active and holds the sum of the entire shared memory array.
4.  This single thread then writes its result (the block's partial sum) to the output array in global memory.

## 3. Detailed Steps

### Step 3.1: Create the Reduction Kernel Interface

Create `include/vesper/ops/reduction.h` to declare the functions. We will only focus on the kernel-level dispatch function for this chapter.

```cpp
// include/vesper/ops/reduction.h
#pragma once

namespace vesper {

class Tensor; // Forward declaration

namespace ops {
    // The full, multi-block sum operation (to be implemented in Ch 8.2)
    Tensor sum(const Tensor& input);

    // The backend-specific dispatch for the full operation
    void sum_hip_dispatch(const Tensor& input, Tensor& output);
}
}
```

### Step 3.2: Implement the `reduce_kernel`

Create `src/ops/hip/reduction.hip`. This file will contain only the generic reduction kernel for now.

```cpp
// src/ops/hip/reduction.hip
#include <vesper/ops/reduction.h>
#include <vesper/core/tensor.h>
#include <hip/hip_runtime.h>
#include <functional>

namespace vesper::ops {

// This kernel reduces an array of size `n` and writes the result to `out_data`.
// It's designed to be called by multiple blocks, where each block writes one
// partial result to `out_data[blockIdx.x]`.
template <typename T, typename Op>
__global__ void reduce_kernel(const T* in_data, T* out_data, size_t n, Op op, T neutral_element) {
    // Use dynamic shared memory for flexibility.
    extern __shared__ T sdata[];

    // Each thread loads one element from global memory into shared memory.
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * (blockDim.x * 2) + tid; // Each block handles 2*blockDim.x elements
    size_t gridSize = blockDim.x * 2 * gridDim.x;

    // We use a grid-stride loop to allow each block to reduce more than one chunk.
    // For simplicity in this first version, we'll do a basic version.
    // Each thread loads one element from its half of the chunk.
    sdata[tid] = (i < n) ? in_data[i] : neutral_element;
    if (i + blockDim.x < n) {
        sdata[tid] = op(sdata[tid], in_data[i + blockDim.x]);
    }
     __syncthreads();


    // Perform the reduction in shared memory.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = op(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // The first thread in the block writes the block's partial sum to global memory.
    if (tid == 0) {
        out_data[blockIdx.x] = sdata[0];
    }
}

// The full dispatch function will be implemented in the next chapter.
void sum_hip_dispatch(const Tensor& input, Tensor& output) {
    throw std::runtime_error("Not implemented in this chapter.");
}

} // namespace vesper::ops
```

### Step 3.3: Update CMake
Add the new file to `src/CMakeLists.txt`:
```cmake
target_sources(vesper PRIVATE
    # ...
    ops/hip/reduction.hip # Add this
)
```

## 4. Verification

The test will launch our `reduce_kernel` with a **single block** and verify that it correctly computes the sum of a small array.

### Step 4.1: Create `tests/test_reduction_kernel.cpp`

```cpp
// tests/test_reduction_kernel.cpp
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <hip/hip_runtime.h> // For hipLaunchKernelGGL
#include <iostream>
#include <cassert>
#include <vector>
#include <numeric>
#include <functional>

// Declaration of the kernel from the .hip file for testing purposes
namespace vesper::ops {
    template <typename T, typename Op>
    __global__ void reduce_kernel(const T* in_data, T* out_data, size_t n, Op op, T neutral_element);
}

void test_single_block_reduction() {
    std::cout << "Testing single-block reduction kernel..." << std::endl;
    
    const int num_elements = 256;
    const int threads_per_block = 128; // Each thread handles 2 elements
    
    // 1. Prepare host data and compute ground truth
    std::vector<float> h_input(num_elements);
    std::iota(h_input.begin(), h_input.end(), 1.0f); // 1, 2, 3, ...
    float cpu_sum = std::accumulate(h_input.begin(), h_input.end(), 0.0f);

    // 2. Prepare device tensors
    auto d_input = vesper::empty({num_elements}, vesper::DType::Float32, vesper::Device::HIP);
    auto d_output = vesper::empty({1}, vesper::DType::Float32, vesper::Device::HIP);
    d_input.copy_from_host(h_input.data());

    // 3. Launch the kernel with ONE block
    const size_t shared_mem_size = threads_per_block * sizeof(float);
    hipLaunchKernelGGL(
        (vesper::ops::reduce_kernel<float, std::plus<float>>),
        dim3(1), dim3(threads_per_block), shared_mem_size, 0,
        d_input.data_ptr<const float>(), d_output.data_ptr<float>(),
        num_elements, std::plus<float>(), 0.0f
    );

    // 4. Copy result back and verify
    float gpu_sum = 0.0f;
    d_output.copy_to_host(&gpu_sum);
    
    assert(fabs(cpu_sum - gpu_sum) < 1e-3);
    std::cout << "Single-block reduction kernel test passed!" << std::endl;
}

int main() {
    test_single_block_reduction();
    return 0;
}
```

### Step 4.2: Add Test to `tests/CMakeLists.txt`
```cmake
add_executable(reduction_kernel_tests test_reduction_kernel.cpp)
# We need to link the test to the library to resolve hipLaunchKernelGGL etc.
target_link_libraries(reduction_kernel_tests PRIVATE vesper)
add_test(NAME ReductionKernelTests COMMAND reduction_kernel_tests)
```
A passing test confirms the core reduction logic is sound. We are now ready to use this kernel to build the full, multi-block `sum` operation.
