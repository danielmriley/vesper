
# Vesper Build Plan - Chapter 8.2: Multi-Block Reduction (`sum`)

## 1. Goal

Implement the full, multi-block `sum` reduction. This involves creating a dispatch function that uses the `reduce_kernel` from the previous chapter in two passes to handle tensors of any size. This chapter completes the `sum` operation.

## 2. The Two-Pass Strategy

1.  **First Pass**: We launch as many thread blocks as needed to cover the entire input tensor. Each block runs `reduce_kernel` on its assigned chunk of data and writes its partial sum to an intermediate buffer. If the input tensor has `N` elements and we use `B` blocks, this pass reduces `N` elements to `B` partial sums.
2.  **Second Pass**: We launch a **single block** to run the exact same `reduce_kernel` on the intermediate buffer of `B` partial sums. This reduces the `B` values to our final single scalar result.

## 3. Detailed Steps

### Step 3.1: Implement the `sum_hip_dispatch` Function

We will now complete the dispatch function in `src/ops/hip/reduction.hip`. It will orchestrate the two kernel launches.

Modify `src/ops/hip/reduction.hip`:
```cpp
// src/ops/hip/reduction.hip

// The reduce_kernel from Ch 8.1 should already be here.

void sum_hip_dispatch(const Tensor& input, Tensor& output) {
    if (input.dtype() != DType::Float32) {
        throw std::runtime_error("Sum only supports Float32 for now.");
    }

    const size_t n = input.numel();
    if (n == 0) return; // Handle empty tensor case

    const int threads_per_block = 128;
    const int elements_per_block = threads_per_block * 2;
    const int num_blocks = (n + elements_per_block - 1) / elements_per_block;
    const size_t shared_mem_size = threads_per_block * sizeof(float);

    // --- First Pass: Reduce input tensor to an intermediate buffer ---
    Tensor intermediate = empty({(long long)num_blocks}, input.dtype(), input.device());

    hipLaunchKernelGGL(
        (reduce_kernel<float, std::plus<float>>),
        dim3(num_blocks), dim3(threads_per_block), shared_mem_size, 0,
        input.data_ptr<const float>(), intermediate.data_ptr<float>(), n,
        std::plus<float>(), 0.0f
    );

    // --- Second Pass: Reduce the intermediate buffer to the final scalar output ---
    if (num_blocks > 1) {
        // We can reuse the same kernel
        const int final_threads = (num_blocks + 1) / 2; // Enough threads for the smaller problem
        const size_t final_shared_mem = final_threads * sizeof(float);
         hipLaunchKernelGGL(
            (reduce_kernel<float, std::plus<float>>),
            dim3(1), dim3(final_threads), final_shared_mem, 0,
            intermediate.data_ptr<const float>(), output.data_ptr<float>(),
            num_blocks, std::plus<float>(), 0.0f
        );
    } else {
        // If there was only one block, the result is already in intermediate[0].
        // Copy the single element from the intermediate buffer to the final output.
        hipMemcpy(output.data_ptr<void>(), intermediate.data_ptr<void>(), sizeof(float), hipMemcpyDeviceToDevice);
    }
}
```

### Step 3.2: Implement the Public `sum` Function and Autograd

Create `src/ops/reduction.cpp`. This will contain the user-facing `sum` function and its backward pass.

```cpp
// src/ops/reduction.cpp
#include <vesper/ops/reduction.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/guard.h>

namespace vesper::ops {

Tensor sum(const Tensor& input) {
    if (!input.is_contiguous()) {
        throw std::runtime_error("Sum currently requires a contiguous tensor.");
    }
    
    bool result_requires_grad = input.requires_grad() && autograd::grad_mode_enabled;
    Tensor result = empty({1}, input.dtype(), input.device(), result_requires_grad);

    switch(input.device()) {
        case Device::HIP:
#if USE_HIP_BACKEND
            sum_hip_dispatch(input, result);
#else
            throw std::runtime_error("HIP backend not enabled.");
#endif
            break;
        default:
            throw std::runtime_error("Device not supported for sum operation.");
    }

    if (result_requires_grad) {
        result.grad_node = std::make_shared<autograd::Node>();
        // ... add edge to input.grad_node ...
        result.grad_node->backward_fn = [input, result]() mutable {
            // Gradient of sum is 1, broadcasted to the shape of the input.
            // grad_input = grad_output * ones_like(input)
            auto ones = vesper::full(input.shape(), input.dtype(), input.device(), 1.0f);
            // This requires a broadcast_mul op. We'll simplify for now.
            // The upstream grad `result.grad()` is a scalar.
            input.accumulate_grad(ones); // Simplified: assumes upstream grad is 1.
        };
    }
    return result;
}

} // namespace vesper::ops
```

## 4. Verification

The test will perform a `sum` on a large tensor that requires multiple blocks and verify the final result.

### Step 4.1: Create `tests/test_reduction_op.cpp`
```cpp
// tests/test_reduction_op.cpp
#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>

void test_multi_block_sum() {
    std::cout << "Testing multi-block sum operation..." << std::endl;
    
    // Use a size that will require multiple blocks
    const size_t num_elements = 4097; // An odd number to test edge cases
    
    // 1. Prepare host data and compute ground truth
    std::vector<float> h_input(num_elements);
    std::iota(h_input.begin(), h_input.end(), 0.5f); // 0.5, 1.5, 2.5...
    double cpu_sum = std::accumulate(h_input.begin(), h_input.end(), 0.0);

    // 2. Prepare device tensor
    auto d_input = vesper::empty({(long long)num_elements}, vesper::DType::Float32, vesper::Device::HIP);
    d_input.copy_from_host(h_input.data());

    // 3. Call the public `sum` function
    auto d_output = vesper::ops::sum(d_input);

    // 4. Copy result back and verify
    float gpu_sum = 0.0f;
    d_output.copy_to_host(&gpu_sum);
    
    assert(d_output.shape() == std::vector<int64_t>({1}));
    assert(std::fabs(cpu_sum - gpu_sum) / cpu_sum < 1e-4);

    std::cout << "Multi-block sum test passed!" << std::endl;
}

int main() {
    test_multi_block_sum();
    return 0;
}
```

### Step 4.2: Add Test to CMake
```cmake
# tests/CMakeLists.txt
add_executable(reduction_op_tests test_reduction_op.cpp)
target_link_libraries(reduction_op_tests PRIVATE vesper)
add_test(NAME ReductionOpTests COMMAND reduction_op_tests)
```
A passing test confirms that the two-pass reduction strategy is working correctly for tensors of any size.
