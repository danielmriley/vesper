# Vesper Build Plan - Chapter 19.2: The `stack` Operation for Batching

## 1. Goal

Implement the `vesper::ops::stack` operation. This is a fundamental tensor manipulation primitive that combines a list of tensors into a single new tensor along a new dimension. It is the core mechanism that a `DataLoader` uses to collate individual samples into a single batch tensor.

## 2. The `stack` Operation

If you have `N` tensors of shape `[C, H, W]`, `stack(tensors, dim=0)` will produce a single tensor of shape `[N, C, H, W]`. It creates a new dimension and concatenates the tensors along it.

Implementing a highly optimized `stack` kernel is complex. For this chapter, we will focus on a simple, correct implementation that copies data via the CPU. This is inefficient but unblocks the development of the `DataLoader`.

## 3. Detailed Steps

### Step 3.1: Implement `ops::stack`

Create `include/vesper/ops/stack.h` and `src/ops/stack.cpp`.
```cpp
// include/vesper/ops/stack.h
#pragma once
#include <vesper/core/tensor.h>
#include <vector>

namespace vesper::ops {
    // Stacks a vector of tensors along a new dimension `dim`.
    Tensor stack(const std::vector<Tensor>& tensors, int dim = 0);
}
```
Create `src/ops/stack.cpp`:
```cpp
// src/ops/stack.cpp
#include <vesper/ops/stack.h>
#include <vesper/core/factories.h>

namespace vesper::ops {

Tensor stack(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::runtime_error("Cannot stack an empty list of tensors.");
    }

    // 1. All tensors must have the same shape and device
    const auto& first_shape = tensors[0].shape();
    const auto& device = tensors[0].device();
    const auto& dtype = tensors[0].dtype();
    for (size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i].shape() != first_shape || tensors[i].device() != device) {
            throw std::runtime_error("All tensors in stack must have the same shape and device.");
        }
    }

    // 2. Determine the output shape
    auto output_shape = first_shape;
    output_shape.insert(output_shape.begin() + dim, tensors.size());
    Tensor output = empty(output_shape, dtype, device);

    // 3. Copy data (slow, CPU-based implementation)
    size_t single_tensor_size = tensors[0].numel();
    size_t single_tensor_bytes = single_tensor_size * GetDTypeSize(dtype);
    std::vector<char> host_buffer(output.numel() * GetDTypeSize(dtype));

    for (size_t i = 0; i < tensors.size(); ++i) {
        // Copy each tensor's data into the correct slice of the host buffer
        tensors[i].copy_to_host(host_buffer.data() + i * single_tensor_bytes);
    }
    
    // Copy the entire collated buffer to the output tensor
    output.copy_from_host(host_data.data());

    // NOTE: This implementation does not support autograd for simplicity.
    return output;
}

} // namespace vesper::ops
```
*Note: A performant version would use a custom GPU kernel to copy slices directly on the device, avoiding the round-trip to the host.*

### Step 3.2: Update CMake
Add `src/ops/stack.cpp` to `src/CMakeLists.txt`.

## 4. Verification

The test will stack a few small tensors and verify that the resulting tensor has the correct shape and that the data is in the correct order.

### Step 4.1: Create `tests/test_stack_op.cpp`
```cpp
// tests/test_stack_op.cpp
#include <vesper/ops/stack.h>
#include <iostream>
#include <cassert>

void test_stack_op() {
    std::cout << "Testing stack operation..." << std::endl;
    auto device = vesper::Device::CPU;

    // 1. Create tensors to stack
    auto t1 = vesper::full({2}, vesper::DType::Float32, device, 1.0f); // {1, 1}
    auto t2 = vesper::full({2}, vesper::DType::Float32, device, 2.0f); // {2, 2}
    auto t3 = vesper::full({2}, vesper::DType::Float32, device, 3.0f); // {3, 3}

    std::vector<vesper::Tensor> tensors = {t1, t2, t3};
    
    // 2. Stack them
    auto stacked = vesper::ops::stack(tensors, 0);

    // 3. Verify shape
    assert(stacked.shape() == std::vector<int64_t>({3, 2}));

    // 4. Verify data
    std::vector<float> data(stacked.numel());
    stacked.copy_to_host(data.data());

    assert(fabs(data[0] - 1.0f) < 1e-6); assert(fabs(data[1] - 1.0f) < 1e-6);
    assert(fabs(data[2] - 2.0f) < 1e-6); assert(fabs(data[3] - 2.0f) < 1e-6);
    assert(fabs(data[4] - 3.0f) < 1e-6); assert(fabs(data[5] - 3.0f) < 1e-6);
    
    std::cout << "Stack operation test passed!" << std::endl;
}

int main() {
    test_stack_op();
    return 0;
}
```
### Step 4.2: Add to CMake
A passing test confirms the `stack` operation works correctly, unblocking the implementation of an efficient `DataLoader`.