
# Vesper Build Plan - Chapter 5: HIP Backend: Element-wise Kernels

## 1. Goal

Implement the first computational capabilities of Vesper: element-wise binary operations (e.g., addition) for tensors residing on a HIP device. This involves writing a generic HIP kernel, creating a C++ dispatch mechanism, and adding methods for copying data to and from the GPU to enable testing.

## 2. Prerequisites

- Chapter 4: The `Tensor` data structure is complete.
- ROCm toolkit is installed, and CMake is configured to find it.

## 3. Detailed Steps

### Step 3.1: Add Data Transfer Methods to Tensor

Before we can test operations, we need a way to move data between the host (CPU) and the device (GPU). We'll add `copy_from_host` and `copy_to_host` methods to the `Tensor` class.

Add the declarations to `include/vesper/core/tensor.h`:
```cpp
// In public section of Tensor class
// Copies data from a CPU buffer to the tensor's storage
void copy_from_host(const void* host_ptr);
// Copies data from the tensor's storage to a CPU buffer
void copy_to_host(void* host_ptr) const;
```

Add the implementations in `src/core/tensor.cpp`:
```cpp
// src/core/tensor.cpp
// ... include hip/hip_runtime.h if not already there
#if USE_HIP_BACKEND
#include <hip/hip_runtime.h>
#endif

// ... inside namespace vesper

void Tensor::copy_from_host(const void* host_ptr) {
    const size_t size_bytes = this->numel() * GetDTypeSize(this->dtype_);
    switch (this->device()) {
        case Device::HIP:
#if USE_HIP_BACKEND
            hipMemcpy(this->data_ptr<void>(), host_ptr, size_bytes, hipMemcpyHostToDevice);
#else
            throw std::runtime_error("HIP backend not enabled.");
#endif
            break;
        case Device::CPU:
            memcpy(this->data_ptr<void>(), host_ptr, size_bytes);
            break;
        default:
            throw std::runtime_error("Device not supported for copy_from_host.");
    }
}

void Tensor::copy_to_host(void* host_ptr) const {
    const size_t size_bytes = this->numel() * GetDTypeSize(this->dtype_);
    switch (this->device()) {
        case Device::HIP:
#if USE_HIP_BACKEND
            hipMemcpy(host_ptr, this->data_ptr<const void>(), size_bytes, hipMemcpyDeviceToHost);
#else
            throw std::runtime_error("HIP backend not enabled.");
#endif
            break;
        case Device::CPU:
            memcpy(host_ptr, this->data_ptr<const void>(), size_bytes);
            break;
        default:
            throw std::runtime_error("Device not supported for copy_to_host.");
    }
}
```

### Step 3.2: Create the Generic Element-wise HIP Kernel

This kernel will perform an operation on each element of two input arrays and store the result in an output array. Using a C++ functor as a template parameter makes it highly reusable.

Create the directory `src/ops/hip/` and the file `elementwise.hip` inside it.
```sh
mkdir -p src/ops/hip
```
Create `src/ops/hip/elementwise.hip`:
```cpp
// src/ops/hip/elementwise.hip
#include <hip/hip_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/ops/elementwise.h> // We will create this next
#include <functional>

namespace vesper::ops {

// Generic kernel for any element-wise binary operation
template <typename T, typename Op>
__global__ void elementwise_binary_kernel(const T* a, const T* b, T* out, size_t n, Op op) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = op(a[idx], b[idx]);
    }
}

// C++ dispatch function that launches the 'add' kernel
void add_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    // For now, assume float32
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("Only Float32 is supported for now.");
    }

    const int threads_per_block = 256;
    const int num_blocks = (a.numel() + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(
        elementwise_binary_kernel<float, std::plus<float>>,
        dim3(num_blocks),
        dim3(threads_per_block),
        0, 0, // shared mem bytes, stream
        a.data_ptr<const float>(),
        b.data_ptr<const float>(),
        out.data_ptr<float>(),
        a.numel(),
        std::plus<float>()
    );
}

} // namespace vesper::ops
```

### Step 3.3: Create the C++ Operation Interface

We need a C++ header to declare the dispatch function and an `add` function that tensors can call.

Create `include/vesper/ops/elementwise.h`:
```cpp
// include/vesper/ops/elementwise.h
#pragma once

namespace vesper {

class Tensor; // Forward declaration

namespace ops {
    // The public-facing function for addition
    Tensor add(const Tensor& a, const Tensor& b);

    // The backend-specific dispatch function (implemented in .hip file)
    void add_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out);
}
}
```

Create `src/ops/elementwise.cpp` to implement the generic `add` function:
```cpp
// src/ops/elementwise.cpp
#include <vesper/ops/elementwise.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <stdexcept>

namespace vesper::ops {

Tensor add(const Tensor& a, const Tensor& b) {
    // --- Pre-condition checks ---
    if (a.shape() != b.shape() || a.device() != b.device()) {
        throw std::runtime_error("Tensor shapes or devices do not match for add operation.");
    }

    // Create a result tensor with the same properties
    Tensor result = empty(a.shape(), a.dtype(), a.device());

    // --- Dispatch to the correct backend ---
    switch(a.device()) {
        case Device::HIP:
            add_hip_dispatch(a, b, result);
            break;
        default:
            throw std::runtime_error("Device not supported for add operation.");
    }

    return result;
}

} // namespace vesper::ops
```

### Step 3.4: Update `src/CMakeLists.txt`

You need to tell CMake to compile the `.hip` file and the new `.cpp` file.

```cmake
# Vesper/src/CMakeLists.txt
# ...

target_sources(vesper PRIVATE
    core/storage.cpp
    core/tensor.cpp
    ops/elementwise.cpp     # Add this
    ops/hip/elementwise.hip # And this
)
# ...
# Add include directory for ops
target_include_directories(vesper
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src # Allows ops/hip to find ops/elementwise.h
)
```

## 4. Code Structure Suggestions

-   **Separation of Concerns**:
    -   `.hip` file: Contains only CUDA/HIP C++ code (kernels and their launchers).
    -   `ops/elementwise.cpp`: Contains backend-agnostic logic (error checks, creating result tensor).
    -   `ops/elementwise.h`: The public API for the operation.
-   **Kernel Genericity**: The `elementwise_binary_kernel` is generic. You can reuse it for subtraction, multiplication, etc., just by passing a different functor (e.g., `std::minus<float>`).
-   **Dispatch Mechanism**: The `add` function acts as a dispatcher. As you add CPU and CUDA backends, you'll add more cases to its `switch` statement.

## 5. Potential Pitfalls

-   **HIP Compilation**: If CMake doesn't compile the `.hip` file correctly, it's likely because `find_package(amdhip-runtime)` didn't run or failed. Ensure your ROCm environment is set up.
-   **GPU Errors**: `hipLaunchKernelGGL` is an asynchronous call. Errors that happen *on the GPU* won't be reported immediately. You can add `hipDeviceSynchronize()` after the kernel launch during debugging to make it a blocking call and catch kernel errors.
-   **Include Paths**: The `.hip` file needs to include a C++ header from another directory. The `target_include_directories(... PRIVATE ...)` command in CMake is essential to make this work.

## 6. Integration and Verification

Create a test that performs an end-to-end `add` operation on the GPU.

### Step 6.1: Create `tests/test_elementwise_ops.cpp`

```cpp
// tests/test_elementwise_ops.cpp
#include <vesper/ops/elementwise.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

void test_add_op() {
#if USE_HIP_BACKEND
    std::cout << "Testing element-wise add operation..." << std::endl;

    const std::vector<int64_t> shape = {2, 2};
    const vesper::DType dtype = vesper::DType::Float32;
    const vesper::Device device = vesper::Device::HIP;

    // 1. Prepare host data
    std::vector<float> a_host = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_host = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> result_host(4);

    // 2. Create tensors and copy data to device
    vesper::Tensor a = vesper::empty(shape, dtype, device);
    vesper::Tensor b = vesper::empty(shape, dtype, device);
    a.copy_from_host(a_host.data());
    b.copy_from_host(b_host.data());

    // 3. Perform the operation
    vesper::Tensor c = vesper::ops::add(a, b);

    // 4. Copy result back to host
    c.copy_to_host(result_host.data());

    // 5. Verify the result
    for (size_t i = 0; i < a_host.size(); ++i) {
        const float expected = a_host[i] + b_host[i];
        assert(std::fabs(result_host[i] - expected) < 1e-6);
    }

    std::cout << "Element-wise add test passed!" << std::endl;
#else
    std::cout << "Skipping element-wise add test (HIP backend disabled)." << std::endl;
#endif
}

int main() {
    test_add_op();
    return 0;
}
```

### Step 6.2: Add the Test to `tests/CMakeLists.txt`
```cmake
# Vesper/tests/CMakeLists.txt
# ...

add_executable(elementwise_op_tests test_elementwise_ops.cpp)
target_link_libraries(elementwise_op_tests PRIVATE vesper)
add_test(NAME ElementwiseOpTests COMMAND elementwise_op_tests)
```

### Step 6.3: Build and Run
```sh
cd /path/to/vesper/build
cmake ..
make -j
ctest --verbose
```
**Expected Output:**
You should see all four tests passing. The output of the `ElementwiseOpTests` should show the successful test message. This confirms you can create tensors, move data to the GPU, execute a kernel, and retrieve the results.

This is a major milestone. You now have a working pipeline for GPU computation.
