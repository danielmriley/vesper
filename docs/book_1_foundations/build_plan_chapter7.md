
# Vesper Build Plan - Chapter 7: Tensor Matmul: Integrating the HIP GEMM Kernel

## 1. Goal

Expose the high-performance GEMM kernel from Chapter 6 through a clean, user-facing `vesper::ops::matmul` function. This involves implementing the backend-agnostic C++ logic that validates inputs, creates the output tensor, and dispatches to the correct backend implementation (currently, HIP).

## 2. Prerequisites

- Chapter 6: The `gemm_tiled_kernel` and its C++ launcher `gemm_hip_dispatch` are implemented and tested.
- The interface `include/vesper/ops/gemm.h` is already created.

## 3. Detailed Steps

### Step 3.1: Implement the `matmul` Dispatch Function

The core of this chapter is the `matmul` function. It serves as a guard and a dispatcher, ensuring that only valid inputs are passed to the fast but less forgiving backend kernels.

Create `src/ops/gemm.cpp`:
```cpp
// src/ops/gemm.cpp
#include <vesper/ops/gemm.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <stdexcept>

namespace vesper::ops {

Tensor matmul(const Tensor& a, const Tensor& b) {
    // --- 1. Pre-condition Checks ---
    if (a.device() != b.device()) {
        throw std::runtime_error("Matmul requires tensors to be on the same device.");
    }
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::runtime_error("Matmul currently only supports 2D tensors.");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("Inner dimensions of matrices do not match for matmul.");
    }
    if (!a.is_contiguous() || !b.is_contiguous()) {
        throw std::runtime_error("Matmul currently requires contiguous tensors.");
    }

    // --- 2. Prepare Output Tensor ---
    const int M = a.shape()[0];
    const int K = a.shape()[1]; // a.k.a. b.shape()[0]
    const int N = b.shape()[1];

    Tensor c = empty({M, N}, a.dtype(), a.device());

    // --- 3. Dispatch to Backend-Specific Implementation ---
    switch (a.device()) {
        case Device::HIP:
#if USE_HIP_BACKEND
            gemm_hip_dispatch(a, b, c);
#else
            throw std::runtime_error("HIP backend not enabled during build, but required for matmul.");
#endif
            break;
        
        case Device::CPU:
        case Device::CUDA:
            throw std::runtime_error("CPU and CUDA backends for matmul are not yet implemented.");

        default:
            throw std::runtime_error("Unsupported device for matmul.");
    }

    return c;
}

} // namespace vesper::ops
```

### Step 3.2: Update `src/CMakeLists.txt`

Add the new `gemm.cpp` file to the `vesper` library sources.

```cmake
# Vesper/src/CMakeLists.txt
target_sources(vesper PRIVATE
    core/storage.cpp
    core/tensor.cpp
    ops/elementwise.cpp
    ops/gemm.cpp             # Add this line
    ops/hip/elementwise.hip
    ops/hip/gemm.hip
)
```

## 4. Code Structure Suggestions

-   **Robust Pre-conditions**: The checks at the beginning of `matmul` are vital. They provide clear, understandable error messages to the user, preventing cryptic GPU errors or crashes that would occur if invalid data were passed to the kernel.
-   **Future-Proofing**: The `switch` statement makes it clear where to add support for `CPU` and `CUDA` backends in the future. The function's logic is already prepared for them.
-   **Contiguous Check**: The check for `is_contiguous()` is an important simplification for now. Handling non-contiguous tensors (e.g., from a transpose operation) requires more complex kernels or preprocessing steps. Explicitly requiring contiguity is a safe starting point.

## 5. Potential Pitfalls

-   **Error Message Clarity**: Vague error messages like "Matmul failed" are unhelpful. The specific checks for dimension mismatch, device mismatch, etc., are much better for the end-user.
-   **Forgetting to Dispatch**: A common mistake is to put all the logic in the dispatch function and forget to call it from the main `matmul` function, resulting in an empty or incorrect output tensor.
-   **Batch Dimensions**: This implementation will fail for batch matrix multiplication (e.g., a tensor with shape `[Batch, M, K]`). The check for `shape.size() != 2` is the guard against this. Supporting batch matmul is a significant future extension.

## 6. Integration and Verification

We will now write a test that uses the public `vesper::ops::matmul` API, simulating how a user would interact with the library.

### Step 6.1: Create `tests/test_matmul_op.cpp`

This test is very similar to the one in Chapter 6, but it validates the full public API path.

```cpp
// tests/test_matmul_op.cpp
#include <vesper/ops/gemm.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

// Re-use the naive CPU GEMM for verification
void naive_gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void test_matmul_public_api() {
#if USE_HIP_BACKEND
    std::cout << "Testing public matmul API..." << std::endl;

    int M = 64, K = 32, N = 48;

    // 1. Prepare host data and compute CPU ground truth
    std::vector<float> h_A(M * K), h_B(K * N), h_C_cpu(M * N), h_C_gpu(M * N);
    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& val : h_A) val = dist(rng);
    for (float& val : h_B) val = dist(rng);
    naive_gemm_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);

    // 2. Prepare device tensors
    vesper::Tensor d_A = vesper::empty({M, K}, vesper::DType::Float32, vesper::Device::HIP);
    vesper::Tensor d_B = vesper::empty({K, N}, vesper::DType::Float32, vesper::Device::HIP);
    d_A.copy_from_host(h_A.data());
    d_B.copy_from_host(h_B.data());

    // 3. Call the public matmul function
    vesper::Tensor d_C = vesper::ops::matmul(d_A, d_B);

    // 4. Copy result back and verify
    d_C.copy_to_host(h_C_gpu.data());

    int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-3) {
            errors++;
        }
    }
    assert(errors == 0);
    std::cout << "Public matmul API test passed!" << std::endl;
#else
    std::cout << "Skipping public matmul API test (HIP backend disabled)." << std::endl;
#endif
}

int main() {
    test_matmul_public_api();
    return 0;
}
```

### Step 6.2: Add the Test to `tests/CMakeLists.txt`
```cmake
# Vesper/tests/CMakeLists.txt
# ...
add_executable(matmul_op_tests test_matmul_op.cpp)
target_link_libraries(matmul_op_tests PRIVATE vesper)
add_test(NAME MatmulOpTests COMMAND matmul_op_tests)
```

### Step 6.3: Build and Run
```sh
cd /path/to/vesper/build
cmake ..
make -j
ctest --verbose
```
**Expected Output:**
All tests should pass. The success of `MatmulOpTests` verifies that your GEMM kernel is not only correct but also properly integrated into the library's public API with robust error checking. This is a huge step towards building neural network layers.
