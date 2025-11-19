
# Vesper Build Plan - Chapter 18.1: Naive CPU Backend

## 1. Goal

Flesh out the CPU backend stub with a simple, correct, single-threaded implementation for all required operations. This provides a baseline that works on any machine, which we can then optimize in subsequent chapters. The focus is on correctness, not performance.

## 2. Detailed Steps

### Step 2.1: Implement Naive CPU `matmul`

We will implement the classic triple-loop matrix multiplication.

In `src/ops/cpu/gemm_cpu.cpp`, modify the dispatch function:
```cpp
// src/ops/cpu/gemm_cpu.cpp
#include <vesper/ops/gemm.h>
#include <vesper/core/tensor.h>
#include <vector>

namespace vesper::ops {

// A naive, single-threaded GEMM implementation.
void gemm_cpu_naive(const float* A, const float* B, float* C, int M, int N, int K) {
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

void gemm_cpu_dispatch(const Tensor& a, const Tensor& b, Tensor& c) {
    const int M = a.shape()[0];
    const int N = b.shape()[1];
    const int K = a.shape()[1]; // Corrected from original plan

    gemm_cpu_naive(a.data_ptr<const float>(), b.data_ptr<const float>(), c.data_ptr<float>(), M, N, K);
}

} // namespace vesper::ops
```
*Note: The `matmul` dispatch in `src/ops/gemm.cpp` should already be configured to call this function when the device is `CPU`.*

### Step 2.2: Implement Naive CPU Element-wise and Other Ops

Similarly, create CPU dispatch functions for other operations. These are typically simple loops.

Create `src/ops/cpu/elementwise_cpu.cpp`:
```cpp
// src/ops/cpu/elementwise_cpu.cpp
#include <vesper/ops/elementwise.h>

namespace vesper::ops {

void add_cpu_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    auto* a_ptr = a.data_ptr<float>();
    auto* b_ptr = b.data_ptr<float>();
    auto* out_ptr = out.data_ptr<float>();
    for(size_t i = 0; i < a.numel(); ++i) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}
// ... Implement sub_cpu_dispatch, mul_cpu_dispatch ...
}
```
Update the main C++ op dispatchers (e.g., `src/ops/elementwise.cpp`) to call these new CPU functions in the `case Device::CPU:` block.

## 3. Verification

The test is to ensure this baseline implementation is correct. We can re-use the existing `matmul` CPU test.

### Step 3.1: Run the CPU `matmul` Test

Configure the build for the CPU backend and run the tests.
```sh
cd /path/to/vesper/build
cmake .. -DUSE_HIP=OFF -DUSE_CPU=ON
make -j
ctest --verbose -R MatmulOpTests # Run only the matmul test
```
The test `test_matmul_cpu` from Chapter 18 should pass, as it compares the output against a trusted naive implementation (which is now our actual implementation). This confirms the CPU backend is wired correctly and provides correct, albeit slow, results.
