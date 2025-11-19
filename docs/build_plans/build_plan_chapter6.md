
# Vesper Build Plan - Chapter 6: High-Performance GEMM: The Custom HIP Kernel

## 1. Goal

Implement a high-performance General Matrix Multiply (GEMM) kernel using HIP. This is the computational heart of a deep learning library. This chapter focuses on the kernel implementation itself, introducing tiling with shared memory to achieve significant speedups over naive approaches. The C++ integration will follow in the next chapter.

## 2. Prerequisites

- Chapter 5: Familiarity with writing, compiling, and launching basic HIP kernels.
- A solid understanding of matrix multiplication `(C = A * B)`.

## 3. The Challenge: Why Naive GEMM is Slow on GPUs

A naive implementation of matrix multiplication uses a triple nested loop. If ported directly to a GPU kernel where each thread computes one element of the output matrix C, it would look like this:

```cpp
// For each element C(row, col)
float sum = 0;
for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col]; // Global memory access
}
C[row * N + col] = sum;
```

For each output element, this requires `2 * K` reads from global GPU memory (DRAM). Global memory is vast but has high latency. The key to GPU performance is to minimize these slow global memory accesses by using the fast, on-chip **shared memory**.

## 4. The Solution: Tiled Matrix Multiplication

We can break the `A` and `B` matrices into smaller blocks, or **tiles**, that can fit into shared memory. A block of threads works together to compute a corresponding tile of the output matrix `C`.

The process for one output tile is:
1.  Load a tile from `A` and a tile from `B` into shared memory.
2.  **Synchronize** the thread block to ensure loading is complete.
3.  Perform a matrix multiplication using the data *in shared memory*. Each thread computes one element of the output tile, accumulating partial sums.
4.  **Synchronize** again before moving to the next pair of tiles.
5.  Repeat until all necessary tiles from `A` and `B` have been processed.
6.  Each thread writes its final computed value to the output matrix `C` in global memory.

This approach dramatically reduces global memory traffic. Instead of `M * N * 2 * K` global reads, we get `M*K + K*N` reads (the cost of loading all tiles once).

## 5. Detailed Steps

### Step 5.1: Create `gemm.hip` and its Interface

First, create the interface header that C++ code will use to interact with our GEMM operation.

Create `include/vesper/ops/gemm.h`:
```cpp
// include/vesper/ops/gemm.h
#pragma once

namespace vesper {

class Tensor; // Forward declaration

namespace ops {
    // The public-facing function for matrix multiplication
    Tensor matmul(const Tensor& a, const Tensor& b);

    // Backend-specific dispatch function (to be implemented in gemm.hip)
    void gemm_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& c);
}
}
```

Now, create the HIP implementation file `src/ops/hip/gemm.hip`:
```cpp
// src/ops/hip/gemm.hip
#include <vesper/ops/gemm.h>
#include <vesper/core/tensor.h>
#include <hip/hip_runtime.h>

namespace vesper::ops {

// Define the size of the tile. This must be a compile-time constant.
// 16 or 32 are common values. Must match the thread block size.
constexpr int TILE_WIDTH = 16;

// Tiled GEMM Kernel for C = A * B
// A is (M x K), B is (K x N), C is (M x N)
template <typename T>
__global__ void gemm_tiled_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    // 1. Thread and block identification
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // 2. Allocate tiles in shared memory
    __shared__ T tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ T tileB[TILE_WIDTH][TILE_WIDTH];

    // Accumulator for the partial sum computed by this thread
    T partial_sum = 0.0;

    // 3. Loop over tiles
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // 4. Load one tile of A and one tile of B into shared memory
        // Each thread in the block loads one element of each tile.
        int a_row = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int a_col = t * TILE_WIDTH + threadIdx.x;
        if (a_row < M && a_col < K) {
            tileA[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }

        int b_row = t * TILE_WIDTH + threadIdx.y;
        int b_col = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (b_row < K && b_col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }

        // 5. Synchronize to ensure all threads have finished loading
        __syncthreads();

        // 6. Multiply the tiles from shared memory and accumulate results
        for (int k = 0; k < TILE_WIDTH; ++k) {
            partial_sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // 7. Synchronize before loading the next tile
        __syncthreads();
    }

    // 8. Write the final result to global memory
    if (row < M && col < N) {
        C[row * N + col] = partial_sum;
    }
}


// The C++ dispatch function that launches the kernel
void gemm_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& c) {
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("GEMM only supports Float32 for now.");
    }
    
    int M = a.shape()[0];
    int K = a.shape()[1];
    int N = b.shape()[1];

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    hipLaunchKernelGGL(
        gemm_tiled_kernel<float>,
        blocks,
        threads,
        0, 0,
        a.data_ptr<const float>(),
        b.data_ptr<const float>(),
        c.data_ptr<float>(),
        M, N, K
    );
}

} // namespace vesper::ops
```

### Step 5.2: Update CMake

Add the new `gemm.hip` file to your `src/CMakeLists.txt`. We will create `gemm.cpp` in the next chapter.

```cmake
# Vesper/src/CMakeLists.txt
target_sources(vesper PRIVATE
    core/storage.cpp
    core/tensor.cpp
    ops/elementwise.cpp
    ops/hip/elementwise.hip
    ops/hip/gemm.hip      # Add this line
)
```

## 6. Potential Pitfalls

-   **Incorrect Indexing**: This is the #1 source of bugs in custom kernels. Double-check the global memory indices (`a_row * K + a_col`) and shared memory indices (`tileA[threadIdx.y][threadIdx.x]`).
-   **Boundary Conditions**: The `if (row < M && col < N)` checks are critical for matrices whose dimensions are not a perfect multiple of `TILE_WIDTH`. Without them, threads would write out of bounds.
-   **Missing `__syncthreads()`**: Forgetting to synchronize after loading into shared memory or after computing partial sums will cause a race condition, leading to incorrect results.
-   **Shared Memory Size**: Shared memory is a limited resource (~48-96 KB per SM). `TILE_WIDTH=16` uses `16*16*4 (float) * 2 (matrices) = 2 KB`, which is safe. Larger tile sizes can improve performance but risk exceeding the limit.

## 7. Integration and Verification

Even though we haven't created `ops::matmul` yet, we can test the `gemm_hip_dispatch` function directly to validate our kernel. We'll compare its output to a simple, trustworthy CPU implementation.

### Step 7.1: Create `tests/test_gemm_kernel.cpp`

```cpp
// tests/test_gemm_kernel.cpp
#include <vesper/core/factories.h>
#include <vesper/ops/gemm.h> // For the dispatch function
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

// Naive, single-threaded CPU GEMM for verification
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

void test_gemm() {
#if USE_HIP_BACKEND
    std::cout << "Testing GEMM kernel..." << std::endl;

    int M = 32, K = 48, N = 64; // Non-square, non-multiple-of-16 dimensions

    // 1. Prepare host data with random values
    std::vector<float> h_A(M * K), h_B(K * N), h_C_cpu(M * N), h_C_gpu(M * N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& val : h_A) val = dist(rng);
    for (float& val : h_B) val = dist(rng);

    // 2. Compute ground truth on CPU
    naive_gemm_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);

    // 3. Prepare device tensors and copy data
    vesper::Tensor d_A = vesper::empty({M, K}, vesper::DType::Float32, vesper::Device::HIP);
    vesper::Tensor d_B = vesper::empty({K, N}, vesper::DType::Float32, vesper::Device::HIP);
    vesper::Tensor d_C = vesper::empty({M, N}, vesper::DType::Float32, vesper::Device::HIP);
    d_A.copy_from_host(h_A.data());
    d_B.copy_from_host(h_B.data());

    // 4. Launch the kernel via the dispatch function
    vesper::ops::gemm_hip_dispatch(d_A, d_B, d_C);
    
    // 5. Copy result back to host
    d_C.copy_to_host(h_C_gpu.data());

    // 6. Verify GPU result against CPU ground truth
    int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-3) { // Use a tolerance for FP math
            errors++;
        }
    }
    assert(errors == 0);
    if (errors > 0) {
        std::cerr << "GEMM test failed with " << errors << " errors." << std::endl;
    } else {
        std::cout << "GEMM kernel test passed!" << std::endl;
    }
#else
    std::cout << "Skipping GEMM kernel test (HIP backend disabled)." << std::endl;
#endif
}

int main() {
    test_gemm();
    return 0;
}
```

### Step 7.2: Add Test to `tests/CMakeLists.txt`
```cmake
# Vesper/tests/CMakeLists.txt
# ...
add_executable(gemm_kernel_tests test_gemm_kernel.cpp)
target_link_libraries(gemm_kernel_tests PRIVATE vesper)
add_test(NAME GemmKernelTests COMMAND gemm_kernel_tests)
```

### Step 7.3: Build and Run
```sh
cd /path/to/vesper/build && cmake .. && make -j && ctest --verbose
```
A passing test confirms your tiled GEMM kernel is correctly implemented, even with non-ideal matrix dimensions. You are now ready to expose this powerful kernel through a user-friendly `matmul` function.
