# Vesper Build Plan - Chapter 18.2: Tiled CPU GEMM for Cache Performance

## 1. Goal

Optimize the single-threaded CPU GEMM by introducing tiling. This technique improves performance by maximizing CPU cache utilization, reducing the number of slow reads from main RAM.

## 2. The Tiling Strategy

The logic is identical to the GPU version but adapted for a CPU's memory hierarchy. By processing the matrices in small blocks (tiles) that are likely to fit in the L1 or L2 cache, we ensure that the data needed for the inner loops of the multiplication is already in a fast layer of memory.

## 3. Detailed Steps

### Step 3.1: Implement the Tiled CPU GEMM

We will replace the naive triple-loop in `src/ops/cpu/gemm_cpu.cpp` with a tiled version.

Modify `src/ops/cpu/gemm_cpu.cpp`:
```cpp
// src/ops/cpu/gemm_cpu.cpp

namespace vesper::ops {

// Define a tile size suitable for CPU caches. 16, 32, or 64 are common.
constexpr int TILE_SIZE = 32;

// A tiled, single-threaded GEMM implementation.
void gemm_cpu_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    // Initialize C to zero
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0.0f;
        }
    }
    
    // Iterate over tiles
    for (int i0 = 0; i0 < M; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {
            for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                // Multiply the tiles, accumulating results into C
                for (int i = i0; i < std::min(i0 + TILE_SIZE, M); ++i) {
                    for (int j = j0; j < std::min(j0 + TILE_SIZE, N); ++j) {
                        for (int k = k0; k < std::min(k0 + TILE_SIZE, K); ++k) {
                            C[i * N + j] += A[i * K + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// The dispatch function now calls the tiled version
void gemm_cpu_dispatch(const Tensor& a, const Tensor& b, Tensor& c) {
    // ... get M, N, K, and data pointers ...
    gemm_cpu_tiled(a.data_ptr<const float>(), b.data_ptr<const float>(), c.data_ptr<float>(), M, N, K);
}

} // namespace vesper::ops
```

## 4. Verification

The test remains the same as in Chapter 18.1. The implementation has changed, but the result should be identical. Re-running the `matmul` CPU test will verify that our tiling optimization is numerically correct.

A more advanced verification would be to benchmark this version against the naive implementation from the previous chapter on a large matrix, expecting to see a noticeable performance improvement.

```sh
# In tests/test_matmul_op.cpp, you could add a simple timer
#include <chrono>
auto start = std::chrono::high_resolution_clock::now();
vesper::Tensor c_cpu = vesper::ops::matmul(a_cpu, b_cpu);
auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double, std::milli> ms_double = end - start;
std::cout << "Tiled GEMM took " << ms_double.count() << "ms\n";
```
Comparing this timing to the one from Chapter 18.1 is the true validation of this optimization.

```