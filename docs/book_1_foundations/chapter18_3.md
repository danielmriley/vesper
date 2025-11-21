
# Vesper Build Plan - Chapter 18.3: Multi-Threaded CPU GEMM

## 1. Goal

Parallelize the tiled CPU GEMM from the previous chapter using `std::thread`. This will distribute the work across all available CPU cores, providing the final and most significant performance boost for the CPU backend.

## 2. The Multi-Threading Strategy

We will parallelize the outermost loop of the matrix multiplication. The computation of each row (or a chunk of rows) of the output matrix `C` is independent of the others. This makes it an "embarrassingly parallel" problem.

We will:
1.  Query the number of available hardware threads.
2.  Divide the rows of the output matrix `C` among the threads.
3.  Launch each thread to compute its assigned chunk of rows using the tiled algorithm.
4.  Wait for all threads to complete.

## 3. Detailed Steps

### Step 3.1: Parallelize the `gemm_cpu_dispatch` Function

Modify `src/ops/cpu/gemm_cpu.cpp` to manage and launch threads.

```cpp
// src/ops/cpu/gemm_cpu.cpp
#include <thread> // New include

namespace vesper::ops {

// The tiled worker function from Ch 18.2, now takes a row range
void gemm_cpu_tiled_worker(const float* A, const float* B, float* C, int M, int N, int K, int start_row, int end_row) {
    // ... same tiled implementation as before, but it only iterates from
    // start_row to end_row.
}


void gemm_cpu_dispatch(const Tensor& a, const Tensor& b, Tensor& c) {
    const int M = a.shape()[0];
    const int N = b.shape()[1];
    const int K = a.shape()[1];

    const float* a_ptr = a.data_ptr<const float>();
    const float* b_ptr = b.data_ptr<const float>();
    float* c_ptr = c.data_ptr<float>();

    // 1. Determine number of threads
    const unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // 2. Divide the work
    int rows_per_thread = (M + num_threads - 1) / num_threads;

    // 3. Launch threads
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = std::min(start_row + rows_per_thread, M);
        
        if (start_row < end_row) {
            threads.emplace_back(gemm_cpu_tiled_worker, a_ptr, b_ptr, c_ptr, M, N, K, start_row, end_row);
        }
    }

    // 4. Wait for all threads to finish
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

} // namespace vesper::ops
```

## 4. Verification

The correctness test is the same as before; the result must be numerically identical to the single-threaded versions. The real verification is another benchmark.

### Step 4.1: Benchmark the Multi-Threaded Version

Using the same timing code from the previous chapter, run the `matmul` test again. You should see a significant speedup compared to the single-threaded tiled version, especially on machines with many cores. The speedup may not be linear with the number of cores due to overhead, but it should be substantial.

A passing correctness test and a measured performance improvement validate this final stage of CPU optimization. The Vesper library now has a genuinely performant CPU backend for its most critical operation.
