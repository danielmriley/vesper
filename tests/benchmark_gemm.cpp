#include <vesper/core/factories.h>
#include <vesper/core/reference_ops.h>
#include <vesper/ops/gemm.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

#if defined(USE_HIP_BACKEND)
#include <hip/hip_runtime.h>
#endif

#if defined(USE_CUDA_BACKEND)
#include <cuda_runtime.h>
#endif

// Reference naive tiled kernel launch (simulated here by calling cpu reference, but we want to measure GPU kernel)
// Actually, since we overwrote the dispatch function, we can only benchmark the CURRENT kernel.
// To benchmark against the naive tiled version, we would have needed to keep it under a different name.
// However, we can benchmark against `reference::gemm` (CPU) to show massive speedup, 
// or just report absolute GFLOPS.

// Absolute GFLOPS = (2 * M * N * K) / (time_seconds * 1e9)

void benchmark_gemm(int M, int N, int K) {
    std::cout << "Benchmarking GEMM " << M << "x" << N << "x" << K << "..." << std::endl;

#if defined(USE_HIP_BACKEND)
    vesper::Device device = vesper::Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    vesper::Device device = vesper::Device::CUDA;
#else
    vesper::Device device = vesper::Device::CPU;
#endif

    auto A = vesper::empty({M, K}, vesper::DType::Float32, device);
    auto B = vesper::empty({K, N}, vesper::DType::Float32, device);
    
    // Warmup
    auto C = vesper::ops::matmul(A, B);
    
    // Benchmark
    int iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        auto C_loop = vesper::ops::matmul(A, B);
        // Ensure completion for timing
        // C_loop.copy_to_host(nullptr); // Invalid
        
        // Sync by copying one element
        float val;
        // We can't easily copy one element without slicing which is slow?
        // Or just copy the whole tensor to a dummy buffer if small.
        // For benchmark we want to measure kernel time. 
        // Allocation overhead is included.
        
        // Device synchronization is better.
#if defined(USE_CUDA_BACKEND)
        cudaDeviceSynchronize();
#elif defined(USE_HIP_BACKEND)
        hipDeviceSynchronize();
#endif
    }
    
    // Sync last op
    std::vector<float> dummy(1);
    // Just copy 1 element to force sync
    // Actually, `C_loop` goes out of scope, freeing memory.
    // We need to keep result alive or sync explicitly.
    
    // Better benchmark loop:
    auto C_out = vesper::empty({M, N}, vesper::DType::Float32, device);
    
    // Re-run with pre-allocated output? `matmul` allocates. 
    // `gemm_dispatch` takes output. We can't call dispatch directly easily from here without including internal headers.
    // We'll time `matmul` which includes cached allocation overhead.
    
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    double seconds = diff.count() / iterations;
    
    double gflops = (2.0 * M * N * K) * 1e-9;
    double perf = gflops / seconds;
    
    std::cout << "  Avg Time: " << std::fixed << std::setprecision(4) << seconds * 1000 << " ms" << std::endl;
    std::cout << "  Performance: " << perf << " GFLOPS" << std::endl;
}

int main() {
    // Small
    benchmark_gemm(256, 256, 256);
    // Medium
    benchmark_gemm(1024, 1024, 1024);
    // Large (if memory allows)
    // benchmark_gemm(4096, 4096, 4096);
    return 0;
}
