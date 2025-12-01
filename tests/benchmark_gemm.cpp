#include <vesper/core/factories.h>
#include <vesper/core/reference_ops.h>
#include <vesper/ops/gemm.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <hip/hip_runtime.h>

using namespace vesper;

void benchmark_gemm(int M, int N, int K, int warmup = 5, int iterations = 20) {
    std::cout << "\nGEMM [" << M << " x " << K << "] @ [" << K << " x " << N << "] -> [" << M << " x " << N << "]" << std::endl;

    Device device = Device::HIP;
    
    // Create tensors
    auto A = randn({M, K}, DType::Float32, device);
    auto B = randn({K, N}, DType::Float32, device);
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        auto C = ops::matmul(A, B);
    }
    hipDeviceSynchronize();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto C = ops::matmul(A, B);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double time_per_iter_ms = time_ms / iterations;
    
    // GFLOPS = 2*M*N*K / time_seconds / 1e9
    double gflops = (2.0 * M * N * K) / (time_per_iter_ms * 1e-3) / 1e9;
    
    // Memory bandwidth (simplified: read A + B, write C)
    double bytes = (M * K + K * N + M * N) * sizeof(float);
    double bandwidth_gbps = (bytes / 1e9) / (time_per_iter_ms * 1e-3);
    
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << time_per_iter_ms << " ms" << std::endl;
    std::cout << "  Performance: " << std::setprecision(1) << gflops << " GFLOPS" << std::endl;
    std::cout << "  Memory BW: " << std::setprecision(1) << bandwidth_gbps << " GB/s" << std::endl;
    
    // RX 6950 XT theoretical: 23.6 TFLOPS FP32, 576 GB/s
    double efficiency = (gflops / 23600.0) * 100;
    std::cout << "  GPU Efficiency: " << std::setprecision(1) << efficiency << "% of peak" << std::endl;
}

int main() {
    std::cout << "=== GEMM Benchmark ===" << std::endl;
    std::cout << "Hardware: AMD RX 6950 XT (23.6 TFLOPS FP32, 576 GB/s)" << std::endl;
    
    // Transformer typical shapes for 271M model (dim=1024, ffn=2816, vocab=32000)
    // batch*seq = 4*256 = 1024 tokens
    
    std::cout << "\n--- Linear Layer GEMMs (most common) ---" << std::endl;
    
    // Q/K/V projections: [batch*seq, dim] @ [dim, dim] -> [1024, 1024] @ [1024, 1024]
    benchmark_gemm(1024, 1024, 1024);
    
    // FFN up/gate: [batch*seq, dim] @ [dim, ffn_dim] -> [1024, 1024] @ [1024, 2816]
    benchmark_gemm(1024, 2816, 1024);
    
    // FFN down: [batch*seq, ffn_dim] @ [ffn_dim, dim] -> [1024, 2816] @ [2816, 1024]
    benchmark_gemm(1024, 1024, 2816);
    
    // Output projection (vocab): [batch*seq, dim] @ [dim, vocab] -> [1024, 1024] @ [1024, 32000]
    benchmark_gemm(1024, 32000, 1024);
    
    std::cout << "\n--- Attention Score GEMMs (batched) ---" << std::endl;
    // Q @ K^T: for each head [seq, head_dim] @ [head_dim, seq] -> [256, 64] @ [64, 256]
    // With batch*heads = 64 (4 batches * 16 heads)
    benchmark_gemm(256, 256, 64);  // Single head
    
    // Attention @ V: [seq, seq] @ [seq, head_dim] -> [256, 256] @ [256, 64]
    benchmark_gemm(256, 64, 256);  // Single head
    
    std::cout << "\n--- Large GEMMs ---" << std::endl;
    benchmark_gemm(2048, 2048, 2048);
    benchmark_gemm(4096, 4096, 4096);
    
    std::cout << "\n=== Benchmark Complete ===" << std::endl;
    return 0;
}
