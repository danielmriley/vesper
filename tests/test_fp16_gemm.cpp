/**
 * @file test_fp16_gemm.cpp
 * @brief Tests for FP16 GEMM kernel
 */

#include <vesper/vesper.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace vesper;

void test_fp16_gemm_correctness() {
    std::cout << "Testing FP16 GEMM correctness...\n";
    
    int M = 128, K = 256, N = 128;
    
    // Create FP32 tensors for reference
    auto a_fp32 = randn({M, K}, DType::Float32, Device::HIP);
    auto b_fp32 = randn({K, N}, DType::Float32, Device::HIP);
    
    // Reference FP32 result
    auto c_ref = ops::matmul(a_fp32, b_fp32);
    
    // Convert to FP16
    auto a_fp16 = a_fp32.to(DType::Float16);
    auto b_fp16 = b_fp32.to(DType::Float16);
    
    // FP16 GEMM
    auto c_fp16 = ops::matmul(a_fp16, b_fp16);
    
    // Convert back to FP32 for comparison
    auto c_fp16_as_fp32 = c_fp16.to(DType::Float32);
    
    // Compare (allow larger tolerance due to FP16 precision)
    auto c_ref_cpu = c_ref.to(Device::CPU);
    auto c_test_cpu = c_fp16_as_fp32.to(Device::CPU);
    
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    float* ref_data = c_ref_cpu.data_ptr<float>();
    float* test_data = c_test_cpu.data_ptr<float>();
    
    int num_large_errors = 0;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(ref_data[i] - test_data[i]);
        float rel_diff = diff / (std::abs(ref_data[i]) + 1e-6f);
        if (diff > max_diff) {
            max_diff = diff;
            max_rel_diff = rel_diff;
        }
        if (rel_diff > 0.1f) num_large_errors++;
    }
    
    std::cout << "  Max absolute diff: " << max_diff << "\n";
    std::cout << "  Max relative diff: " << max_rel_diff << "\n";
    std::cout << "  Elements with >10% error: " << num_large_errors << " / " << (M*N) << "\n";
    
    // FP16 has ~3 decimal digits of precision
    // For K=256 accumulation, expected error ~ sqrt(256) * 1e-3 ~ 0.016
    // Allow 5% relative error for most elements
    if (num_large_errors < M * N * 0.01) {
        std::cout << "  PASSED!\n";
    } else {
        std::cout << "  FAILED - too many elements with large error!\n";
    }
}

void test_fp16_gemm_performance() {
    std::cout << "\nTesting FP16 GEMM performance...\n";
    
    // Matrix sizes typical for transformer (batch*seq, dim, dim)
    int M = 1024, K = 1024, N = 1024;
    int warmup = 10;
    int iterations = 100;
    
    // FP32 tensors
    auto a_fp32 = randn({M, K}, DType::Float32, Device::HIP);
    auto b_fp32 = randn({K, N}, DType::Float32, Device::HIP);
    
    // FP16 tensors
    auto a_fp16 = a_fp32.to(DType::Float16);
    auto b_fp16 = b_fp32.to(DType::Float16);
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        auto _ = ops::matmul(a_fp32, b_fp32);
    }
    hipDeviceSynchronize();
    
    // Benchmark FP32
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto _ = ops::matmul(a_fp32, b_fp32);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Warmup FP16
    for (int i = 0; i < warmup; ++i) {
        auto _ = ops::matmul(a_fp16, b_fp16);
    }
    hipDeviceSynchronize();
    
    // Benchmark FP16
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto _ = ops::matmul(a_fp16, b_fp16);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double fp16_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Calculate TFLOPS
    double flops = 2.0 * M * K * N * iterations;
    double fp32_tflops = flops / (fp32_ms * 1e9);
    double fp16_tflops = flops / (fp16_ms * 1e9);
    
    std::cout << "  Matrix size: " << M << "x" << K << " @ " << K << "x" << N << "\n";
    std::cout << "  FP32: " << fp32_ms << " ms, " << fp32_tflops << " TFLOPS\n";
    std::cout << "  FP16: " << fp16_ms << " ms, " << fp16_tflops << " TFLOPS\n";
    std::cout << "  Speedup: " << (fp32_ms / fp16_ms) << "x\n";
}

void test_fp16_gemm_larger() {
    std::cout << "\nTesting FP16 GEMM with larger matrices...\n";
    
    // Size similar to transformer FFN: [batch*seq, dim] @ [dim, 4*dim]
    int M = 2048, K = 1024, N = 4096;  // Like [2*512, 1024] @ [1024, 4096]
    int warmup = 10;
    int iterations = 50;
    
    auto a_fp32 = randn({M, K}, DType::Float32, Device::HIP);
    auto b_fp32 = randn({K, N}, DType::Float32, Device::HIP);
    auto a_fp16 = a_fp32.to(DType::Float16);
    auto b_fp16 = b_fp32.to(DType::Float16);
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        auto _ = ops::matmul(a_fp32, b_fp32);
        auto __ = ops::matmul(a_fp16, b_fp16);
    }
    hipDeviceSynchronize();
    
    // Benchmark FP32
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto _ = ops::matmul(a_fp32, b_fp32);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Benchmark FP16
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto _ = ops::matmul(a_fp16, b_fp16);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double fp16_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    double flops = 2.0 * M * K * N * iterations;
    double fp32_tflops = flops / (fp32_ms * 1e9);
    double fp16_tflops = flops / (fp16_ms * 1e9);
    
    std::cout << "  Matrix size: " << M << "x" << K << " @ " << K << "x" << N << "\n";
    std::cout << "  FP32: " << fp32_ms << " ms, " << fp32_tflops << " TFLOPS\n";
    std::cout << "  FP16: " << fp16_ms << " ms, " << fp16_tflops << " TFLOPS\n";
    std::cout << "  Speedup: " << (fp32_ms / fp16_ms) << "x\n";
}

int main() {
    std::cout << "=== FP16 GEMM Tests ===\n";
    
    test_fp16_gemm_correctness();
    test_fp16_gemm_performance();
    test_fp16_gemm_larger();
    
    std::cout << "\n=== All FP16 GEMM tests completed ===\n";
    return 0;
}
