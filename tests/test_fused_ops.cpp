/**
 * @file test_fused_ops.cpp
 * @brief Tests for fused GPU operations
 */

#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/ops/fused_ops.h>
#include <vesper/ops/normalization.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/random.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

#if defined(USE_HIP_BACKEND)
#include <hip/hip_runtime.h>
#endif

using namespace vesper;

// Helper for approximate comparison
bool allclose(const Tensor& a, const Tensor& b, float rtol = 0.01f, float atol = 1e-4f) {
    Tensor a_cpu = a.to(Device::CPU);
    Tensor b_cpu = b.to(Device::CPU);
    
    const float* pa = a_cpu.data_ptr<float>();
    const float* pb = b_cpu.data_ptr<float>();
    
    for (size_t i = 0; i < a_cpu.numel(); ++i) {
        float diff = std::abs(pa[i] - pb[i]);
        float tolerance = atol + rtol * std::max(std::abs(pa[i]), std::abs(pb[i]));
        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << pa[i] << " vs " << pb[i] 
                      << " (diff=" << diff << ", tol=" << tolerance << ")" << std::endl;
            return false;
        }
    }
    return true;
}

void test_fused_rmsnorm_linear_correctness() {
    std::cout << "Testing fused RMSNorm+Linear correctness..." << std::endl;
    
    Device device = Device::CPU;
#if defined(USE_HIP_BACKEND)
    device = Device::HIP;
#endif
    
    // Test configuration
    int64_t batch = 2;
    int64_t seq = 4;
    int64_t dim_in = 64;
    int64_t dim_out = 32;
    float eps = 1e-5f;
    
    // Create inputs with random values
    Tensor input = vesper::empty({batch, seq, dim_in}, DType::Float32, device, false);
    ops::normal_(input, 0.0f, 1.0f);
    
    Tensor norm_weight = vesper::empty({dim_in}, DType::Float32, device, false);
    ops::normal_(norm_weight, 0.0f, 0.1f);
    
    Tensor linear_weight = vesper::empty({dim_out, dim_in}, DType::Float32, device, false);
    ops::normal_(linear_weight, 0.0f, 0.1f);
    
    Tensor bias = vesper::empty({dim_out}, DType::Float32, device, false);
    ops::normal_(bias, 0.0f, 0.1f);
    
    // Reference: unfused computation
    std::vector<int64_t> normalized_shape = {dim_in};
    Tensor normalized = ops::rms_norm(input, normalized_shape, norm_weight, eps);
    Tensor ref_output = ops::matmul(normalized, linear_weight.transpose(-2, -1));
    ref_output = ops::add(ref_output, bias);
    
    // Fused computation
    Tensor fused_output = ops::fused_rmsnorm_linear(input, norm_weight, linear_weight, bias, eps);
    
    // Compare
    std::cout << "  Reference output shape: " << ref_output.shape()[0] << " x " 
              << ref_output.shape()[1] << " x " << ref_output.shape()[2] << std::endl;
    std::cout << "  Fused output shape: " << fused_output.shape()[0] << " x "
              << fused_output.shape()[1] << " x " << fused_output.shape()[2] << std::endl;
    
    // Move to CPU for comparison
    Tensor ref_cpu = ref_output.to(Device::CPU);
    Tensor fused_cpu = fused_output.to(Device::CPU);
    
    bool match = allclose(ref_cpu, fused_cpu, 0.01f, 1e-3f);
    
    if (match) {
        std::cout << "  PASSED - Fused output matches reference!" << std::endl;
    } else {
        std::cout << "  FAILED - Output mismatch!" << std::endl;
        
        // Print first few values
        const float* ref_data = ref_cpu.data_ptr<float>();
        const float* fused_data = fused_cpu.data_ptr<float>();
        std::cout << "  First 10 values:" << std::endl;
        for (int i = 0; i < 10 && i < ref_cpu.numel(); ++i) {
            std::cout << "    [" << i << "] ref=" << ref_data[i] 
                      << " fused=" << fused_data[i] << std::endl;
        }
    }
    
    assert(match);
}

void test_fused_rmsnorm_linear_performance() {
    std::cout << "\nTesting fused RMSNorm+Linear performance..." << std::endl;
    
    Device device = Device::CPU;
#if defined(USE_HIP_BACKEND)
    device = Device::HIP;
#endif
    
    // Larger test for performance
    int64_t batch = 2;
    int64_t seq = 512;
    int64_t dim = 1024;
    int64_t hidden_dim = 4096;
    float eps = 1e-5f;
    int num_iters = 100;
    
    Tensor input = vesper::empty({batch, seq, dim}, DType::Float32, device, false);
    ops::normal_(input, 0.0f, 1.0f);
    
    Tensor norm_weight = vesper::empty({dim}, DType::Float32, device, false);
    ops::normal_(norm_weight, 0.0f, 0.1f);
    
    Tensor linear_weight = vesper::empty({hidden_dim, dim}, DType::Float32, device, false);
    ops::normal_(linear_weight, 0.0f, 0.1f);
    
    Tensor bias = vesper::empty({hidden_dim}, DType::Float32, device, false);
    ops::normal_(bias, 0.0f, 0.1f);
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        Tensor out = ops::fused_rmsnorm_linear(input, norm_weight, linear_weight, bias, eps);
    }
    
#if defined(USE_HIP_BACKEND)
    hipDeviceSynchronize();
#endif
    
    // Benchmark fused
    auto start_fused = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i) {
        Tensor out = ops::fused_rmsnorm_linear(input, norm_weight, linear_weight, bias, eps);
    }
#if defined(USE_HIP_BACKEND)
    hipDeviceSynchronize();
#endif
    auto end_fused = std::chrono::high_resolution_clock::now();
    float fused_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_fused - start_fused).count() / 1000.0f;
    
    // Warmup unfused
    std::vector<int64_t> normalized_shape = {dim};
    for (int i = 0; i < 10; ++i) {
        Tensor normalized = ops::rms_norm(input, normalized_shape, norm_weight, eps);
        Tensor out = ops::matmul(normalized, linear_weight.transpose(-2, -1));
        out = ops::add(out, bias);
    }
    
#if defined(USE_HIP_BACKEND)
    hipDeviceSynchronize();
#endif
    
    // Benchmark unfused
    auto start_unfused = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i) {
        Tensor normalized = ops::rms_norm(input, normalized_shape, norm_weight, eps);
        Tensor out = ops::matmul(normalized, linear_weight.transpose(-2, -1));
        out = ops::add(out, bias);
    }
#if defined(USE_HIP_BACKEND)
    hipDeviceSynchronize();
#endif
    auto end_unfused = std::chrono::high_resolution_clock::now();
    float unfused_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_unfused - start_unfused).count() / 1000.0f;
    
    std::cout << "  Fused: " << fused_ms << " ms for " << num_iters << " iterations" << std::endl;
    std::cout << "  Unfused: " << unfused_ms << " ms for " << num_iters << " iterations" << std::endl;
    std::cout << "  Speedup: " << (unfused_ms / fused_ms) << "x" << std::endl;
}

int main() {
    try {
        std::cout << "=== Fused Operations Tests ===" << std::endl;
        
#if defined(USE_HIP_BACKEND)
        std::cout << "Running on HIP device" << std::endl;
#else
        std::cout << "Running on CPU" << std::endl;
#endif
        
        test_fused_rmsnorm_linear_correctness();
        test_fused_rmsnorm_linear_performance();
        
        std::cout << "\n=== All tests passed! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
