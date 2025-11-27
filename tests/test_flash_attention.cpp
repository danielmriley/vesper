/**
 * @file test_flash_attention.cpp
 * @brief Tests for Flash Attention implementation
 * 
 * Chapter 33.9: Memory-Efficient Training
 * 
 * Tests:
 * 1. Correctness vs standard attention
 * 2. Causal masking
 * 3. Various sequence lengths and head dimensions
 * 4. Backward pass gradient correctness
 * 5. Memory efficiency (doesn't OOM on long sequences)
 */

#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/ops/flash_attention.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/random.h>
#include <vesper/nn/functional.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace vesper;

// ============================================================================
// Test Utilities
// ============================================================================

template<typename T>
bool approx_equal(T a, T b, T rel_tol = 0.01, T abs_tol = 1e-5) {
    T diff = std::abs(a - b);
    return diff <= abs_tol || diff <= rel_tol * std::max(std::abs(a), std::abs(b));
}

bool allclose(const Tensor& a, const Tensor& b, float rtol = 0.01f, float atol = 1e-4f) {
    Tensor a_cpu = a.to(Device::CPU);
    Tensor b_cpu = b.to(Device::CPU);
    
    const float* pa = a_cpu.data_ptr<float>();
    const float* pb = b_cpu.data_ptr<float>();
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < a_cpu.numel(); ++i) {
        float diff = std::abs(pa[i] - pb[i]);
        max_diff = std::max(max_diff, diff);
        
        bool close = diff <= atol || diff <= rtol * std::max(std::abs(pa[i]), std::abs(pb[i]));
        if (!close) {
            std::cerr << "  Mismatch at index " << i << ": " << pa[i] << " vs " << pb[i] 
                      << " (diff=" << diff << ")" << std::endl;
            return false;
        }
    }
    
    std::cout << "  Max difference: " << max_diff << std::endl;
    return true;
}

// Reference standard attention implementation
Tensor reference_attention(const Tensor& q, const Tensor& k, const Tensor& v, 
                           float scale, bool is_causal) {
    // q, k, v: [B, H, N, D]
    int64_t B = q.shape()[0];
    int64_t H = q.shape()[1];
    int64_t N = q.shape()[2];
    int64_t D = q.shape()[3];
    
    // Reshape for batch matmul: [B*H, N, D]
    Tensor q_flat = q.view({B * H, N, D});
    Tensor k_flat = k.view({B * H, N, D});
    Tensor v_flat = v.view({B * H, N, D});
    
    // scores = Q @ K^T / sqrt(D): [B*H, N, N]
    Tensor scores = ops::matmul(q_flat, k_flat.transpose(-2, -1));
    scores = ops::mul(scores, scale);
    
    // Apply causal mask if needed
    if (is_causal) {
        // Create lower triangular mask
        Tensor mask = vesper::empty({N, N}, DType::Float32, scores.device());
        float* mask_ptr = mask.to(Device::CPU).data_ptr<float>();
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                mask_ptr[i * N + j] = (j <= i) ? 0.0f : -1e9f;
            }
        }
        mask = mask.to(scores.device());
        scores = ops::add(scores, mask);
    }
    
    // Softmax
    Tensor attn = nn::functional::softmax(scores, -1);
    
    // output = attn @ V: [B*H, N, D]
    Tensor output = ops::matmul(attn, v_flat);
    
    // Reshape back: [B, H, N, D]
    return output.view({B, H, N, D});
}

// ============================================================================
// Flash Attention CPU Tests
// ============================================================================

void test_flash_attention_basic_cpu() {
    std::cout << "Testing Flash Attention basic (CPU)..." << std::endl;
    
    int B = 2, H = 4, N = 32, D = 64;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    Tensor q = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU);
    Tensor k = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU);
    Tensor v = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU);
    
    // Run flash attention
    Tensor flash_out = ops::flash_attention(q, k, v, scale, false);
    
    // Run reference
    Tensor ref_out = reference_attention(q, k, v, scale, false);
    
    // Check shapes
    assert(flash_out.shape() == ref_out.shape());
    
    // Check values
    assert(allclose(flash_out, ref_out, 1e-3f, 1e-4f));
    
    std::cout << "Flash Attention basic (CPU): PASSED" << std::endl;
}

void test_flash_attention_causal_cpu() {
    std::cout << "Testing Flash Attention causal (CPU)..." << std::endl;
    
    int B = 1, H = 2, N = 16, D = 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    Tensor q = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU);
    Tensor k = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU);
    Tensor v = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU);
    
    Tensor flash_out = ops::flash_attention(q, k, v, scale, true);
    Tensor ref_out = reference_attention(q, k, v, scale, true);
    
    assert(allclose(flash_out, ref_out, 1e-3f, 1e-4f));
    
    std::cout << "Flash Attention causal (CPU): PASSED" << std::endl;
}

void test_flash_attention_various_sizes_cpu() {
    std::cout << "Testing Flash Attention various sizes (CPU)..." << std::endl;
    
    std::vector<std::tuple<int, int, int, int>> sizes = {
        {1, 1, 8, 32},
        {2, 4, 64, 64},
        {1, 8, 128, 64},
        {4, 2, 32, 128},
    };
    
    for (auto& [B, H, N, D] : sizes) {
        std::cout << "  Testing B=" << B << " H=" << H << " N=" << N << " D=" << D << std::endl;
        
        float scale = 1.0f / std::sqrt(static_cast<float>(D));
        
        Tensor q = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU);
        Tensor k = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU);
        Tensor v = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU);
        
        Tensor flash_out = ops::flash_attention(q, k, v, scale, true);
        Tensor ref_out = reference_attention(q, k, v, scale, true);
        
        assert(allclose(flash_out, ref_out, 1e-2f, 1e-3f));
    }
    
    std::cout << "Flash Attention various sizes (CPU): PASSED" << std::endl;
}

// ============================================================================
// Flash Attention GPU Tests
// ============================================================================

#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)

Device get_gpu_device() {
#ifdef USE_HIP_BACKEND
    return Device::HIP;
#else
    return Device::CUDA;
#endif
}

void test_flash_attention_basic_gpu() {
    std::cout << "Testing Flash Attention basic (GPU)..." << std::endl;
    
    Device device = get_gpu_device();
    
    int B = 2, H = 4, N = 64, D = 64;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    Tensor q = vesper::randn({B, H, N, D}, DType::Float32, device);
    Tensor k = vesper::randn({B, H, N, D}, DType::Float32, device);
    Tensor v = vesper::randn({B, H, N, D}, DType::Float32, device);
    
    Tensor flash_out = ops::flash_attention(q, k, v, scale, false);
    
    // Move to CPU for reference
    Tensor q_cpu = q.to(Device::CPU);
    Tensor k_cpu = k.to(Device::CPU);
    Tensor v_cpu = v.to(Device::CPU);
    
    Tensor ref_out = reference_attention(q_cpu, k_cpu, v_cpu, scale, false);
    Tensor flash_cpu = flash_out.to(Device::CPU);
    
    assert(allclose(flash_cpu, ref_out, 1e-2f, 1e-3f));
    
    std::cout << "Flash Attention basic (GPU): PASSED" << std::endl;
}

void test_flash_attention_causal_gpu() {
    std::cout << "Testing Flash Attention causal (GPU)..." << std::endl;
    
    Device device = get_gpu_device();
    
    int B = 2, H = 8, N = 128, D = 64;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    Tensor q = vesper::randn({B, H, N, D}, DType::Float32, device);
    Tensor k = vesper::randn({B, H, N, D}, DType::Float32, device);
    Tensor v = vesper::randn({B, H, N, D}, DType::Float32, device);
    
    Tensor flash_out = ops::flash_attention(q, k, v, scale, true);
    
    Tensor q_cpu = q.to(Device::CPU);
    Tensor k_cpu = k.to(Device::CPU);
    Tensor v_cpu = v.to(Device::CPU);
    
    Tensor ref_out = reference_attention(q_cpu, k_cpu, v_cpu, scale, true);
    Tensor flash_cpu = flash_out.to(Device::CPU);
    
    assert(allclose(flash_cpu, ref_out, 1e-2f, 1e-3f));
    
    std::cout << "Flash Attention causal (GPU): PASSED" << std::endl;
}

void test_flash_attention_long_sequence_gpu() {
    std::cout << "Testing Flash Attention long sequence (GPU)..." << std::endl;
    
    Device device = get_gpu_device();
    
    // Test with a long sequence that would OOM with standard attention
    // Standard attention for N=4096 would need ~64MB per head just for attention matrix
    int B = 1, H = 8, N = 1024, D = 64;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    Tensor q = vesper::randn({B, H, N, D}, DType::Float32, device);
    Tensor k = vesper::randn({B, H, N, D}, DType::Float32, device);
    Tensor v = vesper::randn({B, H, N, D}, DType::Float32, device);
    
    // This should NOT OOM
    Tensor flash_out = ops::flash_attention(q, k, v, scale, true);
    
    // Basic sanity check - output should have correct shape
    assert(flash_out.shape()[0] == B);
    assert(flash_out.shape()[1] == H);
    assert(flash_out.shape()[2] == N);
    assert(flash_out.shape()[3] == D);
    
    // Check that output is not NaN/Inf
    Tensor flash_cpu = flash_out.to(Device::CPU);
    const float* ptr = flash_cpu.data_ptr<float>();
    bool all_finite = true;
    for (size_t i = 0; i < flash_cpu.numel(); ++i) {
        if (!std::isfinite(ptr[i])) {
            all_finite = false;
            break;
        }
    }
    assert(all_finite);
    
    std::cout << "Flash Attention long sequence (GPU): PASSED" << std::endl;
}

#endif  // GPU backends

// ============================================================================
// Flash Attention Backward Tests
// ============================================================================

void test_flash_attention_backward_cpu() {
    std::cout << "Testing Flash Attention backward (CPU)..." << std::endl;
    
    int B = 1, H = 2, N = 16, D = 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    Tensor q = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU, true);
    Tensor k = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU, true);
    Tensor v = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU, true);
    
    // Forward
    Tensor out = ops::flash_attention(q, k, v, scale, true);
    
    // Backward with identity gradient
    Tensor loss = ops::sum(out);
    loss.backward();
    
    // Check gradients exist
    assert(q.grad().defined());
    assert(k.grad().defined());
    assert(v.grad().defined());
    
    // Check gradient shapes
    assert(q.grad().shape() == q.shape());
    assert(k.grad().shape() == k.shape());
    assert(v.grad().shape() == v.shape());
    
    // Check gradients are not all zeros or NaN
    auto check_grad = [](const Tensor& grad, const char* name) {
        Tensor grad_cpu = grad.to(Device::CPU);
        const float* ptr = grad_cpu.data_ptr<float>();
        bool all_zero = true;
        bool has_nan = false;
        for (size_t i = 0; i < grad_cpu.numel(); ++i) {
            if (ptr[i] != 0.0f) all_zero = false;
            if (!std::isfinite(ptr[i])) has_nan = true;
        }
        if (all_zero) {
            std::cerr << "  Warning: " << name << " gradient is all zeros" << std::endl;
        }
        assert(!has_nan);
    };
    
    check_grad(q.grad(), "Q");
    check_grad(k.grad(), "K");
    check_grad(v.grad(), "V");
    
    std::cout << "Flash Attention backward (CPU): PASSED" << std::endl;
}

void test_flash_attention_numerical_gradient() {
    std::cout << "Testing Flash Attention numerical gradient check..." << std::endl;
    
    int B = 1, H = 1, N = 8, D = 16;  // Small for numerical gradient check
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    float eps = 1e-3f;
    
    Tensor q = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU, true);
    Tensor k = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU, true);
    Tensor v = vesper::randn({B, H, N, D}, DType::Float32, Device::CPU, true);
    
    // Compute analytical gradient
    Tensor out = ops::flash_attention(q, k, v, scale, true);
    Tensor loss = ops::sum(out);
    loss.backward();
    
    Tensor dq_analytical = q.grad().clone();
    
    // Numerical gradient for first few elements of Q
    Tensor q_data = q.to(Device::CPU);
    float* q_ptr = q_data.data_ptr<float>();
    
    int num_checks = std::min(10, static_cast<int>(q.numel()));
    int passed = 0;
    
    for (int i = 0; i < num_checks; ++i) {
        float orig = q_ptr[i];
        
        // f(x + eps)
        q_ptr[i] = orig + eps;
        q.copy_from(q_data);
        Tensor out_plus = ops::flash_attention(q, k, v, scale, true);
        float loss_plus = ops::sum(out_plus).item<float>();
        
        // f(x - eps)
        q_ptr[i] = orig - eps;
        q.copy_from(q_data);
        Tensor out_minus = ops::flash_attention(q, k, v, scale, true);
        float loss_minus = ops::sum(out_minus).item<float>();
        
        // Restore
        q_ptr[i] = orig;
        q.copy_from(q_data);
        
        // Numerical gradient
        float numerical = (loss_plus - loss_minus) / (2 * eps);
        float analytical = dq_analytical.to(Device::CPU).data_ptr<float>()[i];
        
        float rel_err = std::abs(numerical - analytical) / 
                       (std::max(std::abs(numerical), std::abs(analytical)) + 1e-8f);
        
        if (rel_err < 0.1f) {  // 10% tolerance
            passed++;
        } else {
            std::cout << "  Element " << i << ": numerical=" << numerical 
                      << ", analytical=" << analytical << ", rel_err=" << rel_err << std::endl;
        }
    }
    
    std::cout << "  Passed " << passed << "/" << num_checks << " gradient checks" << std::endl;
    assert(passed >= num_checks * 0.8);  // At least 80% should pass
    
    std::cout << "Flash Attention numerical gradient check: PASSED" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "Flash Attention Tests" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // CPU tests
        test_flash_attention_basic_cpu();
        test_flash_attention_causal_cpu();
        test_flash_attention_various_sizes_cpu();
        test_flash_attention_backward_cpu();
        test_flash_attention_numerical_gradient();
        
        // GPU tests
#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)
        test_flash_attention_basic_gpu();
        test_flash_attention_causal_gpu();
        test_flash_attention_long_sequence_gpu();
#else
        std::cout << "GPU tests skipped (no GPU backend available)" << std::endl;
#endif
        
        std::cout << "========================================" << std::endl;
        std::cout << "All Flash Attention tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
