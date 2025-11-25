/**
 * @file test_normalization_grad.cpp
 * @brief Comprehensive gradient tests for LayerNorm and RMSNorm
 * 
 * Tests:
 * - Numerical gradient verification using finite differences
 * - Backward pass correctness
 * - Gradient flow through learnable parameters (weight, bias)
 * - Edge cases: single element, large tensors, near-zero variance
 * - Multi-backend consistency for gradients
 */

#include <vesper/nn/normalization.h>
#include <vesper/core/factories.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/random.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <functional>

using namespace vesper;

// ============================================================================
// Test Utilities
// ============================================================================

void assert_tensors_close(const Tensor& t1, const Tensor& t2, float tol = 1e-4f, 
                          const std::string& msg = "") {
    assert(t1.numel() == t2.numel());
    std::vector<float> d1(t1.numel());
    std::vector<float> d2(t2.numel());
    auto c1 = t1.contiguous();
    auto c2 = t2.contiguous();
    c1.copy_to_host(d1.data());
    c2.copy_to_host(d2.data());
    for(size_t i = 0; i < t1.numel(); ++i) {
        float diff = std::abs(d1[i] - d2[i]);
        if (diff > tol) {
            std::cerr << msg << " Mismatch at " << i << ": " << d1[i] 
                      << " vs " << d2[i] << " (diff: " << diff << ")" << std::endl;
            assert(false);
        }
    }
}

float max_abs_diff(const Tensor& t1, const Tensor& t2) {
    std::vector<float> d1(t1.numel());
    std::vector<float> d2(t2.numel());
    auto c1 = t1.contiguous();
    auto c2 = t2.contiguous();
    c1.copy_to_host(d1.data());
    c2.copy_to_host(d2.data());
    float max_diff = 0.0f;
    for(size_t i = 0; i < t1.numel(); ++i) {
        max_diff = std::max(max_diff, std::abs(d1[i] - d2[i]));
    }
    return max_diff;
}

// Compute numerical gradient using finite differences
Tensor compute_numerical_gradient(
    std::function<float(Tensor&)> loss_fn, 
    Tensor& input, 
    float epsilon = 1e-3f) {
    
    std::vector<float> data(input.numel());
    input.copy_to_host(data.data());
    
    std::vector<float> grad(input.numel());
    
    for (size_t i = 0; i < input.numel(); ++i) {
        float orig = data[i];
        
        // f(x + eps)
        data[i] = orig + epsilon;
        input.copy_from_host(data.data());
        float loss_plus = loss_fn(input);
        
        // f(x - eps)
        data[i] = orig - epsilon;
        input.copy_from_host(data.data());
        float loss_minus = loss_fn(input);
        
        // Central difference
        grad[i] = (loss_plus - loss_minus) / (2.0f * epsilon);
        
        // Restore
        data[i] = orig;
    }
    
    input.copy_from_host(data.data());
    
    Tensor result = empty(input.shape(), input.dtype(), Device::CPU);
    result.copy_from_host(grad.data());
    return result;
}

// ============================================================================
// LayerNorm Gradient Tests
// ============================================================================

void test_layer_norm_backward_basic() {
    std::cout << "Testing LayerNorm backward (basic)..." << std::endl;
    
    auto ln = nn::LayerNorm({4});
    
    // Input: [2, 4] with requires_grad
    auto input = empty({2, 4}, DType::Float32, Device::CPU);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 
                                5.0f, 6.0f, 7.0f, 8.0f};
    input.copy_from_host(data.data());
    input.set_requires_grad(true);
    
    // Forward
    auto output = ln.forward(input);
    
    // Backward with sum loss
    auto loss = ops::sum(output);
    loss.backward();
    
    // Check that gradients exist
    assert(input.grad().defined());
    assert(ln.weight.grad().defined());
    assert(ln.bias.grad().defined());
    
    // Gradient shapes should match
    assert(input.grad().shape() == input.shape());
    assert(ln.weight.grad().shape() == ln.weight.shape());
    assert(ln.bias.grad().shape() == ln.bias.shape());
    
    std::cout << "LayerNorm backward (basic) passed!" << std::endl;
}

void test_layer_norm_numerical_gradient() {
    std::cout << "Testing LayerNorm numerical gradient..." << std::endl;
    
    auto ln = nn::LayerNorm({3});
    
    // Ensure weight is 1 and bias is 0 for simpler gradient check
    std::vector<float> w = {1.0f, 1.0f, 1.0f};
    std::vector<float> b = {0.0f, 0.0f, 0.0f};
    ln.weight.copy_from_host(w.data());
    ln.bias.copy_from_host(b.data());
    
    auto input = empty({2, 3}, DType::Float32, Device::CPU);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 
                                4.0f, 5.0f, 6.0f};
    input.copy_from_host(data.data());
    
    // Loss function
    auto loss_fn = [&](Tensor& x) -> float {
        auto out = ln.forward(x);
        auto loss = ops::sum(out);
        std::vector<float> val(1);
        loss.copy_to_host(val.data());
        return val[0];
    };
    
    // Numerical gradient
    Tensor num_grad = compute_numerical_gradient(loss_fn, input, 1e-4f);
    
    // Analytical gradient
    input.set_requires_grad(true);
    auto output = ln.forward(input);
    auto loss = ops::sum(output);
    loss.backward();
    
    Tensor ana_grad = input.grad();
    
    float max_diff = max_abs_diff(num_grad, ana_grad);
    std::cout << "  Max gradient diff: " << max_diff << std::endl;
    
    assert_tensors_close(num_grad, ana_grad, 1e-3f, "LayerNorm input gradient");
    
    std::cout << "LayerNorm numerical gradient passed!" << std::endl;
}

void test_layer_norm_weight_gradient() {
    std::cout << "Testing LayerNorm weight gradient..." << std::endl;
    
    auto ln = nn::LayerNorm({3});
    
    // Fixed input
    auto input = empty({2, 3}, DType::Float32, Device::CPU);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 
                                4.0f, 5.0f, 6.0f};
    input.copy_from_host(data.data());
    
    // Loss function for weight gradient
    auto loss_fn = [&](Tensor& weight) -> float {
        ln.weight = weight;
        auto out = ln.forward(input);
        auto loss = ops::sum(out);
        std::vector<float> val(1);
        loss.copy_to_host(val.data());
        return val[0];
    };
    
    // Numerical gradient
    Tensor num_grad = compute_numerical_gradient(loss_fn, ln.weight, 1e-4f);
    
    // Analytical gradient
    auto output = ln.forward(input);
    auto loss = ops::sum(output);
    loss.backward();
    
    Tensor ana_grad = ln.weight.grad();
    
    float max_diff = max_abs_diff(num_grad, ana_grad);
    std::cout << "  Max weight gradient diff: " << max_diff << std::endl;
    
    assert_tensors_close(num_grad, ana_grad, 1e-3f, "LayerNorm weight gradient");
    
    std::cout << "LayerNorm weight gradient passed!" << std::endl;
}

void test_layer_norm_edge_single_element() {
    std::cout << "Testing LayerNorm edge case: single element per normalization..." << std::endl;
    
    // normalized_shape = {1} means each sample has 1 element
    auto ln = nn::LayerNorm({1});
    
    auto input = empty({3, 1}, DType::Float32, Device::CPU);
    std::vector<float> data = {5.0f, -3.0f, 0.0f};
    input.copy_from_host(data.data());
    input.set_requires_grad(true);
    
    auto output = ln.forward(input);
    auto loss = ops::sum(output);
    loss.backward();
    
    // With single element, variance is 0, output should be 0 (or weight * 0 + bias)
    std::vector<float> out_data(3);
    output.copy_to_host(out_data.data());
    
    // Gradient should still flow
    assert(input.grad().defined());
    
    std::cout << "LayerNorm edge case (single element) passed!" << std::endl;
}

void test_layer_norm_edge_near_zero_variance() {
    std::cout << "Testing LayerNorm edge case: near-zero variance..." << std::endl;
    
    auto ln = nn::LayerNorm({4});
    
    // All elements nearly identical -> near-zero variance
    auto input = empty({1, 4}, DType::Float32, Device::CPU);
    std::vector<float> data = {1.0f, 1.0f + 1e-7f, 1.0f - 1e-7f, 1.0f};
    input.copy_from_host(data.data());
    input.set_requires_grad(true);
    
    auto output = ln.forward(input);
    auto loss = ops::sum(output);
    loss.backward();
    
    // Check no NaN/Inf in gradients
    std::vector<float> grad_data(4);
    input.grad().copy_to_host(grad_data.data());
    
    for (int i = 0; i < 4; ++i) {
        assert(!std::isnan(grad_data[i]) && "Gradient contains NaN");
        assert(!std::isinf(grad_data[i]) && "Gradient contains Inf");
    }
    
    std::cout << "LayerNorm edge case (near-zero variance) passed!" << std::endl;
}

// ============================================================================
// RMSNorm Gradient Tests
// ============================================================================

void test_rms_norm_backward_basic() {
    std::cout << "Testing RMSNorm backward (basic)..." << std::endl;
    
    auto rms = nn::RMSNorm({4});
    
    auto input = empty({2, 4}, DType::Float32, Device::CPU);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 
                                5.0f, 6.0f, 7.0f, 8.0f};
    input.copy_from_host(data.data());
    input.set_requires_grad(true);
    
    auto output = rms.forward(input);
    auto loss = ops::sum(output);
    loss.backward();
    
    assert(input.grad().defined());
    assert(rms.weight.grad().defined());
    
    std::cout << "RMSNorm backward (basic) passed!" << std::endl;
}

void test_rms_norm_numerical_gradient() {
    std::cout << "Testing RMSNorm numerical gradient..." << std::endl;
    
    auto rms = nn::RMSNorm({3});
    
    std::vector<float> w = {1.0f, 1.0f, 1.0f};
    rms.weight.copy_from_host(w.data());
    
    auto input = empty({2, 3}, DType::Float32, Device::CPU);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 
                                4.0f, 5.0f, 6.0f};
    input.copy_from_host(data.data());
    
    auto loss_fn = [&](Tensor& x) -> float {
        auto out = rms.forward(x);
        auto loss = ops::sum(out);
        std::vector<float> val(1);
        loss.copy_to_host(val.data());
        return val[0];
    };
    
    Tensor num_grad = compute_numerical_gradient(loss_fn, input, 1e-4f);
    
    input.set_requires_grad(true);
    auto output = rms.forward(input);
    auto loss = ops::sum(output);
    loss.backward();
    
    Tensor ana_grad = input.grad();
    
    float max_diff = max_abs_diff(num_grad, ana_grad);
    std::cout << "  Max gradient diff: " << max_diff << std::endl;
    
    assert_tensors_close(num_grad, ana_grad, 1e-3f, "RMSNorm input gradient");
    
    std::cout << "RMSNorm numerical gradient passed!" << std::endl;
}

void test_rms_norm_edge_large_values() {
    std::cout << "Testing RMSNorm edge case: large values..." << std::endl;
    
    auto rms = nn::RMSNorm({4});
    
    // Large values that could cause overflow without proper handling
    auto input = empty({1, 4}, DType::Float32, Device::CPU);
    std::vector<float> data = {1e10f, 2e10f, -1e10f, 3e10f};
    input.copy_from_host(data.data());
    input.set_requires_grad(true);
    
    auto output = rms.forward(input);
    auto loss = ops::sum(output);
    loss.backward();
    
    std::vector<float> grad_data(4);
    input.grad().copy_to_host(grad_data.data());
    
    for (int i = 0; i < 4; ++i) {
        assert(!std::isnan(grad_data[i]) && "Gradient contains NaN");
        assert(!std::isinf(grad_data[i]) && "Gradient contains Inf");
    }
    
    std::cout << "RMSNorm edge case (large values) passed!" << std::endl;
}

// ============================================================================
// Multi-Backend Gradient Consistency Tests
// ============================================================================

void test_layer_norm_gradient_consistency() {
    std::cout << "Testing LayerNorm gradient consistency across backends..." << std::endl;
    
    auto input_cpu = empty({4, 8}, DType::Float32, Device::CPU);
    ops::uniform_(input_cpu, -1.0f, 1.0f);
    
    auto ln_cpu = nn::LayerNorm({8});
    
    // CPU gradient
    auto input_cpu_grad = input_cpu.clone();
    input_cpu_grad.set_requires_grad(true);
    auto output_cpu = ln_cpu.forward(input_cpu_grad);
    auto loss_cpu = ops::sum(output_cpu);
    loss_cpu.backward();
    Tensor grad_cpu = input_cpu_grad.grad();

#ifdef USE_CUDA_BACKEND
    {
        auto input_cuda = input_cpu.to(Device::CUDA);
        input_cuda.set_requires_grad(true);
        auto ln_cuda = nn::LayerNorm({8});
        ln_cuda.weight = ln_cpu.weight.to(Device::CUDA);
        ln_cuda.bias = ln_cpu.bias.to(Device::CUDA);
        
        auto output_cuda = ln_cuda.forward(input_cuda);
        auto loss_cuda = ops::sum(output_cuda);
        loss_cuda.backward();
        
        assert_tensors_close(grad_cpu, input_cuda.grad().contiguous().to(Device::CPU), 1e-4f,
                            "LayerNorm grad CPU vs CUDA");
        std::cout << "  LayerNorm gradient CPU vs CUDA passed!" << std::endl;
    }
#endif

#ifdef USE_HIP_BACKEND
    {
        auto input_hip = input_cpu.to(Device::HIP);
        input_hip.set_requires_grad(true);
        auto ln_hip = nn::LayerNorm({8});
        ln_hip.weight = ln_cpu.weight.to(Device::HIP);
        ln_hip.bias = ln_cpu.bias.to(Device::HIP);
        
        auto output_hip = ln_hip.forward(input_hip);
        auto loss_hip = ops::sum(output_hip);
        loss_hip.backward();
        
        assert_tensors_close(grad_cpu, input_hip.grad().contiguous().to(Device::CPU), 1e-4f,
                            "LayerNorm grad CPU vs HIP");
        std::cout << "  LayerNorm gradient CPU vs HIP passed!" << std::endl;
    }
#endif
}

void test_rms_norm_gradient_consistency() {
    std::cout << "Testing RMSNorm gradient consistency across backends..." << std::endl;
    
    auto input_cpu = empty({4, 8}, DType::Float32, Device::CPU);
    ops::uniform_(input_cpu, -1.0f, 1.0f);
    
    auto rms_cpu = nn::RMSNorm({8});
    
    // CPU gradient
    auto input_cpu_grad = input_cpu.clone();
    input_cpu_grad.set_requires_grad(true);
    auto output_cpu = rms_cpu.forward(input_cpu_grad);
    auto loss_cpu = ops::sum(output_cpu);
    loss_cpu.backward();
    Tensor grad_cpu = input_cpu_grad.grad();

#ifdef USE_CUDA_BACKEND
    {
        auto input_cuda = input_cpu.to(Device::CUDA);
        input_cuda.set_requires_grad(true);
        auto rms_cuda = nn::RMSNorm({8});
        rms_cuda.weight = rms_cpu.weight.to(Device::CUDA);
        
        auto output_cuda = rms_cuda.forward(input_cuda);
        auto loss_cuda = ops::sum(output_cuda);
        loss_cuda.backward();
        
        assert_tensors_close(grad_cpu, input_cuda.grad().contiguous().to(Device::CPU), 1e-4f,
                            "RMSNorm grad CPU vs CUDA");
        std::cout << "  RMSNorm gradient CPU vs CUDA passed!" << std::endl;
    }
#endif

#ifdef USE_HIP_BACKEND
    {
        auto input_hip = input_cpu.to(Device::HIP);
        input_hip.set_requires_grad(true);
        auto rms_hip = nn::RMSNorm({8});
        rms_hip.weight = rms_cpu.weight.to(Device::HIP);
        
        auto output_hip = rms_hip.forward(input_hip);
        auto loss_hip = ops::sum(output_hip);
        loss_hip.backward();
        
        assert_tensors_close(grad_cpu, input_hip.grad().contiguous().to(Device::CPU), 1e-4f,
                            "RMSNorm grad CPU vs HIP");
        std::cout << "  RMSNorm gradient CPU vs HIP passed!" << std::endl;
    }
#endif
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Normalization Gradient Tests ===" << std::endl;
    
    // LayerNorm tests
    test_layer_norm_backward_basic();
    test_layer_norm_numerical_gradient();
    test_layer_norm_weight_gradient();
    test_layer_norm_edge_single_element();
    test_layer_norm_edge_near_zero_variance();
    
    // RMSNorm tests
    test_rms_norm_backward_basic();
    test_rms_norm_numerical_gradient();
    test_rms_norm_edge_large_values();
    
    // Consistency tests
    test_layer_norm_gradient_consistency();
    test_rms_norm_gradient_consistency();
    
    std::cout << "\n=== All Normalization Gradient Tests Passed! ===" << std::endl;
    return 0;
}
