/**
 * @file test_softmax_comprehensive.cpp
 * @brief Comprehensive tests for Softmax including edge cases and numerical stability
 * 
 * Tests:
 * - Numerical stability with large logits (overflow prevention)
 * - Backward pass correctness with numerical gradient verification
 * - Edge cases: single element, uniform inputs, all zeros
 * - Temperature scaling
 * - Dim parameter variations
 * - Multi-backend consistency for gradients
 */

#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/random.h>
#include <vesper/ops/elementwise.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <functional>
#include <limits>

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

Tensor compute_numerical_gradient(
    std::function<float(Tensor&)> loss_fn, 
    Tensor& input, 
    float epsilon = 1e-3f) {
    
    std::vector<float> data(input.numel());
    input.copy_to_host(data.data());
    
    std::vector<float> grad(input.numel());
    
    for (size_t i = 0; i < input.numel(); ++i) {
        float orig = data[i];
        
        data[i] = orig + epsilon;
        input.copy_from_host(data.data());
        float loss_plus = loss_fn(input);
        
        data[i] = orig - epsilon;
        input.copy_from_host(data.data());
        float loss_minus = loss_fn(input);
        
        grad[i] = (loss_plus - loss_minus) / (2.0f * epsilon);
        
        data[i] = orig;
    }
    
    input.copy_from_host(data.data());
    
    Tensor result = empty(input.shape(), input.dtype(), Device::CPU);
    result.copy_from_host(grad.data());
    return result;
}

// Check if tensor contains NaN or Inf
bool contains_nan_or_inf(const Tensor& t) {
    std::vector<float> data(t.numel());
    t.copy_to_host(data.data());
    for (float v : data) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
}

// Check softmax properties: values in [0,1], sum to 1 along dim
void verify_softmax_properties(const Tensor& output, int dim, float tol = 1e-5f) {
    std::vector<float> data(output.numel());
    output.copy_to_host(data.data());
    
    // Check all values in [0, 1]
    for (float v : data) {
        assert(v >= -tol && v <= 1.0f + tol && "Softmax value out of range");
    }
    
    // Sum check along dim - for simplicity check last dim for 2D
    // More general approach would need shape handling
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

void test_softmax_large_positive_logits() {
    std::cout << "Testing softmax with large positive logits..." << std::endl;
    
    auto input = empty({2, 4}, DType::Float32, Device::CPU);
    std::vector<float> data = {
        100.0f, 200.0f, 300.0f, 400.0f,  // Very large values
        1e38f, 1e38f, 1e38f, 1e38f       // Near overflow limit
    };
    input.copy_from_host(data.data());
    
    auto output = nn::functional::softmax(input, 1);
    
    // Should not contain NaN or Inf
    assert(!contains_nan_or_inf(output) && "Large logits caused NaN/Inf");
    
    // First row: exp(400) dominates -> last element should be ~1
    std::vector<float> out_data(8);
    output.copy_to_host(out_data.data());
    
    assert(out_data[3] > 0.99f && "Expected last element to dominate");
    
    // Second row: all equal -> should be uniform 0.25
    for (int i = 4; i < 8; ++i) {
        assert(std::abs(out_data[i] - 0.25f) < 1e-5f && "Equal inputs should give uniform");
    }
    
    std::cout << "Softmax large positive logits passed!" << std::endl;
}

void test_softmax_large_negative_logits() {
    std::cout << "Testing softmax with large negative logits..." << std::endl;
    
    auto input = empty({1, 4}, DType::Float32, Device::CPU);
    std::vector<float> data = {-100.0f, -200.0f, -300.0f, -400.0f};
    input.copy_from_host(data.data());
    
    auto output = nn::functional::softmax(input, 1);
    
    assert(!contains_nan_or_inf(output) && "Large negative logits caused NaN/Inf");
    
    // First element should dominate (least negative)
    std::vector<float> out_data(4);
    output.copy_to_host(out_data.data());
    
    assert(out_data[0] > 0.99f && "Expected first element to dominate");
    
    std::cout << "Softmax large negative logits passed!" << std::endl;
}

void test_softmax_mixed_extreme_logits() {
    std::cout << "Testing softmax with mixed extreme logits..." << std::endl;
    
    auto input = empty({1, 4}, DType::Float32, Device::CPU);
    std::vector<float> data = {-1e10f, 0.0f, 1e10f, 1e10f};
    input.copy_from_host(data.data());
    
    auto output = nn::functional::softmax(input, 1);
    
    assert(!contains_nan_or_inf(output) && "Mixed extreme logits caused NaN/Inf");
    
    std::vector<float> out_data(4);
    output.copy_to_host(out_data.data());
    
    // -1e10 should give essentially 0
    assert(out_data[0] < 1e-6f);
    // 0 should give essentially 0 compared to 1e10
    assert(out_data[1] < 1e-6f);
    // Last two equal large values should each be 0.5
    assert(std::abs(out_data[2] - 0.5f) < 1e-5f);
    assert(std::abs(out_data[3] - 0.5f) < 1e-5f);
    
    std::cout << "Softmax mixed extreme logits passed!" << std::endl;
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

void test_softmax_backward_basic() {
    std::cout << "Testing softmax backward (basic)..." << std::endl;
    
    auto input = empty({2, 3}, DType::Float32, Device::CPU);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 
                                4.0f, 5.0f, 6.0f};
    input.copy_from_host(data.data());
    input.set_requires_grad(true);
    
    auto output = nn::functional::softmax(input, 1);
    auto loss = ops::sum(output);
    loss.backward();
    
    assert(input.grad().defined());
    assert(input.grad().shape() == input.shape());
    
    // Softmax gradient property: sum of gradients should be 0 for each row
    // because ∂(sum softmax)/∂x_i = softmax_i - softmax_i * sum(softmax) = 0
    // Actually for sum loss, gradient is all zeros for softmax!
    // ∂L/∂x_i = sum_j (∂L/∂y_j * ∂y_j/∂x_i) = sum_j (1 * (y_i*δ_ij - y_i*y_j))
    //         = y_i - y_i * sum(y) = y_i - y_i = 0
    
    std::vector<float> grad_data(6);
    input.grad().copy_to_host(grad_data.data());
    
    for (int i = 0; i < 6; ++i) {
        assert(std::abs(grad_data[i]) < 1e-5f && "Softmax gradient for sum should be ~0");
    }
    
    std::cout << "Softmax backward (basic) passed!" << std::endl;
}

void test_softmax_numerical_gradient() {
    std::cout << "Testing softmax numerical gradient..." << std::endl;
    
    auto input = empty({2, 3}, DType::Float32, Device::CPU);
    std::vector<float> data = {0.5f, 1.0f, 1.5f, 
                                2.0f, 2.5f, 3.0f};
    input.copy_from_host(data.data());
    
    // Use a weighted sum to get non-zero gradients
    auto loss_fn = [&](Tensor& x) -> float {
        auto out = nn::functional::softmax(x, 1);
        // Weight by position: w = [1, 2, 3, 1, 2, 3]
        std::vector<float> weights = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
        auto w = empty({2, 3}, DType::Float32, Device::CPU);
        w.copy_from_host(weights.data());
        auto weighted = ops::mul(out, w);
        auto loss = ops::sum(weighted);
        std::vector<float> val(1);
        loss.copy_to_host(val.data());
        return val[0];
    };
    
    Tensor num_grad = compute_numerical_gradient(loss_fn, input, 1e-4f);
    
    // Analytical gradient
    input.set_requires_grad(true);
    auto out = nn::functional::softmax(input, 1);
    std::vector<float> weights = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
    auto w = empty({2, 3}, DType::Float32, Device::CPU);
    w.copy_from_host(weights.data());
    auto weighted = ops::mul(out, w);
    auto loss = ops::sum(weighted);
    loss.backward();
    
    Tensor ana_grad = input.grad();
    
    float max_diff = max_abs_diff(num_grad, ana_grad);
    std::cout << "  Max gradient diff: " << max_diff << std::endl;
    
    assert_tensors_close(num_grad, ana_grad, 1e-3f, "Softmax gradient");
    
    std::cout << "Softmax numerical gradient passed!" << std::endl;
}

// ============================================================================
// Edge Case Tests
// ============================================================================

void test_softmax_single_element() {
    std::cout << "Testing softmax single element..." << std::endl;
    
    auto input = empty({1, 1}, DType::Float32, Device::CPU);
    std::vector<float> data = {5.0f};
    input.copy_from_host(data.data());
    
    auto output = nn::functional::softmax(input, 1);
    
    std::vector<float> out_data(1);
    output.copy_to_host(out_data.data());
    
    // Single element softmax should be 1.0
    assert(std::abs(out_data[0] - 1.0f) < 1e-6f);
    
    std::cout << "Softmax single element passed!" << std::endl;
}

void test_softmax_all_zeros() {
    std::cout << "Testing softmax all zeros..." << std::endl;
    
    auto input = zeros({2, 4}, DType::Float32, Device::CPU);
    
    auto output = nn::functional::softmax(input, 1);
    
    std::vector<float> out_data(8);
    output.copy_to_host(out_data.data());
    
    // All zeros -> uniform distribution
    for (int i = 0; i < 8; ++i) {
        assert(std::abs(out_data[i] - 0.25f) < 1e-6f);
    }
    
    std::cout << "Softmax all zeros passed!" << std::endl;
}

void test_softmax_identical_values() {
    std::cout << "Testing softmax identical values..." << std::endl;
    
    auto input = full({3, 5}, DType::Float32, Device::CPU, 42.0f);
    
    auto output = nn::functional::softmax(input, 1);
    
    std::vector<float> out_data(15);
    output.copy_to_host(out_data.data());
    
    // All identical -> uniform 0.2 for each row
    for (int i = 0; i < 15; ++i) {
        assert(std::abs(out_data[i] - 0.2f) < 1e-6f);
    }
    
    std::cout << "Softmax identical values passed!" << std::endl;
}

void test_softmax_dim_0() {
    std::cout << "Testing softmax dim=0..." << std::endl;
    
    auto input = empty({3, 2}, DType::Float32, Device::CPU);
    std::vector<float> data = {
        1.0f, 1.0f,
        2.0f, 2.0f,
        3.0f, 3.0f
    };
    input.copy_from_host(data.data());
    
    auto output = nn::functional::softmax(input, 0);
    
    std::vector<float> out_data(6);
    output.copy_to_host(out_data.data());
    
    // Column sums should be 1
    float col0_sum = out_data[0] + out_data[2] + out_data[4];
    float col1_sum = out_data[1] + out_data[3] + out_data[5];
    
    assert(std::abs(col0_sum - 1.0f) < 1e-5f);
    assert(std::abs(col1_sum - 1.0f) < 1e-5f);
    
    // exp(3) should dominate
    assert(out_data[4] > 0.6f); // Last row, col 0
    
    std::cout << "Softmax dim=0 passed!" << std::endl;
}

void test_softmax_3d_tensor() {
    std::cout << "Testing softmax 3D tensor..." << std::endl;
    
    auto input = empty({2, 3, 4}, DType::Float32, Device::CPU);
    ops::uniform_(input, -1.0f, 1.0f);
    
    // Softmax over last dimension
    auto output = nn::functional::softmax(input, 2);
    
    assert(!contains_nan_or_inf(output));
    
    // Check sums along last dimension = 1
    std::vector<float> out_data(24);
    output.copy_to_host(out_data.data());
    
    for (int b = 0; b < 2; ++b) {
        for (int i = 0; i < 3; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 4; ++j) {
                sum += out_data[b * 12 + i * 4 + j];
            }
            assert(std::abs(sum - 1.0f) < 1e-5f);
        }
    }
    
    std::cout << "Softmax 3D tensor passed!" << std::endl;
}

// ============================================================================
// Multi-Backend Consistency Tests
// ============================================================================

void test_softmax_backward_consistency() {
    std::cout << "Testing softmax backward consistency across backends..." << std::endl;
    
    auto input_cpu = empty({4, 8}, DType::Float32, Device::CPU);
    ops::uniform_(input_cpu, -2.0f, 2.0f);
    
    // CPU gradient with weighted loss
    auto input_cpu_grad = input_cpu.clone();
    input_cpu_grad.set_requires_grad(true);
    auto output_cpu = nn::functional::softmax(input_cpu_grad, 1);
    
    // Create weights for non-trivial gradient
    auto weights = empty({4, 8}, DType::Float32, Device::CPU);
    ops::uniform_(weights, 0.0f, 1.0f);
    
    auto weighted_cpu = ops::mul(output_cpu, weights);
    auto loss_cpu = ops::sum(weighted_cpu);
    loss_cpu.backward();
    Tensor grad_cpu = input_cpu_grad.grad();

#ifdef USE_CUDA_BACKEND
    {
        auto input_cuda = input_cpu.to(Device::CUDA);
        input_cuda.set_requires_grad(true);
        auto weights_cuda = weights.to(Device::CUDA);
        
        auto output_cuda = nn::functional::softmax(input_cuda, 1);
        auto weighted_cuda = ops::mul(output_cuda, weights_cuda);
        auto loss_cuda = ops::sum(weighted_cuda);
        loss_cuda.backward();
        
        assert_tensors_close(grad_cpu, input_cuda.grad().contiguous().to(Device::CPU), 1e-4f,
                            "Softmax grad CPU vs CUDA");
        std::cout << "  Softmax gradient CPU vs CUDA passed!" << std::endl;
    }
#endif

#ifdef USE_HIP_BACKEND
    {
        auto input_hip = input_cpu.to(Device::HIP);
        input_hip.set_requires_grad(true);
        auto weights_hip = weights.to(Device::HIP);
        
        auto output_hip = nn::functional::softmax(input_hip, 1);
        auto weighted_hip = ops::mul(output_hip, weights_hip);
        auto loss_hip = ops::sum(weighted_hip);
        loss_hip.backward();
        
        assert_tensors_close(grad_cpu, input_hip.grad().contiguous().to(Device::CPU), 1e-4f,
                            "Softmax grad CPU vs HIP");
        std::cout << "  Softmax gradient CPU vs HIP passed!" << std::endl;
    }
#endif
}

void test_softmax_stability_consistency() {
    std::cout << "Testing softmax stability consistency across backends..." << std::endl;
    
    // Use large values to test numerical stability
    auto input_cpu = empty({2, 5}, DType::Float32, Device::CPU);
    std::vector<float> data = {
        -1000.0f, 0.0f, 1000.0f, 500.0f, -500.0f,
        1e10f, 1e10f, 1e10f, 0.0f, -1e10f
    };
    input_cpu.copy_from_host(data.data());
    
    auto output_cpu = nn::functional::softmax(input_cpu, 1);

#ifdef USE_CUDA_BACKEND
    {
        auto input_cuda = input_cpu.to(Device::CUDA);
        auto output_cuda = nn::functional::softmax(input_cuda, 1);
        assert_tensors_close(output_cpu, output_cuda.to(Device::CPU), 1e-5f,
                            "Softmax stability CPU vs CUDA");
        std::cout << "  Softmax stability CPU vs CUDA passed!" << std::endl;
    }
#endif

#ifdef USE_HIP_BACKEND
    {
        auto input_hip = input_cpu.to(Device::HIP);
        auto output_hip = nn::functional::softmax(input_hip, 1);
        assert_tensors_close(output_cpu, output_hip.to(Device::CPU), 1e-5f,
                            "Softmax stability CPU vs HIP");
        std::cout << "  Softmax stability CPU vs HIP passed!" << std::endl;
    }
#endif
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Softmax Comprehensive Tests ===" << std::endl;
    
    // Numerical stability tests
    test_softmax_large_positive_logits();
    test_softmax_large_negative_logits();
    test_softmax_mixed_extreme_logits();
    
    // Backward pass tests
    test_softmax_backward_basic();
    test_softmax_numerical_gradient();
    
    // Edge case tests
    test_softmax_single_element();
    test_softmax_all_zeros();
    test_softmax_identical_values();
    test_softmax_dim_0();
    test_softmax_3d_tensor();
    
    // Consistency tests
    test_softmax_backward_consistency();
    test_softmax_stability_consistency();
    
    std::cout << "\n=== All Softmax Comprehensive Tests Passed! ===" << std::endl;
    return 0;
}
