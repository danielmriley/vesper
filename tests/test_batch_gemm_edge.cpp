/**
 * @file test_batch_gemm_edge.cpp
 * @brief Edge case tests for Batch GEMM operations
 * 
 * Tests:
 * - Batch size 1 (degenerate case)
 * - Single element matrices
 * - Very tall/wide matrices
 * - Mixed batch dimensions (broadcasting)
 * - Numerical gradient verification
 * - Multi-backend consistency for edge cases
 */

#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/random.h>
#include <vesper/ops/elementwise.h>
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

bool contains_nan_or_inf(const Tensor& t) {
    std::vector<float> data(t.numel());
    auto c = t.contiguous();
    c.copy_to_host(data.data());
    for (float v : data) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
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

float max_abs_diff(const Tensor& t1, const Tensor& t2) {
    std::vector<float> d1(t1.numel());
    std::vector<float> d2(t2.numel());
    t1.copy_to_host(d1.data());
    t2.copy_to_host(d2.data());
    float max_diff = 0.0f;
    for(size_t i = 0; i < t1.numel(); ++i) {
        max_diff = std::max(max_diff, std::abs(d1[i] - d2[i]));
    }
    return max_diff;
}

// ============================================================================
// Edge Case Tests
// ============================================================================

void test_batch_gemm_single_batch() {
    std::cout << "Testing batch GEMM with batch size 1..." << std::endl;
    
    Tensor A = empty({1, 4, 3}, DType::Float32, Device::CPU);
    Tensor B = empty({1, 3, 5}, DType::Float32, Device::CPU);
    
    ops::normal_(A, 0.0f, 1.0f);
    ops::normal_(B, 0.0f, 1.0f);
    
    Tensor C = ops::matmul(A, B);
    
    assert(C.ndim() == 3);
    assert(C.shape()[0] == 1);
    assert(C.shape()[1] == 4);
    assert(C.shape()[2] == 5);
    assert(!contains_nan_or_inf(C));
    
    std::cout << "Batch GEMM single batch passed!" << std::endl;
}

void test_batch_gemm_single_element_matrices() {
    std::cout << "Testing batch GEMM with 1x1 matrices..." << std::endl;
    
    Tensor A = empty({4, 1, 1}, DType::Float32, Device::CPU);
    Tensor B = empty({4, 1, 1}, DType::Float32, Device::CPU);
    
    std::vector<float> a_data = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> b_data = {3.0f, 4.0f, 5.0f, 6.0f};
    A.copy_from_host(a_data.data());
    B.copy_from_host(b_data.data());
    
    Tensor C = ops::matmul(A, B);
    
    std::vector<float> c_data(4);
    C.copy_to_host(c_data.data());
    
    // Expected: [6, 12, 20, 30]
    assert(std::abs(c_data[0] - 6.0f) < 1e-5f);
    assert(std::abs(c_data[1] - 12.0f) < 1e-5f);
    assert(std::abs(c_data[2] - 20.0f) < 1e-5f);
    assert(std::abs(c_data[3] - 30.0f) < 1e-5f);
    
    std::cout << "Batch GEMM single element matrices passed!" << std::endl;
}

void test_batch_gemm_tall_matrices() {
    std::cout << "Testing batch GEMM with tall matrices..." << std::endl;
    
    // Tall: many rows, few columns
    Tensor A = empty({2, 128, 4}, DType::Float32, Device::CPU);
    Tensor B = empty({2, 4, 8}, DType::Float32, Device::CPU);
    
    ops::normal_(A, 0.0f, 0.1f);
    ops::normal_(B, 0.0f, 0.1f);
    
    Tensor C = ops::matmul(A, B);
    
    assert(C.shape()[0] == 2);
    assert(C.shape()[1] == 128);
    assert(C.shape()[2] == 8);
    assert(!contains_nan_or_inf(C));
    
    std::cout << "Batch GEMM tall matrices passed!" << std::endl;
}

void test_batch_gemm_wide_matrices() {
    std::cout << "Testing batch GEMM with wide matrices..." << std::endl;
    
    // Wide: few rows, many columns
    Tensor A = empty({2, 4, 128}, DType::Float32, Device::CPU);
    Tensor B = empty({2, 128, 8}, DType::Float32, Device::CPU);
    
    ops::normal_(A, 0.0f, 0.1f);
    ops::normal_(B, 0.0f, 0.1f);
    
    Tensor C = ops::matmul(A, B);
    
    assert(C.shape()[0] == 2);
    assert(C.shape()[1] == 4);
    assert(C.shape()[2] == 8);
    assert(!contains_nan_or_inf(C));
    
    std::cout << "Batch GEMM wide matrices passed!" << std::endl;
}

void test_batch_gemm_square_matrices() {
    std::cout << "Testing batch GEMM with square matrices..." << std::endl;
    
    Tensor A = empty({3, 32, 32}, DType::Float32, Device::CPU);
    Tensor B = empty({3, 32, 32}, DType::Float32, Device::CPU);
    
    ops::normal_(A, 0.0f, 0.1f);
    ops::normal_(B, 0.0f, 0.1f);
    
    Tensor C = ops::matmul(A, B);
    
    assert(C.shape()[0] == 3);
    assert(C.shape()[1] == 32);
    assert(C.shape()[2] == 32);
    assert(!contains_nan_or_inf(C));
    
    std::cout << "Batch GEMM square matrices passed!" << std::endl;
}

void test_batch_gemm_vector_matrix() {
    std::cout << "Testing batch GEMM vector-matrix multiplication..." << std::endl;
    
    // Row vector times matrix: [B, 1, K] x [B, K, N] -> [B, 1, N]
    Tensor A = empty({4, 1, 8}, DType::Float32, Device::CPU);
    Tensor B = empty({4, 8, 16}, DType::Float32, Device::CPU);
    
    ops::normal_(A, 0.0f, 1.0f);
    ops::normal_(B, 0.0f, 1.0f);
    
    Tensor C = ops::matmul(A, B);
    
    assert(C.shape()[0] == 4);
    assert(C.shape()[1] == 1);
    assert(C.shape()[2] == 16);
    assert(!contains_nan_or_inf(C));
    
    std::cout << "Batch GEMM vector-matrix passed!" << std::endl;
}

void test_batch_gemm_matrix_vector() {
    std::cout << "Testing batch GEMM matrix-vector multiplication..." << std::endl;
    
    // Matrix times column vector: [B, M, K] x [B, K, 1] -> [B, M, 1]
    Tensor A = empty({4, 16, 8}, DType::Float32, Device::CPU);
    Tensor B = empty({4, 8, 1}, DType::Float32, Device::CPU);
    
    ops::normal_(A, 0.0f, 1.0f);
    ops::normal_(B, 0.0f, 1.0f);
    
    Tensor C = ops::matmul(A, B);
    
    assert(C.shape()[0] == 4);
    assert(C.shape()[1] == 16);
    assert(C.shape()[2] == 1);
    assert(!contains_nan_or_inf(C));
    
    std::cout << "Batch GEMM matrix-vector passed!" << std::endl;
}

// ============================================================================
// 4D Batch GEMM Tests
// ============================================================================

void test_batch_gemm_4d_basic() {
    std::cout << "Testing 4D batch GEMM..." << std::endl;
    
    // [B, H, S, D] x [B, H, D, S] for attention-like operation
    Tensor A = empty({2, 4, 8, 16}, DType::Float32, Device::CPU);
    Tensor B = empty({2, 4, 16, 8}, DType::Float32, Device::CPU);
    
    ops::normal_(A, 0.0f, 0.1f);
    ops::normal_(B, 0.0f, 0.1f);
    
    Tensor C = ops::matmul(A, B);
    
    assert(C.ndim() == 4);
    assert(C.shape()[0] == 2);
    assert(C.shape()[1] == 4);
    assert(C.shape()[2] == 8);
    assert(C.shape()[3] == 8);
    assert(!contains_nan_or_inf(C));
    
    std::cout << "4D batch GEMM passed!" << std::endl;
}

void test_batch_gemm_4d_single_batch_head() {
    std::cout << "Testing 4D batch GEMM with batch=1, head=1..." << std::endl;
    
    Tensor A = empty({1, 1, 4, 8}, DType::Float32, Device::CPU);
    Tensor B = empty({1, 1, 8, 4}, DType::Float32, Device::CPU);
    
    ops::normal_(A, 0.0f, 1.0f);
    ops::normal_(B, 0.0f, 1.0f);
    
    Tensor C = ops::matmul(A, B);
    
    assert(C.shape()[0] == 1);
    assert(C.shape()[1] == 1);
    assert(C.shape()[2] == 4);
    assert(C.shape()[3] == 4);
    assert(!contains_nan_or_inf(C));
    
    std::cout << "4D batch GEMM single batch/head passed!" << std::endl;
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

void test_batch_gemm_backward_basic() {
    std::cout << "Testing batch GEMM backward (basic)..." << std::endl;
    
    Tensor A = empty({2, 4, 3}, DType::Float32, Device::CPU);
    Tensor B = empty({2, 3, 5}, DType::Float32, Device::CPU);
    
    ops::normal_(A, 0.0f, 0.5f);
    ops::normal_(B, 0.0f, 0.5f);
    
    A.set_requires_grad(true);
    B.set_requires_grad(true);
    
    Tensor C = ops::matmul(A, B);
    Tensor loss = ops::sum(C);
    loss.backward();
    
    assert(A.grad().defined());
    assert(B.grad().defined());
    assert(A.grad().shape() == A.shape());
    assert(B.grad().shape() == B.shape());
    assert(!contains_nan_or_inf(A.grad()));
    assert(!contains_nan_or_inf(B.grad()));
    
    std::cout << "Batch GEMM backward (basic) passed!" << std::endl;
}

void test_batch_gemm_numerical_gradient_A() {
    std::cout << "Testing batch GEMM numerical gradient for A..." << std::endl;
    
    Tensor A = empty({2, 3, 4}, DType::Float32, Device::CPU);
    Tensor B = empty({2, 4, 2}, DType::Float32, Device::CPU);
    
    ops::uniform_(A, -0.5f, 0.5f);
    ops::uniform_(B, -0.5f, 0.5f);
    
    auto loss_fn = [&](Tensor& a) -> float {
        auto c = ops::matmul(a, B);
        auto loss = ops::sum(c);
        std::vector<float> val(1);
        loss.copy_to_host(val.data());
        return val[0];
    };
    
    Tensor num_grad = compute_numerical_gradient(loss_fn, A, 1e-4f);
    
    A.set_requires_grad(true);
    auto c = ops::matmul(A, B);
    auto loss = ops::sum(c);
    loss.backward();
    
    Tensor ana_grad = A.grad();
    
    float max_diff = max_abs_diff(num_grad, ana_grad);
    std::cout << "  A gradient max diff: " << max_diff << std::endl;
    
    assert_tensors_close(num_grad, ana_grad, 1e-3f, "Batch GEMM A gradient");
    
    std::cout << "Batch GEMM numerical gradient A passed!" << std::endl;
}

void test_batch_gemm_numerical_gradient_B() {
    std::cout << "Testing batch GEMM numerical gradient for B..." << std::endl;
    
    Tensor A = empty({2, 3, 4}, DType::Float32, Device::CPU);
    Tensor B = empty({2, 4, 2}, DType::Float32, Device::CPU);
    
    ops::uniform_(A, -0.5f, 0.5f);
    ops::uniform_(B, -0.5f, 0.5f);
    
    auto loss_fn = [&](Tensor& b) -> float {
        auto c = ops::matmul(A, b);
        auto loss = ops::sum(c);
        std::vector<float> val(1);
        loss.copy_to_host(val.data());
        return val[0];
    };
    
    Tensor num_grad = compute_numerical_gradient(loss_fn, B, 1e-4f);
    
    B.set_requires_grad(true);
    auto c = ops::matmul(A, B);
    auto loss = ops::sum(c);
    loss.backward();
    
    Tensor ana_grad = B.grad();
    
    float max_diff = max_abs_diff(num_grad, ana_grad);
    std::cout << "  B gradient max diff: " << max_diff << std::endl;
    
    assert_tensors_close(num_grad, ana_grad, 1e-3f, "Batch GEMM B gradient");
    
    std::cout << "Batch GEMM numerical gradient B passed!" << std::endl;
}

// ============================================================================
// Multi-Backend Consistency
// ============================================================================

void test_batch_gemm_consistency() {
    std::cout << "Testing batch GEMM consistency across backends..." << std::endl;
    
    Tensor A_cpu = empty({4, 8, 16}, DType::Float32, Device::CPU);
    Tensor B_cpu = empty({4, 16, 8}, DType::Float32, Device::CPU);
    
    ops::uniform_(A_cpu, -1.0f, 1.0f);
    ops::uniform_(B_cpu, -1.0f, 1.0f);
    
    Tensor C_cpu = ops::matmul(A_cpu, B_cpu);

#ifdef USE_CUDA_BACKEND
    {
        Tensor A_cuda = A_cpu.to(Device::CUDA);
        Tensor B_cuda = B_cpu.to(Device::CUDA);
        Tensor C_cuda = ops::matmul(A_cuda, B_cuda);
        
        assert_tensors_close(C_cpu, C_cuda.to(Device::CPU), 1e-4f, 
                            "Batch GEMM CPU vs CUDA");
        std::cout << "  Batch GEMM CPU vs CUDA passed!" << std::endl;
    }
#endif

#ifdef USE_HIP_BACKEND
    {
        Tensor A_hip = A_cpu.to(Device::HIP);
        Tensor B_hip = B_cpu.to(Device::HIP);
        Tensor C_hip = ops::matmul(A_hip, B_hip);
        
        assert_tensors_close(C_cpu, C_hip.to(Device::CPU), 1e-4f, 
                            "Batch GEMM CPU vs HIP");
        std::cout << "  Batch GEMM CPU vs HIP passed!" << std::endl;
    }
#endif
}

void test_batch_gemm_edge_consistency() {
    std::cout << "Testing batch GEMM edge case consistency across backends..." << std::endl;
    
    // Single batch, 1x1 matrices - edge case
    Tensor A_cpu = empty({1, 1, 1}, DType::Float32, Device::CPU);
    Tensor B_cpu = empty({1, 1, 1}, DType::Float32, Device::CPU);
    
    std::vector<float> a_data = {3.0f};
    std::vector<float> b_data = {4.0f};
    A_cpu.copy_from_host(a_data.data());
    B_cpu.copy_from_host(b_data.data());
    
    Tensor C_cpu = ops::matmul(A_cpu, B_cpu);

#ifdef USE_CUDA_BACKEND
    {
        Tensor A_cuda = A_cpu.to(Device::CUDA);
        Tensor B_cuda = B_cpu.to(Device::CUDA);
        Tensor C_cuda = ops::matmul(A_cuda, B_cuda);
        
        assert_tensors_close(C_cpu, C_cuda.to(Device::CPU), 1e-5f, 
                            "Batch GEMM edge CPU vs CUDA");
        std::cout << "  Batch GEMM edge CPU vs CUDA passed!" << std::endl;
    }
#endif

#ifdef USE_HIP_BACKEND
    {
        Tensor A_hip = A_cpu.to(Device::HIP);
        Tensor B_hip = B_cpu.to(Device::HIP);
        Tensor C_hip = ops::matmul(A_hip, B_hip);
        
        assert_tensors_close(C_cpu, C_hip.to(Device::CPU), 1e-5f, 
                            "Batch GEMM edge CPU vs HIP");
        std::cout << "  Batch GEMM edge CPU vs HIP passed!" << std::endl;
    }
#endif
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Batch GEMM Edge Case Tests ===" << std::endl;
    
    // Edge case tests
    test_batch_gemm_single_batch();
    test_batch_gemm_single_element_matrices();
    test_batch_gemm_tall_matrices();
    test_batch_gemm_wide_matrices();
    test_batch_gemm_square_matrices();
    test_batch_gemm_vector_matrix();
    test_batch_gemm_matrix_vector();
    
    // 4D tests
    test_batch_gemm_4d_basic();
    test_batch_gemm_4d_single_batch_head();
    
    // Backward tests
    test_batch_gemm_backward_basic();
    test_batch_gemm_numerical_gradient_A();
    test_batch_gemm_numerical_gradient_B();
    
    // Consistency tests
    test_batch_gemm_consistency();
    test_batch_gemm_edge_consistency();
    
    std::cout << "\n=== All Batch GEMM Edge Case Tests Passed! ===" << std::endl;
    return 0;
}
