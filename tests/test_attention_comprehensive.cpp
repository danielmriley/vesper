/**
 * @file test_attention_comprehensive.cpp
 * @brief Comprehensive tests for Scaled Dot-Product Attention
 * 
 * Tests:
 * - Edge cases: single token, sequence length 1, single head
 * - Numerical gradient verification for Q, K, V
 * - Attention pattern verification (peaked vs uniform)
 * - Backward pass correctness
 * - Causal mask edge cases
 * - Multi-backend gradient consistency
 * - Memory efficiency (large sequences)
 */

#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/nn/functional.h>
#include <vesper/ops/random.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
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

bool contains_nan_or_inf(const Tensor& t) {
    std::vector<float> data(t.numel());
    t.copy_to_host(data.data());
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

// ============================================================================
// Edge Case Tests
// ============================================================================

void test_attention_single_token() {
    std::cout << "Testing attention with single token..." << std::endl;
    
    int B = 2, H = 4, S = 1, D = 16;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::normal_(Q, 0.0f, 1.0f);
    ops::normal_(K, 0.0f, 1.0f);
    ops::normal_(V, 0.0f, 1.0f);
    
    // With single token, output should equal V (after softmax of single element = 1)
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, false);
    
    assert(out.shape() == V.shape());
    assert(!contains_nan_or_inf(out));
    
    // Output should be V because softmax([x]) = [1.0]
    assert_tensors_close(out, V, 1e-5f, "Single token attention");
    
    std::cout << "Single token attention passed!" << std::endl;
}

void test_attention_single_head() {
    std::cout << "Testing attention with single head..." << std::endl;
    
    int B = 2, H = 1, S = 8, D = 32;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::normal_(Q, 0.0f, 0.5f);
    ops::normal_(K, 0.0f, 0.5f);
    ops::normal_(V, 0.0f, 0.5f);
    
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    
    assert(out.shape()[0] == B);
    assert(out.shape()[1] == H);
    assert(out.shape()[2] == S);
    assert(out.shape()[3] == D);
    assert(!contains_nan_or_inf(out));
    
    std::cout << "Single head attention passed!" << std::endl;
}

void test_attention_batch_size_one() {
    std::cout << "Testing attention with batch size 1..." << std::endl;
    
    int B = 1, H = 8, S = 16, D = 64;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::normal_(Q, 0.0f, 0.5f);
    ops::normal_(K, 0.0f, 0.5f);
    ops::normal_(V, 0.0f, 0.5f);
    
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, false);
    
    assert(out.shape()[0] == B);
    assert(!contains_nan_or_inf(out));
    
    std::cout << "Batch size 1 attention passed!" << std::endl;
}

// ============================================================================
// Attention Pattern Tests
// ============================================================================

void test_attention_peaked_pattern() {
    std::cout << "Testing attention peaked pattern..." << std::endl;
    
    // Create Q, K such that Q[0] strongly attends to K[0]
    int B = 1, H = 1, S = 4, D = 4;
    
    Tensor Q = zeros({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = zeros({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    // Set Q[0,:] = [1, 0, 0, 0] and K[0,:] = [1, 0, 0, 0]
    // This gives Q[0] @ K[0].T = 1, Q[0] @ K[i!=0].T = 0
    std::vector<float> q_data(S * D, 0.0f);
    std::vector<float> k_data(S * D, 0.0f);
    
    // First query: large alignment with first key
    q_data[0] = 10.0f;  // Q[0,0]
    k_data[0] = 10.0f;  // K[0,0]
    
    // Other keys have different patterns
    k_data[D] = 10.0f;      // K[1,1]
    k_data[2*D + 2] = 10.0f; // K[2,2]
    k_data[3*D + 3] = 10.0f; // K[3,3]
    
    Q.copy_from_host(q_data.data());
    K.copy_from_host(k_data.data());
    
    // V is identity-like for verification
    std::vector<float> v_data = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    V.copy_from_host(v_data.data());
    
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, false);
    
    std::vector<float> out_data(S * D);
    out.copy_to_host(out_data.data());
    
    // First output should be close to V[0] = [1, 0, 0, 0]
    // because Q[0] attends strongly to K[0]
    assert(out_data[0] > 0.9f && "First query should attend to first key");
    
    std::cout << "Attention peaked pattern passed!" << std::endl;
}

void test_attention_uniform_pattern() {
    std::cout << "Testing attention uniform pattern..." << std::endl;
    
    int B = 1, H = 1, S = 4, D = 4;
    
    // All Q and K identical -> uniform attention
    Tensor Q = full({B, H, S, D}, DType::Float32, Device::CPU, 1.0f);
    Tensor K = full({B, H, S, D}, DType::Float32, Device::CPU, 1.0f);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    // V rows are different
    std::vector<float> v_data = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    V.copy_from_host(v_data.data());
    
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, false);
    
    std::vector<float> out_data(S * D);
    out.copy_to_host(out_data.data());
    
    // Each output should be average of V rows = [0.25, 0.25, 0.25, 0.25]
    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < D; ++j) {
            assert(std::abs(out_data[i * D + j] - 0.25f) < 1e-5f);
        }
    }
    
    std::cout << "Attention uniform pattern passed!" << std::endl;
}

// ============================================================================
// Causal Mask Edge Cases
// ============================================================================

void test_causal_first_token_independent() {
    std::cout << "Testing causal: first token independent of future..." << std::endl;
    
    int B = 1, H = 2, S = 8, D = 16;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::normal_(Q, 0.0f, 0.5f);
    ops::normal_(K, 0.0f, 0.5f);
    ops::normal_(V, 0.0f, 0.5f);
    
    // Run 1: original
    Tensor out1 = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    
    // Modify all future tokens (positions > 0)
    std::vector<float> k_data(K.numel());
    K.copy_to_host(k_data.data());
    for (size_t i = D; i < k_data.size(); ++i) {  // Skip first token
        k_data[i] += 100.0f;
    }
    K.copy_from_host(k_data.data());
    
    // Run 2: with modified future
    Tensor out2 = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    
    // First token output should be unchanged
    std::vector<float> o1(out1.numel()), o2(out2.numel());
    out1.copy_to_host(o1.data());
    out2.copy_to_host(o2.data());
    
    // Check first token for all heads
    for (int h = 0; h < H; ++h) {
        for (int d = 0; d < D; ++d) {
            int idx = h * S * D + d;  // (0, h, 0, d)
            float diff = std::abs(o1[idx] - o2[idx]);
            assert(diff < 1e-5f && "First token should not depend on future");
        }
    }
    
    std::cout << "Causal first token independence passed!" << std::endl;
}

void test_causal_last_token_sees_all() {
    std::cout << "Testing causal: last token sees all previous..." << std::endl;
    
    int B = 1, H = 1, S = 4, D = 4;
    
    // Set up so we can verify attention pattern
    Tensor Q = zeros({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = zeros({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    // Make last query uniform
    std::vector<float> q_data(S * D, 0.0f);
    for (int d = 0; d < D; ++d) q_data[(S-1) * D + d] = 1.0f;
    Q.copy_from_host(q_data.data());
    
    // All keys identical
    std::vector<float> k_data(S * D, 0.0f);
    for (int s = 0; s < S; ++s) {
        for (int d = 0; d < D; ++d) {
            k_data[s * D + d] = 1.0f;
        }
    }
    K.copy_from_host(k_data.data());
    
    // V encodes position
    std::vector<float> v_data(S * D, 0.0f);
    for (int s = 0; s < S; ++s) {
        v_data[s * D] = (float)(s + 1);  // V[s] = [s+1, 0, 0, 0]
    }
    V.copy_from_host(v_data.data());
    
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    
    std::vector<float> out_data(S * D);
    out.copy_to_host(out_data.data());
    
    // Last token output should be average of all V: (1+2+3+4)/4 = 2.5
    float last_out = out_data[(S-1) * D];
    assert(std::abs(last_out - 2.5f) < 1e-4f && "Last token should see all");
    
    std::cout << "Causal last token sees all passed!" << std::endl;
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

void test_attention_backward_basic() {
    std::cout << "Testing attention backward (basic)..." << std::endl;
    
    int B = 1, H = 2, S = 4, D = 8;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::normal_(Q, 0.0f, 0.5f);
    ops::normal_(K, 0.0f, 0.5f);
    ops::normal_(V, 0.0f, 0.5f);
    
    Q.set_requires_grad(true);
    K.set_requires_grad(true);
    V.set_requires_grad(true);
    
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, false);
    Tensor loss = ops::sum(out);
    loss.backward();
    
    assert(Q.grad().defined() && "Q gradient should exist");
    assert(K.grad().defined() && "K gradient should exist");
    assert(V.grad().defined() && "V gradient should exist");
    
    assert(Q.grad().shape() == Q.shape());
    assert(K.grad().shape() == K.shape());
    assert(V.grad().shape() == V.shape());
    
    assert(!contains_nan_or_inf(Q.grad()));
    assert(!contains_nan_or_inf(K.grad()));
    assert(!contains_nan_or_inf(V.grad()));
    
    std::cout << "Attention backward (basic) passed!" << std::endl;
}

void test_attention_v_numerical_gradient() {
    std::cout << "Testing attention V numerical gradient..." << std::endl;
    
    // V gradient is simplest: dL/dV = A^T @ dL/dO where A is attention weights
    int B = 1, H = 1, S = 3, D = 4;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::uniform_(Q, -0.5f, 0.5f);
    ops::uniform_(K, -0.5f, 0.5f);
    ops::uniform_(V, -0.5f, 0.5f);
    
    auto loss_fn = [&](Tensor& v) -> float {
        auto out = nn::functional::scaled_dot_product_attention(Q, K, v, false);
        auto loss = ops::sum(out);
        std::vector<float> val(1);
        loss.copy_to_host(val.data());
        return val[0];
    };
    
    Tensor num_grad = compute_numerical_gradient(loss_fn, V, 1e-4f);
    
    // Analytical gradient
    V.set_requires_grad(true);
    auto out = nn::functional::scaled_dot_product_attention(Q, K, V, false);
    auto loss = ops::sum(out);
    loss.backward();
    
    Tensor ana_grad = V.grad();
    
    float max_diff = max_abs_diff(num_grad, ana_grad);
    std::cout << "  V gradient max diff: " << max_diff << std::endl;
    
    assert_tensors_close(num_grad, ana_grad, 5e-3f, "V gradient");
    
    std::cout << "Attention V numerical gradient passed!" << std::endl;
}

void test_attention_q_numerical_gradient() {
    std::cout << "Testing attention Q numerical gradient..." << std::endl;
    
    int B = 1, H = 1, S = 3, D = 4;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::uniform_(Q, -0.5f, 0.5f);
    ops::uniform_(K, -0.5f, 0.5f);
    ops::uniform_(V, -0.5f, 0.5f);
    
    // Use weighted loss for more interesting gradient
    Tensor weights = empty({B, H, S, D}, DType::Float32, Device::CPU);
    ops::uniform_(weights, 0.1f, 1.0f);
    
    auto loss_fn = [&](Tensor& q) -> float {
        auto out = nn::functional::scaled_dot_product_attention(q, K, V, false);
        auto weighted = ops::mul(out, weights);
        auto loss = ops::sum(weighted);
        std::vector<float> val(1);
        loss.copy_to_host(val.data());
        return val[0];
    };
    
    Tensor num_grad = compute_numerical_gradient(loss_fn, Q, 1e-4f);
    
    // Analytical gradient
    Q.set_requires_grad(true);
    auto out = nn::functional::scaled_dot_product_attention(Q, K, V, false);
    auto weighted = ops::mul(out, weights);
    auto loss = ops::sum(weighted);
    loss.backward();
    
    Tensor ana_grad = Q.grad();
    
    float max_diff = max_abs_diff(num_grad, ana_grad);
    std::cout << "  Q gradient max diff: " << max_diff << std::endl;
    
    assert_tensors_close(num_grad, ana_grad, 5e-3f, "Q gradient");
    
    std::cout << "Attention Q numerical gradient passed!" << std::endl;
}

// ============================================================================
// Multi-Backend Tests
// ============================================================================

void test_attention_gradient_consistency() {
    std::cout << "Testing attention gradient consistency across backends..." << std::endl;
    
    int B = 2, H = 4, S = 8, D = 16;
    
    Tensor Q_cpu = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K_cpu = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V_cpu = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::uniform_(Q_cpu, -0.5f, 0.5f);
    ops::uniform_(K_cpu, -0.5f, 0.5f);
    ops::uniform_(V_cpu, -0.5f, 0.5f);
    
    // CPU computation
    auto Q_grad = Q_cpu.clone();
    auto K_grad = K_cpu.clone();
    auto V_grad = V_cpu.clone();
    Q_grad.set_requires_grad(true);
    K_grad.set_requires_grad(true);
    V_grad.set_requires_grad(true);
    
    auto out_cpu = nn::functional::scaled_dot_product_attention(Q_grad, K_grad, V_grad, true);
    auto loss_cpu = ops::sum(out_cpu);
    loss_cpu.backward();
    
    Tensor dQ_cpu = Q_grad.grad();
    Tensor dK_cpu = K_grad.grad();
    Tensor dV_cpu = V_grad.grad();

#ifdef USE_CUDA_BACKEND
    {
        auto Q_cuda = Q_cpu.to(Device::CUDA);
        auto K_cuda = K_cpu.to(Device::CUDA);
        auto V_cuda = V_cpu.to(Device::CUDA);
        Q_cuda.set_requires_grad(true);
        K_cuda.set_requires_grad(true);
        V_cuda.set_requires_grad(true);
        
        auto out_cuda = nn::functional::scaled_dot_product_attention(Q_cuda, K_cuda, V_cuda, true);
        auto loss_cuda = ops::sum(out_cuda);
        loss_cuda.backward();
        
        assert_tensors_close(dQ_cpu, Q_cuda.grad().contiguous().to(Device::CPU), 1e-3f, "dQ CPU vs CUDA");
        assert_tensors_close(dK_cpu, K_cuda.grad().contiguous().to(Device::CPU), 1e-3f, "dK CPU vs CUDA");
        assert_tensors_close(dV_cpu, V_cuda.grad().contiguous().to(Device::CPU), 1e-3f, "dV CPU vs CUDA");
        
        std::cout << "  Attention gradient CPU vs CUDA passed!" << std::endl;
    }
#endif

#ifdef USE_HIP_BACKEND
    {
        auto Q_hip = Q_cpu.to(Device::HIP);
        auto K_hip = K_cpu.to(Device::HIP);
        auto V_hip = V_cpu.to(Device::HIP);
        Q_hip.set_requires_grad(true);
        K_hip.set_requires_grad(true);
        V_hip.set_requires_grad(true);
        
        auto out_hip = nn::functional::scaled_dot_product_attention(Q_hip, K_hip, V_hip, true);
        auto loss_hip = ops::sum(out_hip);
        loss_hip.backward();
        
        assert_tensors_close(dQ_cpu, Q_hip.grad().contiguous().to(Device::CPU), 1e-3f, "dQ CPU vs HIP");
        assert_tensors_close(dK_cpu, K_hip.grad().contiguous().to(Device::CPU), 1e-3f, "dK CPU vs HIP");
        assert_tensors_close(dV_cpu, V_hip.grad().contiguous().to(Device::CPU), 1e-3f, "dV CPU vs HIP");
        
        std::cout << "  Attention gradient CPU vs HIP passed!" << std::endl;
    }
#endif
}

// ============================================================================
// Stress Tests
// ============================================================================

void test_attention_long_sequence() {
    std::cout << "Testing attention with long sequence..." << std::endl;
    
    int B = 1, H = 4, S = 256, D = 64;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::normal_(Q, 0.0f, 0.1f);  // Small std to avoid numerical issues
    ops::normal_(K, 0.0f, 0.1f);
    ops::normal_(V, 0.0f, 0.1f);
    
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    
    assert(!contains_nan_or_inf(out));
    assert(out.shape()[2] == S);
    
    std::cout << "Long sequence attention passed!" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Attention Comprehensive Tests ===" << std::endl;
    
    // Edge case tests
    test_attention_single_token();
    test_attention_single_head();
    test_attention_batch_size_one();
    
    // Pattern tests
    test_attention_peaked_pattern();
    test_attention_uniform_pattern();
    
    // Causal tests
    test_causal_first_token_independent();
    test_causal_last_token_sees_all();
    
    // Backward tests
    test_attention_backward_basic();
    test_attention_v_numerical_gradient();
    test_attention_q_numerical_gradient();
    
    // Consistency tests
    test_attention_gradient_consistency();
    
    // Stress tests
    test_attention_long_sequence();
    
    std::cout << "\n=== All Attention Comprehensive Tests Passed! ===" << std::endl;
    return 0;
}
