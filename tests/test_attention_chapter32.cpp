/**
 * @file test_attention_chapter32.cpp
 * @brief Chapter 32 Comprehensive Tests for Attention Mechanism
 * 
 * Tests from Chapter 32 testing strategy matrix not covered elsewhere:
 * - Cross-attention (different seq lengths for Q vs K/V)
 * - Causal mask structure verification
 * - Softmax row sums = 1
 * - Large values numerical stability
 * - Known values hand-computed test
 * - Naive implementation comparison
 * - Determinism verification
 * - Very long sequences (S=4096)
 * - Dropout behavior tests
 */

#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/nn/functional.h>
#include <vesper/nn/transformer.h>
#include <vesper/ops/random.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/gemm.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <limits>
#include <chrono>

using namespace vesper;

// ============================================================================
// Utilities
// ============================================================================

bool contains_nan_or_inf(const Tensor& t) {
    std::vector<float> data(t.numel());
    t.contiguous().copy_to_host(data.data());
    for (float v : data) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
}

float max_abs_diff(const Tensor& t1, const Tensor& t2) {
    std::vector<float> d1(t1.numel()), d2(t2.numel());
    t1.contiguous().copy_to_host(d1.data());
    t2.contiguous().copy_to_host(d2.data());
    float max_diff = 0.0f;
    for (size_t i = 0; i < d1.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(d1[i] - d2[i]));
    }
    return max_diff;
}

// Naive attention implementation for correctness comparison
Tensor naive_attention(const Tensor& Q, const Tensor& K, const Tensor& V, bool is_causal) {
    // Q, K, V: [B, H, S_q, D] or [B, H, S_kv, D]
    auto B = Q.shape()[0];
    auto H = Q.shape()[1];
    auto S_q = Q.shape()[2];
    auto S_kv = K.shape()[2];
    auto D = Q.shape()[3];
    
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    // Get data
    std::vector<float> q_data(Q.numel()), k_data(K.numel()), v_data(V.numel());
    Q.contiguous().copy_to_host(q_data.data());
    K.contiguous().copy_to_host(k_data.data());
    V.contiguous().copy_to_host(v_data.data());
    
    std::vector<float> out_data(B * H * S_q * D, 0.0f);
    
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t i = 0; i < S_q; ++i) {
                // Compute scores for query i
                std::vector<float> scores(S_kv);
                float max_score = -std::numeric_limits<float>::infinity();
                
                for (int64_t j = 0; j < S_kv; ++j) {
                    float score = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        int64_t q_idx = ((b * H + h) * S_q + i) * D + d;
                        int64_t k_idx = ((b * H + h) * S_kv + j) * D + d;
                        score += q_data[q_idx] * k_data[k_idx];
                    }
                    score *= scale;
                    
                    // Causal mask
                    if (is_causal && j > i) {
                        score = -std::numeric_limits<float>::infinity();
                    }
                    
                    scores[j] = score;
                    max_score = std::max(max_score, score);
                }
                
                // Softmax
                float sum_exp = 0.0f;
                for (int64_t j = 0; j < S_kv; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                for (int64_t j = 0; j < S_kv; ++j) {
                    scores[j] /= sum_exp;
                }
                
                // Weighted sum of values
                for (int64_t d = 0; d < D; ++d) {
                    float val = 0.0f;
                    for (int64_t j = 0; j < S_kv; ++j) {
                        int64_t v_idx = ((b * H + h) * S_kv + j) * D + d;
                        val += scores[j] * v_data[v_idx];
                    }
                    int64_t out_idx = ((b * H + h) * S_q + i) * D + d;
                    out_data[out_idx] = val;
                }
            }
        }
    }
    
    Tensor out = empty({B, H, S_q, D}, DType::Float32, Device::CPU);
    out.copy_from_host(out_data.data());
    return out.to(Q.device());
}

// ============================================================================
// Shape Tests
// ============================================================================

void test_cross_attention_different_seq_lens() {
    std::cout << "Testing cross-attention with different sequence lengths..." << std::endl;
    
    int B = 2, H = 4;
    int S_q = 64;   // Query sequence length
    int S_kv = 128; // Key/Value sequence length
    int D = 32;
    
    Tensor Q = empty({B, H, S_q, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S_kv, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S_kv, D}, DType::Float32, Device::CPU);
    
    ops::normal_(Q, 0.0f, 0.5f);
    ops::normal_(K, 0.0f, 0.5f);
    ops::normal_(V, 0.0f, 0.5f);
    
    // Non-causal cross-attention (causal doesn't make sense with different seq lens)
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, false);
    
    assert(out.ndim() == 4);
    assert(out.shape()[0] == B);
    assert(out.shape()[1] == H);
    assert(out.shape()[2] == S_q);  // Output seq len = Query seq len
    assert(out.shape()[3] == D);
    assert(!contains_nan_or_inf(out));
    
    std::cout << "Cross-attention different seq lengths passed!" << std::endl;
}

// ============================================================================
// Causal Mask Tests
// ============================================================================

void test_causal_mask_structure() {
    std::cout << "Testing causal mask structure..." << std::endl;
    
    // Use attention with known Q, K to verify mask effect
    int B = 1, H = 1, S = 4, D = 4;
    
    // All Q and K identical → without mask, uniform attention
    Tensor Q = full({B, H, S, D}, DType::Float32, Device::CPU, 1.0f);
    Tensor K = full({B, H, S, D}, DType::Float32, Device::CPU, 1.0f);
    
    // V encodes position
    std::vector<float> v_data(S * D, 0.0f);
    for (int i = 0; i < S; ++i) {
        v_data[i * D] = static_cast<float>(i);  // V[i] = [i, 0, 0, 0]
    }
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    V.copy_from_host(v_data.data());
    
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    
    std::vector<float> out_data(S * D);
    out.copy_to_host(out_data.data());
    
    // Position 0: attends only to 0 → output[0,0] = 0
    // Position 1: attends to 0,1 → output[1,0] = (0+1)/2 = 0.5
    // Position 2: attends to 0,1,2 → output[2,0] = (0+1+2)/3 = 1.0
    // Position 3: attends to 0,1,2,3 → output[3,0] = (0+1+2+3)/4 = 1.5
    
    float tol = 1e-5f;
    assert(std::abs(out_data[0 * D] - 0.0f) < tol);
    assert(std::abs(out_data[1 * D] - 0.5f) < tol);
    assert(std::abs(out_data[2 * D] - 1.0f) < tol);
    assert(std::abs(out_data[3 * D] - 1.5f) < tol);
    
    std::cout << "Causal mask structure passed!" << std::endl;
}

// ============================================================================
// Numerical Tests
// ============================================================================

void test_attention_softmax_sum_to_one() {
    std::cout << "Testing attention softmax rows sum to 1..." << std::endl;
    
    // We need to access the attention weights, but our API doesn't expose them.
    // Instead, we verify indirectly: if attention weights sum to 1, 
    // then output = sum(weights * V) and if V is constant, output = V.
    
    int B = 2, H = 4, S = 8, D = 16;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::normal_(Q, 0.0f, 0.5f);
    ops::normal_(K, 0.0f, 0.5f);
    
    // V = constant tensor
    float v_val = 3.14159f;
    Tensor V = full({B, H, S, D}, DType::Float32, Device::CPU, v_val);
    
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, false);
    
    // If softmax sums to 1, output should be exactly v_val everywhere
    std::vector<float> out_data(out.numel());
    out.copy_to_host(out_data.data());
    
    for (float val : out_data) {
        assert(std::abs(val - v_val) < 1e-5f && "Softmax does not sum to 1");
    }
    
    std::cout << "Attention softmax sum to one passed!" << std::endl;
}

void test_large_values_stability() {
    std::cout << "Testing attention numerical stability with large values..." << std::endl;
    
    int B = 2, H = 4, S = 16, D = 32;
    
    // Large values that could cause overflow without proper numerical handling
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::normal_(Q, 0.0f, 10.0f);  // Std = 10 → values ~[-30, 30]
    ops::normal_(K, 0.0f, 10.0f);
    ops::normal_(V, 0.0f, 10.0f);
    
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    
    assert(!contains_nan_or_inf(out) && "Large values caused numerical instability");
    
    // Also test backward
    Q.set_requires_grad(true);
    K.set_requires_grad(true);
    V.set_requires_grad(true);
    
    Tensor out2 = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    Tensor loss = ops::sum(out2);
    loss.backward();
    
    assert(Q.grad().defined() && !contains_nan_or_inf(Q.grad()));
    assert(K.grad().defined() && !contains_nan_or_inf(K.grad()));
    assert(V.grad().defined() && !contains_nan_or_inf(V.grad()));
    
    std::cout << "Large values stability passed!" << std::endl;
}

// ============================================================================
// Correctness Tests
// ============================================================================

void test_known_values_hand_computed() {
    std::cout << "Testing attention with hand-computed values..." << std::endl;
    
    // Simple 1x1x2x2 case for manual verification
    int B = 1, H = 1, S = 2, D = 2;
    
    // Q = [[1, 0], [0, 1]]
    // K = [[1, 0], [0, 1]]
    // V = [[1, 2], [3, 4]]
    std::vector<float> q_data = {1, 0, 0, 1};
    std::vector<float> k_data = {1, 0, 0, 1};
    std::vector<float> v_data = {1, 2, 3, 4};
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    Q.copy_from_host(q_data.data());
    K.copy_from_host(k_data.data());
    V.copy_from_host(v_data.data());
    
    // Non-causal attention
    // scores = Q @ K.T / sqrt(D) = [[1, 0], [0, 1]] / sqrt(2) = [[0.707, 0], [0, 0.707]]
    // softmax([[0.707, 0], [0, 0.707]]) = [[exp(0.707), exp(0)], [exp(0), exp(0.707)]] / sum
    // exp(0.707) ≈ 2.028, exp(0) = 1
    // Row 0: [2.028, 1] / 3.028 = [0.6698, 0.3302]
    // Row 1: [1, 2.028] / 3.028 = [0.3302, 0.6698]
    // output[0] = 0.6698 * [1,2] + 0.3302 * [3,4] = [0.6698 + 0.9906, 1.3396 + 1.3208] = [1.6604, 2.6604]
    // output[1] = 0.3302 * [1,2] + 0.6698 * [3,4] = [0.3302 + 2.0094, 0.6604 + 2.6792] = [2.3396, 3.3396]
    
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, false);
    
    std::vector<float> out_data(S * D);
    out.copy_to_host(out_data.data());
    
    float scale = 1.0f / std::sqrt(2.0f);
    float exp_high = std::exp(scale);
    float exp_low = std::exp(0.0f);
    float sum = exp_high + exp_low;
    float p_high = exp_high / sum;
    float p_low = exp_low / sum;
    
    float expected_00 = p_high * 1.0f + p_low * 3.0f;
    float expected_01 = p_high * 2.0f + p_low * 4.0f;
    float expected_10 = p_low * 1.0f + p_high * 3.0f;
    float expected_11 = p_low * 2.0f + p_high * 4.0f;
    
    float tol = 1e-4f;
    assert(std::abs(out_data[0] - expected_00) < tol);
    assert(std::abs(out_data[1] - expected_01) < tol);
    assert(std::abs(out_data[2] - expected_10) < tol);
    assert(std::abs(out_data[3] - expected_11) < tol);
    
    std::cout << "  Expected: [" << expected_00 << ", " << expected_01 << ", " 
              << expected_10 << ", " << expected_11 << "]" << std::endl;
    std::cout << "  Got:      [" << out_data[0] << ", " << out_data[1] << ", "
              << out_data[2] << ", " << out_data[3] << "]" << std::endl;
    
    std::cout << "Hand-computed values passed!" << std::endl;
}

void test_vs_naive_implementation() {
    std::cout << "Testing attention vs naive implementation..." << std::endl;
    
    int B = 2, H = 2, S = 8, D = 8;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::uniform_(Q, -1.0f, 1.0f);
    ops::uniform_(K, -1.0f, 1.0f);
    ops::uniform_(V, -1.0f, 1.0f);
    
    // Test non-causal
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, false);
    Tensor out_naive = naive_attention(Q, K, V, false);
    
    float diff = max_abs_diff(out, out_naive);
    std::cout << "  Non-causal diff: " << diff << std::endl;
    assert(diff < 1e-5f && "Non-causal attention differs from naive");
    
    // Test causal
    Tensor out_causal = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    Tensor out_causal_naive = naive_attention(Q, K, V, true);
    
    float diff_causal = max_abs_diff(out_causal, out_causal_naive);
    std::cout << "  Causal diff: " << diff_causal << std::endl;
    assert(diff_causal < 1e-5f && "Causal attention differs from naive");
    
    std::cout << "Attention vs naive implementation passed!" << std::endl;
}

// ============================================================================
// Consistency Tests
// ============================================================================

void test_attention_determinism() {
    std::cout << "Testing attention determinism..." << std::endl;
    
    int B = 2, H = 4, S = 16, D = 32;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::uniform_(Q, -1.0f, 1.0f);
    ops::uniform_(K, -1.0f, 1.0f);
    ops::uniform_(V, -1.0f, 1.0f);
    
    // Run twice on CPU
    Tensor out1 = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    Tensor out2 = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    
    float cpu_diff = max_abs_diff(out1, out2);
    std::cout << "  CPU determinism diff: " << cpu_diff << std::endl;
    assert(cpu_diff == 0.0f && "CPU attention not deterministic");

#ifdef USE_HIP_BACKEND
    Tensor Q_hip = Q.to(Device::HIP);
    Tensor K_hip = K.to(Device::HIP);
    Tensor V_hip = V.to(Device::HIP);
    
    Tensor out1_hip = nn::functional::scaled_dot_product_attention(Q_hip, K_hip, V_hip, true);
    Tensor out2_hip = nn::functional::scaled_dot_product_attention(Q_hip, K_hip, V_hip, true);
    
    float hip_diff = max_abs_diff(out1_hip.to(Device::CPU), out2_hip.to(Device::CPU));
    std::cout << "  HIP determinism diff: " << hip_diff << std::endl;
    assert(hip_diff == 0.0f && "HIP attention not deterministic");
#endif
    
    std::cout << "Attention determinism passed!" << std::endl;
}

// ============================================================================
// Edge Cases
// ============================================================================

void test_very_long_sequence() {
    std::cout << "Testing attention with very long sequence (S=2048)..." << std::endl;
    
    // Note: S=4096 might be too slow for CPU testing, use 2048
    int B = 1, H = 4, S = 2048, D = 64;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::normal_(Q, 0.0f, 0.1f);  // Small std for stability
    ops::normal_(K, 0.0f, 0.1f);
    ops::normal_(V, 0.0f, 0.1f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    assert(!contains_nan_or_inf(out));
    assert(out.shape()[2] == S);
    
    std::cout << "  S=2048 CPU time: " << duration.count() << " ms" << std::endl;

#ifdef USE_HIP_BACKEND
    Tensor Q_hip = Q.to(Device::HIP);
    Tensor K_hip = K.to(Device::HIP);
    Tensor V_hip = V.to(Device::HIP);
    
    // Warmup
    nn::functional::scaled_dot_product_attention(Q_hip, K_hip, V_hip, true);
    
    start = std::chrono::high_resolution_clock::now();
    Tensor out_hip = nn::functional::scaled_dot_product_attention(Q_hip, K_hip, V_hip, true);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    assert(!contains_nan_or_inf(out_hip.to(Device::CPU)));
    std::cout << "  S=2048 HIP time: " << duration.count() << " ms" << std::endl;
#endif
    
    std::cout << "Very long sequence passed!" << std::endl;
}

// ============================================================================
// Dropout Tests
// ============================================================================

void test_attention_dropout_zero() {
    std::cout << "Testing attention with dropout=0..." << std::endl;
    
    int B = 2, H = 4, S = 8, D = 16;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::uniform_(Q, -1.0f, 1.0f);
    ops::uniform_(K, -1.0f, 1.0f);
    ops::uniform_(V, -1.0f, 1.0f);
    
    // dropout_p = 0 should give identical results
    Tensor out1 = nn::functional::scaled_dot_product_attention(Q, K, V, true, 0.0);
    Tensor out2 = nn::functional::scaled_dot_product_attention(Q, K, V, true, 0.0);
    
    float diff = max_abs_diff(out1, out2);
    assert(diff == 0.0f && "Dropout=0 should be deterministic");
    
    std::cout << "Attention dropout=0 passed!" << std::endl;
}

void test_attention_dropout_nonzero() {
    std::cout << "Testing attention with dropout>0..." << std::endl;
    
    int B = 2, H = 4, S = 16, D = 32;
    
    Tensor Q = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K = empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V = empty({B, H, S, D}, DType::Float32, Device::CPU);
    
    ops::uniform_(Q, -1.0f, 1.0f);
    ops::uniform_(K, -1.0f, 1.0f);
    ops::uniform_(V, -1.0f, 1.0f);
    
    // dropout_p > 0 should give different results each run (in training mode)
    Tensor out1 = nn::functional::scaled_dot_product_attention(Q, K, V, false, 0.5);
    Tensor out2 = nn::functional::scaled_dot_product_attention(Q, K, V, false, 0.5);
    
    float diff = max_abs_diff(out1, out2);
    std::cout << "  Dropout=0.5 diff between runs: " << diff << std::endl;
    
    // Should be different due to random dropout
    assert(diff > 0.0f && "Dropout>0 should produce different results");
    
    // Results should still be valid
    assert(!contains_nan_or_inf(out1));
    assert(!contains_nan_or_inf(out2));
    
    std::cout << "Attention dropout>0 passed!" << std::endl;
}

// ============================================================================
// MultiHeadAttention Module Tests
// ============================================================================

void test_mha_different_head_counts() {
    std::cout << "Testing MHA with different head counts..." << std::endl;
    
    int embed_dim = 64;
    // Note: head_dim=1 (64 heads) is not tested as RoPE requires head_dim >= 2
    std::vector<int> head_counts = {1, 2, 4, 8, 16, 32};
    
    for (int num_heads : head_counts) {
        if (embed_dim % num_heads != 0) continue;
        int head_dim = embed_dim / num_heads;
        if (head_dim < 2) continue;  // RoPE needs pairs
        
        nn::MultiHeadAttention mha(embed_dim, num_heads, 0.0f);
        
        Tensor x = empty({2, 8, embed_dim}, DType::Float32, Device::CPU);
        ops::normal_(x, 0.0f, 0.1f);
        
        Tensor y = mha.forward(x, true);
        
        assert(y.shape()[0] == 2);
        assert(y.shape()[1] == 8);
        assert(y.shape()[2] == embed_dim);
        assert(!contains_nan_or_inf(y));
        
        std::cout << "  num_heads=" << num_heads << " (head_dim=" << head_dim << ") passed" << std::endl;
    }
    
    std::cout << "MHA different head counts passed!" << std::endl;
}

void test_mha_large_batch() {
    std::cout << "Testing MHA with large batch..." << std::endl;
    
    int embed_dim = 64;
    int num_heads = 4;
    int batch_size = 128;
    int seq_len = 32;
    
    nn::MultiHeadAttention mha(embed_dim, num_heads, 0.0f);
    
    Tensor x = empty({batch_size, seq_len, embed_dim}, DType::Float32, Device::CPU);
    ops::normal_(x, 0.0f, 0.1f);
    
    Tensor y = mha.forward(x, true);
    
    assert(y.shape()[0] == batch_size);
    assert(!contains_nan_or_inf(y));
    
    std::cout << "MHA large batch (B=" << batch_size << ") passed!" << std::endl;
}

// ============================================================================
// Backend Consistency for Chapter 32
// ============================================================================

void test_attention_cpu_vs_hip_comprehensive() {
    std::cout << "Testing comprehensive CPU vs HIP attention..." << std::endl;

#ifndef USE_HIP_BACKEND
    std::cout << "  HIP not available, skipping." << std::endl;
    return;
#else
    // Test multiple configurations
    struct TestConfig {
        int B, H, S, D;
        bool causal;
    };
    
    std::vector<TestConfig> configs = {
        {1, 1, 4, 4, false},    // Tiny
        {2, 4, 32, 64, false},  // Standard non-causal
        {2, 4, 32, 64, true},   // Standard causal
        {1, 8, 128, 64, true},  // Single batch, many heads
        {4, 1, 64, 128, true},  // Single head, large D
    };
    
    for (const auto& cfg : configs) {
        Tensor Q = empty({cfg.B, cfg.H, cfg.S, cfg.D}, DType::Float32, Device::CPU);
        Tensor K = empty({cfg.B, cfg.H, cfg.S, cfg.D}, DType::Float32, Device::CPU);
        Tensor V = empty({cfg.B, cfg.H, cfg.S, cfg.D}, DType::Float32, Device::CPU);
        
        ops::uniform_(Q, -0.5f, 0.5f);
        ops::uniform_(K, -0.5f, 0.5f);
        ops::uniform_(V, -0.5f, 0.5f);
        
        Tensor out_cpu = nn::functional::scaled_dot_product_attention(Q, K, V, cfg.causal);
        
        Tensor Q_hip = Q.to(Device::HIP);
        Tensor K_hip = K.to(Device::HIP);
        Tensor V_hip = V.to(Device::HIP);
        
        Tensor out_hip = nn::functional::scaled_dot_product_attention(Q_hip, K_hip, V_hip, cfg.causal);
        
        float diff = max_abs_diff(out_cpu, out_hip.to(Device::CPU));
        
        std::cout << "  Config (B=" << cfg.B << ",H=" << cfg.H << ",S=" << cfg.S 
                  << ",D=" << cfg.D << ",causal=" << cfg.causal << "): diff=" << diff << std::endl;
        
        assert(diff < 1e-4f && "CPU vs HIP mismatch");
    }
    
    std::cout << "CPU vs HIP comprehensive passed!" << std::endl;
#endif
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Chapter 32 Attention Comprehensive Tests ===" << std::endl;
    
    // Shape tests
    test_cross_attention_different_seq_lens();
    
    // Causal mask tests
    test_causal_mask_structure();
    
    // Numerical tests
    test_attention_softmax_sum_to_one();
    test_large_values_stability();
    
    // Correctness tests
    test_known_values_hand_computed();
    test_vs_naive_implementation();
    
    // Consistency tests
    test_attention_determinism();
    
    // Edge cases
    test_very_long_sequence();
    
    // Dropout tests
    test_attention_dropout_zero();
    test_attention_dropout_nonzero();
    
    // MHA module tests
    test_mha_different_head_counts();
    test_mha_large_batch();
    
    // Backend consistency
    test_attention_cpu_vs_hip_comprehensive();
    
    std::cout << "\n=== All Chapter 32 Tests Passed! ===" << std::endl;
    return 0;
}
