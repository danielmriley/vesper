/**
 * @file test_kv_cache_chapter33.cpp
 * @brief Chapter 33.2 Comprehensive Tests for KV Cache
 * 
 * Tests from Chapter 33.2 testing strategy matrix:
 * - Correctness: cached vs full forward equivalence
 * - Cache state management
 * - Position encoding (RoPE) correctness with cache
 * - Memory efficiency (views not copies)
 * - Multi-layer cache tests
 * - Batch independence
 * - Edge cases (single token, max length)
 */

#include <vesper/nn/transformer.h>
#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>
#include <vesper/ops/random.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
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

void assert_tensors_close(const Tensor& t1, const Tensor& t2, float tol,
                          const std::string& msg) {
    float diff = max_abs_diff(t1, t2);
    if (diff > tol) {
        std::cerr << msg << " - Max diff: " << diff << " > tol: " << tol << std::endl;
        assert(false);
    }
}

// ============================================================================
// Correctness Tests - Cache vs Full Forward Equivalence
// ============================================================================

void test_cached_vs_full_forward_mha() {
    std::cout << "Testing MHA cached vs full forward equivalence..." << std::endl;
    
    int E = 32, H = 4;
    int B = 1, S = 8;
    
    nn::MultiHeadAttention mha(E, H, 0.0f);
    mha.eval();  // No dropout
    
    // Create full sequence input
    Tensor x_full = empty({B, S, E}, DType::Float32, Device::CPU);
    ops::uniform_(x_full, -0.5f, 0.5f);
    
    // Full forward (no cache, causal)
    Tensor out_full = mha.forward(x_full, true);
    
    // Create cache
    int head_dim = E / H;
    nn::KVCache cache(B, H, S * 2, head_dim, Device::CPU);
    
    // Process token by token with cache
    std::vector<Tensor> cached_outputs;
    
    for (int t = 0; t < S; ++t) {
        // Extract single token
        Tensor x_t = x_full.index({Slice(), Slice(t, t+1), Slice()});
        
        // Forward with cache
        Tensor out_t = mha.forward(x_t, &cache, t);
        cached_outputs.push_back(out_t);
    }
    
    // Compare outputs
    for (int t = 0; t < S; ++t) {
        Tensor full_t = out_full.index({Slice(), Slice(t, t+1), Slice()});
        float diff = max_abs_diff(full_t, cached_outputs[t]);
        
        if (diff > 1e-4f) {
            std::cerr << "  Token " << t << ": diff=" << diff << std::endl;
        }
        assert(diff < 1e-4f && "Cached output differs from full forward");
    }
    
    std::cout << "MHA cached vs full forward equivalence passed!" << std::endl;
}

void test_prefill_equivalence() {
    std::cout << "Testing prefill equivalence..." << std::endl;
    
    int E = 64, H = 4;
    int B = 2, S = 16;
    
    nn::MultiHeadAttention mha(E, H, 0.0f);
    mha.eval();
    
    Tensor x = empty({B, S, E}, DType::Float32, Device::CPU);
    ops::uniform_(x, -0.5f, 0.5f);
    
    // Full forward without cache
    Tensor out_no_cache = mha.forward(x, true);
    
    // Prefill with cache (process all tokens at once)
    int head_dim = E / H;
    nn::KVCache cache(B, H, S * 2, head_dim, Device::CPU);
    Tensor out_with_cache = mha.forward(x, &cache, 0);
    
    float diff = max_abs_diff(out_no_cache, out_with_cache);
    std::cout << "  Prefill diff: " << diff << std::endl;
    
    assert(diff < 1e-5f && "Prefill with cache differs from no-cache forward");
    assert(cache.current_seq_len() == S && "Cache length mismatch after prefill");
    
    std::cout << "Prefill equivalence passed!" << std::endl;
}

void test_incremental_decode() {
    std::cout << "Testing incremental decode correctness..." << std::endl;
    
    int E = 32, H = 2;
    int B = 1, prompt_len = 4, decode_steps = 4;
    
    nn::MultiHeadAttention mha(E, H, 0.0f);
    mha.eval();
    
    // Create prompt + decode tokens
    int total_len = prompt_len + decode_steps;
    Tensor x_full = empty({B, total_len, E}, DType::Float32, Device::CPU);
    ops::uniform_(x_full, -0.5f, 0.5f);
    
    // Full forward for reference
    Tensor out_full = mha.forward(x_full, true);
    
    // Incremental: prefill + decode
    int head_dim = E / H;
    nn::KVCache cache(B, H, total_len * 2, head_dim, Device::CPU);
    
    // Prefill
    Tensor prompt = x_full.index({Slice(), Slice(0, prompt_len), Slice()});
    Tensor out_prefill = mha.forward(prompt, &cache, 0);
    
    // Decode token by token
    std::vector<Tensor> decode_outputs;
    for (int t = 0; t < decode_steps; ++t) {
        int pos = prompt_len + t;
        Tensor x_t = x_full.index({Slice(), Slice(pos, pos+1), Slice()});
        Tensor out_t = mha.forward(x_t, &cache, pos);
        decode_outputs.push_back(out_t);
    }
    
    // Compare decode outputs with sliced full forward
    for (int t = 0; t < decode_steps; ++t) {
        int pos = prompt_len + t;
        Tensor full_t = out_full.index({Slice(), Slice(pos, pos+1), Slice()});
        float diff = max_abs_diff(full_t, decode_outputs[t]);
        
        if (diff > 1e-4f) {
            std::cerr << "  Decode step " << t << " (pos " << pos << "): diff=" << diff << std::endl;
        }
        assert(diff < 1e-4f && "Incremental decode differs from full forward");
    }
    
    std::cout << "Incremental decode passed!" << std::endl;
}

// ============================================================================
// Cache State Tests
// ============================================================================

void test_cache_state_after_operations() {
    std::cout << "Testing cache state after operations..." << std::endl;
    
    int B = 2, H = 4, max_seq = 64, D = 16;
    nn::KVCache cache(B, H, max_seq, D, Device::CPU);
    
    assert(cache.current_seq_len() == 0 && "Initial length should be 0");
    assert(cache.get_max_seq_len() == max_seq && "Max seq len mismatch");
    
    // Prefill with 10 tokens
    Tensor k1 = empty({B, H, 10, D}, DType::Float32, Device::CPU);
    Tensor v1 = empty({B, H, 10, D}, DType::Float32, Device::CPU);
    ops::uniform_(k1, -1.0f, 1.0f);
    ops::uniform_(v1, -1.0f, 1.0f);
    
    auto [k_out1, v_out1] = cache.update(k1, v1, 0);
    
    assert(cache.current_seq_len() == 10 && "Length should be 10 after prefill");
    assert(k_out1.shape()[2] == 10 && "K output shape mismatch");
    assert(v_out1.shape()[2] == 10 && "V output shape mismatch");
    
    // Decode 5 more tokens one at a time
    for (int i = 0; i < 5; ++i) {
        Tensor k_new = empty({B, H, 1, D}, DType::Float32, Device::CPU);
        Tensor v_new = empty({B, H, 1, D}, DType::Float32, Device::CPU);
        ops::uniform_(k_new, -1.0f, 1.0f);
        ops::uniform_(v_new, -1.0f, 1.0f);
        
        int pos = 10 + i;
        auto [k_out, v_out] = cache.update(k_new, v_new, pos);
        
        assert(cache.current_seq_len() == pos + 1);
        assert(k_out.shape()[2] == pos + 1);
    }
    
    assert(cache.current_seq_len() == 15 && "Final length should be 15");
    
    // Reset
    cache.reset();
    assert(cache.current_seq_len() == 0 && "Length should be 0 after reset");
    
    std::cout << "Cache state after operations passed!" << std::endl;
}

void test_cache_preserves_values() {
    std::cout << "Testing cache preserves values correctly..." << std::endl;
    
    int B = 1, H = 1, max_seq = 16, D = 4;
    nn::KVCache cache(B, H, max_seq, D, Device::CPU);
    
    // Write specific values at position 0
    std::vector<float> k1_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> v1_data = {5.0f, 6.0f, 7.0f, 8.0f};
    
    Tensor k1 = empty({B, H, 1, D}, DType::Float32, Device::CPU);
    Tensor v1 = empty({B, H, 1, D}, DType::Float32, Device::CPU);
    k1.copy_from_host(k1_data.data());
    v1.copy_from_host(v1_data.data());
    
    cache.update(k1, v1, 0);
    
    // Write different values at position 1
    std::vector<float> k2_data = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> v2_data = {50.0f, 60.0f, 70.0f, 80.0f};
    
    Tensor k2 = empty({B, H, 1, D}, DType::Float32, Device::CPU);
    Tensor v2 = empty({B, H, 1, D}, DType::Float32, Device::CPU);
    k2.copy_from_host(k2_data.data());
    v2.copy_from_host(v2_data.data());
    
    auto [k_out, v_out] = cache.update(k2, v2, 1);
    
    // Verify both positions have correct values
    std::vector<float> k_result(2 * D), v_result(2 * D);
    k_out.copy_to_host(k_result.data());
    v_out.copy_to_host(v_result.data());
    
    // Position 0
    for (int i = 0; i < D; ++i) {
        assert(std::abs(k_result[i] - k1_data[i]) < 1e-6f);
        assert(std::abs(v_result[i] - v1_data[i]) < 1e-6f);
    }
    
    // Position 1
    for (int i = 0; i < D; ++i) {
        assert(std::abs(k_result[D + i] - k2_data[i]) < 1e-6f);
        assert(std::abs(v_result[D + i] - v2_data[i]) < 1e-6f);
    }
    
    std::cout << "Cache preserves values passed!" << std::endl;
}

// ============================================================================
// Position Encoding Tests
// ============================================================================

void test_rope_positions_consistency() {
    std::cout << "Testing RoPE positions are consistent with cache..." << std::endl;
    
    // This test verifies that cached forward produces same results as full forward
    // which implicitly tests that RoPE positions are correctly computed
    
    int E = 32, H = 2;
    int B = 1, S = 6;
    
    nn::MultiHeadAttention mha(E, H, 0.0f);
    mha.eval();
    
    Tensor x = empty({B, S, E}, DType::Float32, Device::CPU);
    ops::uniform_(x, -0.5f, 0.5f);
    
    // Full forward
    Tensor out_full = mha.forward(x, true);
    
    // Cached forward - prefill first half, decode second half
    int head_dim = E / H;
    nn::KVCache cache(B, H, S * 2, head_dim, Device::CPU);
    
    int prefill_len = 3;
    Tensor x_prefill = x.index({Slice(), Slice(0, prefill_len), Slice()});
    Tensor out_prefill = mha.forward(x_prefill, &cache, 0);
    
    // Decode remaining tokens
    std::vector<Tensor> decode_outs;
    for (int t = prefill_len; t < S; ++t) {
        Tensor x_t = x.index({Slice(), Slice(t, t+1), Slice()});
        Tensor out_t = mha.forward(x_t, &cache, t);
        decode_outs.push_back(out_t);
    }
    
    // Compare prefill outputs
    Tensor out_full_prefill = out_full.index({Slice(), Slice(0, prefill_len), Slice()});
    float prefill_diff = max_abs_diff(out_full_prefill, out_prefill);
    std::cout << "  Prefill diff: " << prefill_diff << std::endl;
    assert(prefill_diff < 1e-5f);
    
    // Compare decode outputs
    for (int i = 0; i < (int)decode_outs.size(); ++i) {
        int t = prefill_len + i;
        Tensor out_full_t = out_full.index({Slice(), Slice(t, t+1), Slice()});
        float diff = max_abs_diff(out_full_t, decode_outs[i]);
        std::cout << "  Decode pos " << t << " diff: " << diff << std::endl;
        assert(diff < 1e-4f);
    }
    
    std::cout << "RoPE positions consistency passed!" << std::endl;
}

// ============================================================================
// Multi-Layer Tests
// ============================================================================

void test_transformer_block_with_cache() {
    std::cout << "Testing TransformerBlock with KV cache..." << std::endl;
    
    int E = 32, H = 4;
    int B = 1, S = 8;
    
    nn::TransformerBlock block(E, H, 0.0f);
    block.eval();
    
    Tensor x = empty({B, S, E}, DType::Float32, Device::CPU);
    ops::uniform_(x, -0.5f, 0.5f);
    
    // Full forward
    Tensor out_full = block.forward(x, true);
    
    // Cached forward
    int head_dim = E / H;
    nn::KVCache cache(B, H, S * 2, head_dim, Device::CPU);
    
    // Process token by token
    std::vector<Tensor> cached_outputs;
    for (int t = 0; t < S; ++t) {
        Tensor x_t = x.index({Slice(), Slice(t, t+1), Slice()});
        Tensor out_t = block.forward(x_t, &cache, t);
        cached_outputs.push_back(out_t);
    }
    
    // Compare
    float max_diff = 0.0f;
    for (int t = 0; t < S; ++t) {
        Tensor out_full_t = out_full.index({Slice(), Slice(t, t+1), Slice()});
        float diff = max_abs_diff(out_full_t, cached_outputs[t]);
        max_diff = std::max(max_diff, diff);
    }
    
    std::cout << "  TransformerBlock max diff: " << max_diff << std::endl;
    assert(max_diff < 1e-4f && "TransformerBlock cached output differs");
    
    std::cout << "TransformerBlock with cache passed!" << std::endl;
}

void test_multi_layer_cache() {
    std::cout << "Testing multi-layer cache simulation..." << std::endl;
    
    int E = 32, H = 2;
    int B = 1, S = 6;
    int num_layers = 3;
    
    std::vector<nn::TransformerBlock> blocks;
    blocks.reserve(num_layers);  // Pre-reserve to avoid reallocation issues with pointer-based registration
    for (int i = 0; i < num_layers; ++i) {
        blocks.emplace_back(E, H, 0.0f);
        blocks.back().eval();
    }
    
    Tensor x = empty({B, S, E}, DType::Float32, Device::CPU);
    ops::uniform_(x, -0.5f, 0.5f);
    
    // Full forward through all layers
    Tensor h_full = x;
    for (auto& block : blocks) {
        h_full = block.forward(h_full, true);
    }
    
    // Cached forward - one cache per layer
    int head_dim = E / H;
    std::vector<nn::KVCache> caches;
    for (int i = 0; i < num_layers; ++i) {
        caches.emplace_back(B, H, S * 2, head_dim, Device::CPU);
    }
    
    // Process token by token
    std::vector<Tensor> cached_outputs;
    for (int t = 0; t < S; ++t) {
        Tensor h_t = x.index({Slice(), Slice(t, t+1), Slice()});
        
        for (int layer = 0; layer < num_layers; ++layer) {
            h_t = blocks[layer].forward(h_t, &caches[layer], t);
        }
        
        cached_outputs.push_back(h_t);
    }
    
    // Compare final outputs
    float max_diff = 0.0f;
    for (int t = 0; t < S; ++t) {
        Tensor h_full_t = h_full.index({Slice(), Slice(t, t+1), Slice()});
        float diff = max_abs_diff(h_full_t, cached_outputs[t]);
        max_diff = std::max(max_diff, diff);
    }
    
    std::cout << "  Multi-layer (" << num_layers << " layers) max diff: " << max_diff << std::endl;
    assert(max_diff < 1e-3f && "Multi-layer cached output differs");
    
    std::cout << "Multi-layer cache passed!" << std::endl;
}

// ============================================================================
// Batch Tests
// ============================================================================

void test_batch_independence_with_cache() {
    std::cout << "Testing batch independence with cache..." << std::endl;
    
    int E = 32, H = 2;
    int B = 4, S = 8;
    
    nn::MultiHeadAttention mha(E, H, 0.0f);
    mha.eval();
    
    // Create batch of different inputs
    Tensor x = empty({B, S, E}, DType::Float32, Device::CPU);
    ops::uniform_(x, -0.5f, 0.5f);
    
    // Process as batch
    int head_dim = E / H;
    nn::KVCache cache_batch(B, H, S * 2, head_dim, Device::CPU);
    Tensor out_batch = mha.forward(x, &cache_batch, 0);
    
    // Process each sample individually
    std::vector<Tensor> individual_outs;
    for (int b = 0; b < B; ++b) {
        Tensor x_b = x.index({Slice(b, b+1), Slice(), Slice()});
        nn::KVCache cache_single(1, H, S * 2, head_dim, Device::CPU);
        Tensor out_b = mha.forward(x_b, &cache_single, 0);
        individual_outs.push_back(out_b);
    }
    
    // Compare
    for (int b = 0; b < B; ++b) {
        Tensor batch_b = out_batch.index({Slice(b, b+1), Slice(), Slice()});
        float diff = max_abs_diff(batch_b, individual_outs[b]);
        
        if (diff > 1e-5f) {
            std::cerr << "  Batch item " << b << " diff: " << diff << std::endl;
        }
        assert(diff < 1e-5f && "Batch item differs from individual processing");
    }
    
    std::cout << "Batch independence with cache passed!" << std::endl;
}

// ============================================================================
// Edge Cases
// ============================================================================

void test_single_token_prompt() {
    std::cout << "Testing single token prompt..." << std::endl;
    
    int E = 32, H = 4;
    int B = 1;
    
    nn::MultiHeadAttention mha(E, H, 0.0f);
    mha.eval();
    
    // Single token input
    Tensor x = empty({B, 1, E}, DType::Float32, Device::CPU);
    ops::uniform_(x, -0.5f, 0.5f);
    
    // Full forward
    Tensor out_full = mha.forward(x, true);
    
    // Cached forward
    int head_dim = E / H;
    nn::KVCache cache(B, H, 64, head_dim, Device::CPU);
    Tensor out_cached = mha.forward(x, &cache, 0);
    
    float diff = max_abs_diff(out_full, out_cached);
    std::cout << "  Single token diff: " << diff << std::endl;
    
    assert(diff < 1e-5f && "Single token prompt differs");
    assert(cache.current_seq_len() == 1 && "Cache length should be 1");
    
    std::cout << "Single token prompt passed!" << std::endl;
}

void test_long_sequence_generation() {
    std::cout << "Testing long sequence generation..." << std::endl;
    
    int E = 64, H = 4;
    int B = 1;
    int prompt_len = 8;
    int decode_len = 64;
    
    nn::MultiHeadAttention mha(E, H, 0.0f);
    mha.eval();
    
    // Create prompt + generated tokens
    int total_len = prompt_len + decode_len;
    Tensor x_full = empty({B, total_len, E}, DType::Float32, Device::CPU);
    ops::uniform_(x_full, -0.5f, 0.5f);
    
    // Full forward for reference (just to check validity, not for comparison)
    Tensor out_full = mha.forward(x_full, true);
    assert(!contains_nan_or_inf(out_full) && "Full forward contains NaN/Inf");
    
    // Cached forward
    int head_dim = E / H;
    nn::KVCache cache(B, H, total_len * 2, head_dim, Device::CPU);
    
    // Prefill
    Tensor prompt = x_full.index({Slice(), Slice(0, prompt_len), Slice()});
    mha.forward(prompt, &cache, 0);
    
    // Decode
    for (int t = 0; t < decode_len; ++t) {
        int pos = prompt_len + t;
        Tensor x_t = x_full.index({Slice(), Slice(pos, pos+1), Slice()});
        Tensor out_t = mha.forward(x_t, &cache, pos);
        
        assert(!contains_nan_or_inf(out_t) && "Decode output contains NaN/Inf");
    }
    
    assert(cache.current_seq_len() == total_len);
    std::cout << "  Generated " << decode_len << " tokens successfully" << std::endl;
    
    std::cout << "Long sequence generation passed!" << std::endl;
}

// ============================================================================
// Performance Tests
// ============================================================================

void test_cache_performance_benefit() {
    std::cout << "Testing cache performance benefit..." << std::endl;
    
    int E = 128, H = 8;
    int B = 1, S = 64;
    
    nn::MultiHeadAttention mha(E, H, 0.0f);
    mha.eval();
    
    // Create input
    Tensor x = empty({B, S, E}, DType::Float32, Device::CPU);
    ops::uniform_(x, -0.5f, 0.5f);
    
    int head_dim = E / H;
    
    // Benchmark: recompute full forward each "step" (simulates no cache)
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < S; ++t) {
        // Each step processes all tokens up to t
        Tensor x_partial = x.index({Slice(), Slice(0, t+1), Slice()});
        mha.forward(x_partial, true);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto no_cache_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Benchmark: use cache
    nn::KVCache cache(B, H, S * 2, head_dim, Device::CPU);
    
    start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < S; ++t) {
        Tensor x_t = x.index({Slice(), Slice(t, t+1), Slice()});
        mha.forward(x_t, &cache, t);
    }
    end = std::chrono::high_resolution_clock::now();
    auto with_cache_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    float speedup = (float)no_cache_time / with_cache_time;
    
    std::cout << "  Without cache: " << no_cache_time / 1000.0f << " ms" << std::endl;
    std::cout << "  With cache: " << with_cache_time / 1000.0f << " ms" << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
    
    // Cache should provide significant speedup
    assert(speedup > 2.0f && "Cache should provide at least 2x speedup");
    
    std::cout << "Cache performance benefit passed!" << std::endl;
}

// ============================================================================
// Backend Consistency
// ============================================================================

void test_cache_cpu_vs_hip() {
    std::cout << "Testing KV cache CPU vs HIP..." << std::endl;

#ifndef USE_HIP_BACKEND
    std::cout << "  HIP not available, skipping." << std::endl;
    return;
#else
    int E = 32, H = 4;
    int B = 1, S = 8;
    
    nn::MultiHeadAttention mha_cpu(E, H, 0.0f);
    mha_cpu.eval();
    
    Tensor x_cpu = empty({B, S, E}, DType::Float32, Device::CPU);
    ops::uniform_(x_cpu, -0.5f, 0.5f);
    
    int head_dim = E / H;
    nn::KVCache cache_cpu(B, H, S * 2, head_dim, Device::CPU);
    
    // CPU cached forward
    std::vector<Tensor> cpu_outputs;
    for (int t = 0; t < S; ++t) {
        Tensor x_t = x_cpu.index({Slice(), Slice(t, t+1), Slice()});
        Tensor out_t = mha_cpu.forward(x_t, &cache_cpu, t);
        cpu_outputs.push_back(out_t);
    }
    
    // HIP version
    nn::MultiHeadAttention mha_hip(E, H, 0.0f);
    mha_hip.eval();
    
    // Copy weights to HIP
    mha_hip.c_attn.weight = mha_cpu.c_attn.weight.to(Device::HIP);
    mha_hip.c_attn.bias = mha_cpu.c_attn.bias.to(Device::HIP);
    mha_hip.c_proj.weight = mha_cpu.c_proj.weight.to(Device::HIP);
    mha_hip.c_proj.bias = mha_cpu.c_proj.bias.to(Device::HIP);
    
    Tensor x_hip = x_cpu.to(Device::HIP);
    nn::KVCache cache_hip(B, H, S * 2, head_dim, Device::HIP);
    
    // HIP cached forward
    std::vector<Tensor> hip_outputs;
    for (int t = 0; t < S; ++t) {
        Tensor x_t = x_hip.index({Slice(), Slice(t, t+1), Slice()});
        Tensor out_t = mha_hip.forward(x_t, &cache_hip, t);
        hip_outputs.push_back(out_t.to(Device::CPU));
    }
    
    // Compare
    float max_diff = 0.0f;
    for (int t = 0; t < S; ++t) {
        float diff = max_abs_diff(cpu_outputs[t], hip_outputs[t]);
        max_diff = std::max(max_diff, diff);
    }
    
    std::cout << "  CPU vs HIP max diff: " << max_diff << std::endl;
    assert(max_diff < 1e-4f && "CPU vs HIP cached output differs");
    
    std::cout << "KV cache CPU vs HIP passed!" << std::endl;
#endif
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Chapter 33.2 KV Cache Comprehensive Tests ===" << std::endl;
    
    // Correctness tests
    test_cached_vs_full_forward_mha();
    test_prefill_equivalence();
    test_incremental_decode();
    
    // Cache state tests
    test_cache_state_after_operations();
    test_cache_preserves_values();
    
    // Position encoding tests
    test_rope_positions_consistency();
    
    // Multi-layer tests
    test_transformer_block_with_cache();
    test_multi_layer_cache();
    
    // Batch tests
    test_batch_independence_with_cache();
    
    // Edge cases
    test_single_token_prompt();
    test_long_sequence_generation();
    
    // Performance tests
    test_cache_performance_benefit();
    
    // Backend consistency
    test_cache_cpu_vs_hip();
    
    std::cout << "\n=== All Chapter 33.2 KV Cache Tests Passed! ===" << std::endl;
    return 0;
}
