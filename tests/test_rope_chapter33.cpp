/**
 * @file test_rope_chapter33.cpp
 * @brief Comprehensive tests for Chapter 33.3: Rotary Positional Embeddings (RoPE)
 * 
 * Tests cover:
 * - Frequency computation correctness
 * - Rotation preserves vector norms
 * - Relative position property (q_m · k_n depends only on m-n)
 * - Inverse rotation returns original
 * - Long context handling
 * - Batched processing performance
 * - Incremental decoding simulation
 * - NTK-aware scaling
 * - CPU vs HIP consistency
 */

#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/nn/rope.h>
#include <vesper/nn/functional.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include <vector>

#if defined(USE_HIP_BACKEND)
#include <hip/hip_runtime.h>
#endif

using namespace vesper;

constexpr float EPSILON = 1e-5f;

// Helper to check if tensor contains NaN or Inf
bool has_nan_or_inf(const Tensor& t) {
    Tensor t_cpu = t.to(Device::CPU);
    const float* ptr = t_cpu.data_ptr<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        if (std::isnan(ptr[i]) || std::isinf(ptr[i])) {
            return true;
        }
    }
    return false;
}

// Helper to compute L2 norm of a tensor
float compute_norm(const Tensor& t) {
    Tensor sq = ops::mul(t, t);
    Tensor sum = ops::sum(sq);
    Tensor sum_cpu = sum.to(Device::CPU);
    return std::sqrt(sum_cpu.data_ptr<float>()[0]);
}

// Helper to compute max absolute difference
float max_abs_diff(const Tensor& a, const Tensor& b) {
    Tensor a_cpu = a.to(Device::CPU);
    Tensor b_cpu = b.to(Device::CPU);
    const float* pa = a_cpu.data_ptr<float>();
    const float* pb = b_cpu.data_ptr<float>();
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.numel(); ++i) {
        max_diff = std::max(max_diff, std::abs(pa[i] - pb[i]));
    }
    return max_diff;
}

// Test 1: Frequency Computation
void test_frequency_computation() {
    std::cout << "Testing frequency computation..." << std::endl;
    
    nn::RoPEFrequencies rope(128, 64, 10000.0f, Device::CPU);
    
    // At position 0, all angles are 0, so cos=1, sin=0
    Tensor freqs = rope.get(0, 1);
    assert(freqs.shape()[0] == 1);
    assert(freqs.shape()[1] == 32);  // head_dim/2
    assert(freqs.shape()[2] == 2);   // [cos, sin]
    
    const float* data = freqs.data_ptr<float>();
    for (int i = 0; i < 32; ++i) {
        float cos_val = data[i * 2];
        float sin_val = data[i * 2 + 1];
        assert(std::abs(cos_val - 1.0f) < EPSILON);  // cos(0) = 1
        assert(std::abs(sin_val - 0.0f) < EPSILON);  // sin(0) = 0
    }
    
    // Check that angles increase with position
    Tensor angles_0 = rope.get_angles(0, 1);
    Tensor angles_1 = rope.get_angles(1, 1);
    
    const float* a0 = angles_0.data_ptr<float>();
    const float* a1 = angles_1.data_ptr<float>();
    
    // All angles at position 0 should be 0
    for (int i = 0; i < 32; ++i) {
        assert(std::abs(a0[i]) < EPSILON);
    }
    
    // Angles at position 1 should be positive (for freq > 0)
    assert(a1[0] > 0);  // First dimension has highest frequency
    
    // Lower dimensions should have higher frequency (larger angles)
    assert(a1[0] > a1[31]);
    
    std::cout << "Frequency computation passed!" << std::endl;
}

// Test 2: Rotation Preserves Norm
void test_rotation_preserves_norm() {
    std::cout << "Testing rotation preserves norm..." << std::endl;
    
    for (Device device : {Device::CPU, Device::HIP}) {
        std::cout << "  Testing device: " << (device == Device::CPU ? "CPU" : "HIP") << std::endl;
        
        Tensor x = randn({2, 4, 8, 64}, DType::Float32, device);  // [B, H, S, D]
        Tensor x_copy = x.clone();
        
        std::cout << "    x shape: [" << x.shape()[0] << ", " << x.shape()[1] << ", " << x.shape()[2] << ", " << x.shape()[3] << "]" << std::endl;
        
        // Create frequencies on same device
        nn::RoPEFrequencies rope(128, 64, 10000.0f, device);
        Tensor freqs = rope.get(0, 8);
        
        std::cout << "    freqs shape: [" << freqs.shape()[0] << ", " << freqs.shape()[1] << ", " << freqs.shape()[2] << "]" << std::endl;
        
        // Apply RoPE in-place
        nn::functional::apply_rope_inplace(x, freqs);
        
        // Compute norms per head position
        // Compare squared sums along head_dim
        for (int b = 0; b < 2; ++b) {
            for (int h = 0; h < 4; ++h) {
                for (int s = 0; s < 8; ++s) {
                    // Extract vectors
                    Tensor orig_vec = x_copy.index({Slice(b, b+1), Slice(h, h+1), Slice(s, s+1), Slice()});
                    Tensor rot_vec = x.index({Slice(b, b+1), Slice(h, h+1), Slice(s, s+1), Slice()});
                    
                    float orig_norm = compute_norm(orig_vec);
                    float rot_norm = compute_norm(rot_vec);
                    
                    assert(std::abs(orig_norm - rot_norm) < EPSILON * orig_norm);
                }
            }
        }
        std::cout << "  " << (device == Device::CPU ? "CPU" : "HIP") << " passed" << std::endl;
    }
    
    std::cout << "Rotation preserves norm passed!" << std::endl;
}

// Test 3: Relative Position Property
void test_relative_position_property() {
    std::cout << "Testing relative position property..." << std::endl;
    
    // The dot product q_m · k_n should depend only on (m - n)
    // Test: q at pos 5, k at pos 3 (relative = 2) 
    //       vs q at pos 10, k at pos 8 (relative = 2)
    
    nn::RoPEFrequencies rope(128, 64, 10000.0f, Device::CPU);
    
    // Create identical q and k vectors
    Tensor q = randn({1, 1, 1, 64}, DType::Float32, Device::CPU);
    Tensor k = randn({1, 1, 1, 64}, DType::Float32, Device::CPU);
    
    // Scenario 1: q at pos 5, k at pos 3
    Tensor q1 = q.clone();
    Tensor k1 = k.clone();
    Tensor freqs_5 = rope.get(5, 1);
    Tensor freqs_3 = rope.get(3, 1);
    nn::functional::apply_rope_inplace(q1, freqs_5);
    nn::functional::apply_rope_inplace(k1, freqs_3);
    
    Tensor dot1 = ops::sum(ops::mul(q1, k1));
    float dot1_val = dot1.data_ptr<float>()[0];
    
    // Scenario 2: q at pos 10, k at pos 8
    Tensor q2 = q.clone();
    Tensor k2 = k.clone();
    Tensor freqs_10 = rope.get(10, 1);
    Tensor freqs_8 = rope.get(8, 1);
    nn::functional::apply_rope_inplace(q2, freqs_10);
    nn::functional::apply_rope_inplace(k2, freqs_8);
    
    Tensor dot2 = ops::sum(ops::mul(q2, k2));
    float dot2_val = dot2.data_ptr<float>()[0];
    
    // The dot products should be equal (same relative position)
    float relative_error = std::abs(dot1_val - dot2_val) / (std::abs(dot1_val) + EPSILON);
    std::cout << "  Dot1: " << dot1_val << ", Dot2: " << dot2_val << ", Rel Error: " << relative_error << std::endl;
    assert(relative_error < 1e-4);
    
    std::cout << "Relative position property passed!" << std::endl;
}

// Test 4: Inverse Rotation
void test_inverse_rotation() {
    std::cout << "Testing inverse rotation..." << std::endl;
    
    for (Device device : {Device::CPU, Device::HIP}) {
        Tensor x = randn({1, 4, 8, 64}, DType::Float32, device);
        Tensor original = x.clone();
        
        nn::RoPEFrequencies rope(128, 64, 10000.0f, device);
        Tensor freqs = rope.get(0, 8);
        
        // Forward rotation
        nn::functional::apply_rope_inplace(x, freqs);
        
        // Inverse rotation
        nn::functional::apply_rope_inverse(x, freqs);
        
        // Should recover original
        float diff = max_abs_diff(x, original);
        std::cout << "  " << (device == Device::CPU ? "CPU" : "HIP") << " max diff: " << diff << std::endl;
        assert(diff < EPSILON);
    }
    
    std::cout << "Inverse rotation passed!" << std::endl;
}

// Test 5: Stress Test - Long Context
void test_long_context() {
    std::cout << "Testing long context handling..." << std::endl;
    
    std::vector<int64_t> seq_lengths = {1024, 4096, 16384};
    
    for (int64_t seq_len : seq_lengths) {
        nn::RoPEFrequencies rope(seq_len, 128, 10000.0f, Device::CPU);
        
        // Get frequencies at the last position
        Tensor freqs = rope.get(seq_len - 1, 1);
        
        // Should not have NaN or Inf
        assert(!has_nan_or_inf(freqs));
        
        // Cos values should be in [-1, 1]
        Tensor freqs_cpu = freqs.to(Device::CPU);
        const float* data = freqs_cpu.data_ptr<float>();
        for (int i = 0; i < 64; ++i) {
            float cos_val = data[i * 2];
            float sin_val = data[i * 2 + 1];
            assert(cos_val >= -1.0f && cos_val <= 1.0f);
            assert(sin_val >= -1.0f && sin_val <= 1.0f);
        }
        
        std::cout << "  seq_len=" << seq_len << " passed" << std::endl;
    }
    
    std::cout << "Long context handling passed!" << std::endl;
}

// Test 6: Stress Test - Batched Processing
void test_batched_processing() {
    std::cout << "Testing batched processing..." << std::endl;
    
#if defined(USE_HIP_BACKEND)
    // Large batch on GPU
    int batch = 16;
    int heads = 32;
    int seq_len = 512;
    int head_dim = 128;
    
    Tensor x = randn({batch, heads, seq_len, head_dim}, DType::Float32, Device::HIP);
    nn::RoPEFrequencies rope(seq_len * 2, head_dim, 10000.0f, Device::HIP);
    Tensor freqs = rope.get(0, seq_len);
    
    // Synchronize before timing
    hipDeviceSynchronize();
    
    auto start = std::chrono::high_resolution_clock::now();
    nn::functional::apply_rope_inplace(x, freqs);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "  RoPE on [" << batch << ", " << heads << ", " << seq_len << ", " << head_dim << "]: " << us << " us" << std::endl;
    
    // Verify no NaN
    assert(!has_nan_or_inf(x));
#else
    std::cout << "  HIP not enabled, skipping GPU batched test" << std::endl;
#endif
    
    std::cout << "Batched processing passed!" << std::endl;
}

// Test 7: Incremental Decoding
void test_incremental_decoding() {
    std::cout << "Testing incremental decoding..." << std::endl;
    
    Device device = Device::CPU;
#if defined(USE_HIP_BACKEND)
    device = Device::HIP;
#endif
    
    int64_t max_len = 512;
    nn::RoPEFrequencies rope(max_len, 64, 10000.0f, device);
    
    // Initial prompt (32 tokens)
    int prompt_len = 32;
    Tensor q = randn({1, 8, prompt_len, 64}, DType::Float32, device);
    Tensor freqs = rope.get(0, prompt_len);
    nn::functional::apply_rope_inplace(q, freqs);
    assert(!has_nan_or_inf(q));
    
    // Decode 100 more tokens one at a time
    for (int pos = prompt_len; pos < prompt_len + 100; ++pos) {
        Tensor new_q = randn({1, 8, 1, 64}, DType::Float32, device);
        Tensor new_freqs = rope.get(pos, 1);
        nn::functional::apply_rope_inplace(new_q, new_freqs);
        
        assert(!has_nan_or_inf(new_q));
    }
    
    std::cout << "Incremental decoding passed!" << std::endl;
}

// Test 8: NTK-aware Scaling
void test_ntk_scaling() {
    std::cout << "Testing NTK-aware scaling..." << std::endl;
    
    // Original training length
    int64_t original_len = 4096;
    float original_base = 10000.0f;
    
    // Extended length
    int64_t extended_len = 16384;
    
    // Compute new base
    float new_base = nn::rope_utils::compute_ntk_base(original_len, extended_len, original_base, 1.0f);
    
    // New base should be larger to spread rotations
    assert(new_base > original_base);
    
    // Expected: new_base = 10000 * (16384/4096)^1 = 10000 * 4 = 40000
    float expected = original_base * (static_cast<float>(extended_len) / original_len);
    assert(std::abs(new_base - expected) < 1.0f);
    
    std::cout << "  Original base: " << original_base << std::endl;
    std::cout << "  New base: " << new_base << " (expected " << expected << ")" << std::endl;
    
    // Test with alpha parameter
    float new_base_alpha = nn::rope_utils::compute_ntk_base(original_len, extended_len, original_base, 0.5f);
    // With alpha=0.5: new_base = 10000 * 4^0.5 = 10000 * 2 = 20000
    float expected_alpha = original_base * std::pow(4.0f, 0.5f);
    assert(std::abs(new_base_alpha - expected_alpha) < 1.0f);
    
    std::cout << "NTK-aware scaling passed!" << std::endl;
}

// Test 9: Update Base (recompute frequencies)
void test_update_base() {
    std::cout << "Testing update base..." << std::endl;
    
    nn::RoPEFrequencies rope(128, 64, 10000.0f, Device::CPU);
    
    // Get original frequencies at position 1
    Tensor freqs_orig = rope.get(1, 1).clone();
    
    // Update base to larger value
    rope.update_base(50000.0f);
    
    // Get new frequencies at position 1
    Tensor freqs_new = rope.get(1, 1);
    
    // Frequencies should be different
    float diff = max_abs_diff(freqs_orig, freqs_new);
    assert(diff > EPSILON);  // Should be different
    
    std::cout << "  Frequency diff after base update: " << diff << std::endl;
    
    std::cout << "Update base passed!" << std::endl;
}

// Test 10: Device Movement
void test_device_movement() {
    std::cout << "Testing device movement..." << std::endl;
    
#if defined(USE_HIP_BACKEND)
    // Create on CPU
    nn::RoPEFrequencies rope(128, 64, 10000.0f, Device::CPU);
    Tensor freqs_cpu = rope.get(5, 3).clone();
    
    // Move to HIP
    rope.to(Device::HIP);
    assert(rope.device() == Device::HIP);
    
    // Get frequencies (should still be same values)
    Tensor freqs_hip = rope.get(5, 3);
    
    float diff = max_abs_diff(freqs_cpu, freqs_hip.to(Device::CPU));
    assert(diff < EPSILON);
    
    std::cout << "  CPU->HIP movement: diff=" << diff << std::endl;
#else
    std::cout << "  HIP not enabled, skipping device movement test" << std::endl;
#endif
    
    std::cout << "Device movement passed!" << std::endl;
}

// Test 11: CPU vs HIP Consistency
void test_cpu_vs_hip_consistency() {
    std::cout << "Testing CPU vs HIP consistency..." << std::endl;
    
#if defined(USE_HIP_BACKEND)
    int batch = 2;
    int heads = 4;
    int seq_len = 16;
    int head_dim = 64;
    
    // Create same input on both devices
    Tensor x_cpu = randn({batch, heads, seq_len, head_dim}, DType::Float32, Device::CPU);
    Tensor x_hip = x_cpu.to(Device::HIP);
    
    // Create frequencies on both devices
    nn::RoPEFrequencies rope_cpu(128, head_dim, 10000.0f, Device::CPU);
    nn::RoPEFrequencies rope_hip(128, head_dim, 10000.0f, Device::HIP);
    
    Tensor freqs_cpu = rope_cpu.get(0, seq_len);
    Tensor freqs_hip = rope_hip.get(0, seq_len);
    
    // Apply RoPE
    nn::functional::apply_rope_inplace(x_cpu, freqs_cpu);
    nn::functional::apply_rope_inplace(x_hip, freqs_hip);
    
    // Compare
    float diff = max_abs_diff(x_cpu, x_hip.to(Device::CPU));
    std::cout << "  Max diff: " << diff << std::endl;
    assert(diff < 1e-5f);
    
    // Test inverse as well
    nn::functional::apply_rope_inverse(x_cpu, freqs_cpu);
    nn::functional::apply_rope_inverse(x_hip, freqs_hip);
    
    float diff_inv = max_abs_diff(x_cpu, x_hip.to(Device::CPU));
    std::cout << "  Max diff after inverse: " << diff_inv << std::endl;
    assert(diff_inv < 1e-5f);
#else
    std::cout << "  HIP not enabled, skipping CPU vs HIP test" << std::endl;
#endif
    
    std::cout << "CPU vs HIP consistency passed!" << std::endl;
}

// Test 12: Different Head Dimensions
void test_different_head_dims() {
    std::cout << "Testing different head dimensions..." << std::endl;
    
    std::vector<int> head_dims = {32, 64, 128, 256};
    
    for (int head_dim : head_dims) {
        nn::RoPEFrequencies rope(128, head_dim, 10000.0f, Device::CPU);
        
        Tensor x = randn({1, 4, 8, head_dim}, DType::Float32, Device::CPU);
        Tensor original = x.clone();
        
        Tensor freqs = rope.get(0, 8);
        
        // Apply and inverse
        nn::functional::apply_rope_inplace(x, freqs);
        nn::functional::apply_rope_inverse(x, freqs);
        
        float diff = max_abs_diff(x, original);
        std::cout << "  head_dim=" << head_dim << ": round-trip diff=" << diff << std::endl;
        assert(diff < EPSILON);
    }
    
    std::cout << "Different head dimensions passed!" << std::endl;
}

// Test 13: Llama3 Base (500000)
void test_llama3_base() {
    std::cout << "Testing Llama3 base (500000)..." << std::endl;
    
    float llama3_base = 500000.0f;
    nn::RoPEFrequencies rope(128, 128, llama3_base, Device::CPU);
    
    // Get frequencies
    Tensor freqs = rope.get(0, 16);
    assert(!has_nan_or_inf(freqs));
    
    // Apply to tensor
    Tensor x = randn({1, 32, 16, 128}, DType::Float32, Device::CPU);
    nn::functional::apply_rope_inplace(x, freqs);
    
    assert(!has_nan_or_inf(x));
    
    std::cout << "Llama3 base (500000) passed!" << std::endl;
}

// Test 14: Apply RoPE to Q and K together
void test_apply_rope_qk() {
    std::cout << "Testing apply_rope to Q and K together..." << std::endl;
    
    Device device = Device::CPU;
#if defined(USE_HIP_BACKEND)
    device = Device::HIP;
#endif
    
    Tensor q = randn({2, 8, 16, 64}, DType::Float32, device);
    Tensor k = randn({2, 8, 16, 64}, DType::Float32, device);
    
    Tensor q_copy = q.clone();
    Tensor k_copy = k.clone();
    
    nn::RoPEFrequencies rope(128, 64, 10000.0f, device);
    Tensor freqs = rope.get(0, 16);
    
    // Apply to both Q and K
    nn::functional::apply_rope(q, k, freqs);
    
    // Apply separately to copies
    nn::functional::apply_rope_inplace(q_copy, freqs);
    nn::functional::apply_rope_inplace(k_copy, freqs);
    
    // Should be identical
    float q_diff = max_abs_diff(q, q_copy);
    float k_diff = max_abs_diff(k, k_copy);
    
    std::cout << "  Q diff: " << q_diff << ", K diff: " << k_diff << std::endl;
    assert(q_diff < EPSILON);
    assert(k_diff < EPSILON);
    
    std::cout << "Apply RoPE to Q and K together passed!" << std::endl;
}

// Test 15: Backward Compatibility with Legacy API
void test_legacy_api_compatibility() {
    std::cout << "Testing legacy API compatibility..." << std::endl;
    
    // Test the old compute_rope_frequencies and apply_rotary_emb functions
    int seq_len = 8;
    int head_dim = 64;
    
    Tensor freqs_legacy = nn::functional::compute_rope_frequencies(seq_len, head_dim, 0, 10000.0f, Device::CPU);
    
    // Shape should be [seq_len, head_dim/2]
    assert(freqs_legacy.shape()[0] == seq_len);
    assert(freqs_legacy.shape()[1] == head_dim / 2);
    
    // Apply with legacy function
    Tensor x = randn({1, 4, seq_len, head_dim}, DType::Float32, Device::CPU);
    Tensor out = nn::functional::apply_rotary_emb(x, freqs_legacy);
    
    // Output should have same shape
    assert(out.shape() == x.shape());
    assert(!has_nan_or_inf(out));
    
    std::cout << "Legacy API compatibility passed!" << std::endl;
}

// Test 16: Position Offset
void test_position_offset() {
    std::cout << "Testing position offset..." << std::endl;
    
    nn::RoPEFrequencies rope(256, 64, 10000.0f, Device::CPU);
    
    // Get freqs for positions 0-7
    Tensor freqs_0_8 = rope.get(0, 8);
    
    // Get freqs for positions 5-12
    Tensor freqs_5_8 = rope.get(5, 8);
    
    // Position 5-7 in first should match position 0-2 in second
    // Actually, we're comparing different absolute positions, so they should differ
    
    // Instead, verify that get(start, len) returns correct slice
    Tensor freqs_5_1 = rope.get(5, 1);
    
    // This should equal freqs_0_8.slice(5, 6, 1) which gives positions [5:6]
    Tensor expected = freqs_0_8.slice(5, 6, 1);
    
    float diff = max_abs_diff(freqs_5_1, expected);
    std::cout << "  Position offset slice diff: " << diff << std::endl;
    assert(diff < EPSILON);
    
    std::cout << "Position offset passed!" << std::endl;
}

// Test 17: Edge Case - Single Position
void test_single_position() {
    std::cout << "Testing single position..." << std::endl;
    
    nn::RoPEFrequencies rope(128, 64, 10000.0f, Device::CPU);
    
    Tensor x = randn({1, 1, 1, 64}, DType::Float32, Device::CPU);
    Tensor original = x.clone();
    
    // Single position
    Tensor freqs = rope.get(42, 1);
    
    nn::functional::apply_rope_inplace(x, freqs);
    nn::functional::apply_rope_inverse(x, freqs);
    
    float diff = max_abs_diff(x, original);
    assert(diff < EPSILON);
    
    std::cout << "Single position passed!" << std::endl;
}

// Test 18: Vectorized vs Basic Kernel (different head_dims)
void test_vectorized_kernel() {
    std::cout << "Testing vectorized kernel selection..." << std::endl;
    
    // head_dim=4 should use basic kernel (not divisible by 4 for float4)
    {
        nn::RoPEFrequencies rope(128, 4, 10000.0f, Device::CPU);
        Tensor x = randn({1, 1, 8, 4}, DType::Float32, Device::CPU);
        Tensor original = x.clone();
        Tensor freqs = rope.get(0, 8);
        
        nn::functional::apply_rope_inplace(x, freqs);
        nn::functional::apply_rope_inverse(x, freqs);
        
        float diff = max_abs_diff(x, original);
        std::cout << "  head_dim=4 (basic kernel): diff=" << diff << std::endl;
        assert(diff < EPSILON);
    }
    
    // head_dim=64 should use vectorized kernel
    {
        nn::RoPEFrequencies rope(128, 64, 10000.0f, Device::CPU);
        Tensor x = randn({1, 1, 8, 64}, DType::Float32, Device::CPU);
        Tensor original = x.clone();
        Tensor freqs = rope.get(0, 8);
        
        nn::functional::apply_rope_inplace(x, freqs);
        nn::functional::apply_rope_inverse(x, freqs);
        
        float diff = max_abs_diff(x, original);
        std::cout << "  head_dim=64 (vectorized kernel): diff=" << diff << std::endl;
        assert(diff < EPSILON);
    }
    
#if defined(USE_HIP_BACKEND)
    // Test on HIP as well
    {
        nn::RoPEFrequencies rope(128, 64, 10000.0f, Device::HIP);
        Tensor x = randn({1, 1, 8, 64}, DType::Float32, Device::HIP);
        Tensor original = x.clone();
        Tensor freqs = rope.get(0, 8);
        
        nn::functional::apply_rope_inplace(x, freqs);
        nn::functional::apply_rope_inverse(x, freqs);
        
        float diff = max_abs_diff(x, original);
        std::cout << "  HIP head_dim=64: diff=" << diff << std::endl;
        assert(diff < EPSILON);
    }
#endif
    
    std::cout << "Vectorized kernel selection passed!" << std::endl;
}

// Test 19: Performance Benchmark
void test_performance_benchmark() {
    std::cout << "Testing performance benchmark..." << std::endl;
    
#if defined(USE_HIP_BACKEND)
    // Llama-style dimensions
    int batch = 1;
    int heads = 32;
    int head_dim = 128;
    std::vector<int> seq_lens = {128, 512, 2048};
    
    for (int seq_len : seq_lens) {
        Tensor x = randn({batch, heads, seq_len, head_dim}, DType::Float32, Device::HIP);
        nn::RoPEFrequencies rope(4096, head_dim, 10000.0f, Device::HIP);
        Tensor freqs = rope.get(0, seq_len);
        
        // Warmup
        nn::functional::apply_rope_inplace(x, freqs);
        hipDeviceSynchronize();
        
        // Benchmark
        int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            nn::functional::apply_rope_inplace(x, freqs);
        }
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "  seq_len=" << seq_len << ": " << us / iterations << " us/iter" << std::endl;
    }
#else
    std::cout << "  HIP not enabled, skipping performance benchmark" << std::endl;
#endif
    
    std::cout << "Performance benchmark passed!" << std::endl;
}

// Test 20: Edge Case - head_dim=2 (minimum valid)
void test_head_dim_2() {
    std::cout << "Testing minimum head_dim=2..." << std::endl;
    
    nn::RoPEFrequencies rope(128, 2, 10000.0f, Device::CPU);
    
    Tensor x = randn({1, 1, 8, 2}, DType::Float32, Device::CPU);
    Tensor original = x.clone();
    
    Tensor freqs = rope.get(0, 8);
    assert(freqs.shape()[1] == 1);  // head_dim/2 = 1
    
    nn::functional::apply_rope_inplace(x, freqs);
    assert(!has_nan_or_inf(x));
    
    // Verify norm preservation
    float orig_norm = compute_norm(original);
    float rot_norm = compute_norm(x);
    assert(std::abs(orig_norm - rot_norm) < EPSILON * orig_norm);
    
    // Round-trip
    nn::functional::apply_rope_inverse(x, freqs);
    float diff = max_abs_diff(x, original);
    std::cout << "  head_dim=2 round-trip diff: " << diff << std::endl;
    assert(diff < EPSILON);
    
    std::cout << "Minimum head_dim=2 passed!" << std::endl;
}

// Test 21: Very Large Batch (stress test memory)
void test_very_large_batch() {
    std::cout << "Testing very large batch..." << std::endl;
    
#if defined(USE_HIP_BACKEND)
    // Large batch to stress memory
    int batch = 128;
    int heads = 8;
    int seq_len = 64;
    int head_dim = 64;
    
    Tensor x = randn({batch, heads, seq_len, head_dim}, DType::Float32, Device::HIP);
    nn::RoPEFrequencies rope(256, head_dim, 10000.0f, Device::HIP);
    Tensor freqs = rope.get(0, seq_len);
    
    nn::functional::apply_rope_inplace(x, freqs);
    hipDeviceSynchronize();
    
    assert(!has_nan_or_inf(x));
    std::cout << "  Batch=" << batch << " processed successfully" << std::endl;
#else
    // CPU version with smaller batch
    int batch = 32;
    int heads = 4;
    int seq_len = 16;
    int head_dim = 64;
    
    Tensor x = randn({batch, heads, seq_len, head_dim}, DType::Float32, Device::CPU);
    nn::RoPEFrequencies rope(128, head_dim, 10000.0f, Device::CPU);
    Tensor freqs = rope.get(0, seq_len);
    
    nn::functional::apply_rope_inplace(x, freqs);
    
    assert(!has_nan_or_inf(x));
    std::cout << "  Batch=" << batch << " processed successfully (CPU)" << std::endl;
#endif
    
    std::cout << "Very large batch passed!" << std::endl;
}

// Test 22: Numerical Stability at Extreme Positions
void test_extreme_positions() {
    std::cout << "Testing extreme positions..." << std::endl;
    
    // Test very high positions where some frequencies might wrap around multiple times
    int64_t max_len = 131072;  // 128K context
    nn::RoPEFrequencies rope(max_len, 128, 10000.0f, Device::CPU);
    
    // Test positions near the end
    std::vector<int64_t> test_positions = {0, 1, 1000, max_len / 2, max_len - 10, max_len - 1};
    
    for (int64_t pos : test_positions) {
        Tensor freqs = rope.get(pos, 1);
        
        // All cos/sin values should be in [-1, 1]
        const float* data = freqs.data_ptr<float>();
        for (int i = 0; i < 64; ++i) {
            float cos_val = data[i * 2];
            float sin_val = data[i * 2 + 1];
            assert(cos_val >= -1.001f && cos_val <= 1.001f);  // Small tolerance
            assert(sin_val >= -1.001f && sin_val <= 1.001f);
        }
        
        // Verify cos^2 + sin^2 ≈ 1 for each pair
        for (int i = 0; i < 64; ++i) {
            float cos_val = data[i * 2];
            float sin_val = data[i * 2 + 1];
            float sum_sq = cos_val * cos_val + sin_val * sin_val;
            assert(std::abs(sum_sq - 1.0f) < 1e-5f);
        }
    }
    
    std::cout << "  All extreme positions passed numerical checks" << std::endl;
    std::cout << "Extreme positions passed!" << std::endl;
}

// Test 23: Frequency Decay Pattern
void test_frequency_decay_pattern() {
    std::cout << "Testing frequency decay pattern..." << std::endl;
    
    int head_dim = 64;
    nn::RoPEFrequencies rope(128, head_dim, 10000.0f, Device::CPU);
    
    // At position 1, angles should follow geometric decay
    Tensor angles = rope.get_angles(1, 1);
    const float* a = angles.data_ptr<float>();
    
    // Each successive dimension pair should have smaller angle (lower frequency)
    for (int i = 0; i < head_dim / 2 - 1; ++i) {
        // The ratio should be approximately 10000^(-2/head_dim)
        float expected_ratio = std::pow(10000.0f, -2.0f / head_dim);
        float actual_ratio = a[i + 1] / a[i];
        
        // Allow 1% tolerance
        assert(std::abs(actual_ratio - expected_ratio) < 0.01f * expected_ratio);
    }
    
    std::cout << "  Frequency decay pattern verified" << std::endl;
    std::cout << "Frequency decay pattern passed!" << std::endl;
}

// Test 24: KV Cache Integration Simulation
void test_kv_cache_integration() {
    std::cout << "Testing KV cache integration simulation..." << std::endl;
    
    Device device = Device::CPU;
#if defined(USE_HIP_BACKEND)
    device = Device::HIP;
#endif
    
    int batch = 1;
    int heads = 8;
    int head_dim = 64;
    int prompt_len = 32;
    int max_len = 256;
    
    nn::RoPEFrequencies rope(max_len, head_dim, 10000.0f, device);
    
    // Simulate KV cache: store rotated keys
    std::vector<Tensor> kv_cache;
    
    // Process prompt (all at once)
    Tensor q_prompt = randn({batch, heads, prompt_len, head_dim}, DType::Float32, device);
    Tensor k_prompt = randn({batch, heads, prompt_len, head_dim}, DType::Float32, device);
    
    Tensor freqs_prompt = rope.get(0, prompt_len);
    nn::functional::apply_rope(q_prompt, k_prompt, freqs_prompt);
    
    // Store in "cache"
    kv_cache.push_back(k_prompt.clone());
    
    // Generate 10 new tokens
    int current_pos = prompt_len;
    for (int i = 0; i < 10; ++i) {
        // New Q and K for single token
        Tensor q_new = randn({batch, heads, 1, head_dim}, DType::Float32, device);
        Tensor k_new = randn({batch, heads, 1, head_dim}, DType::Float32, device);
        
        // Apply RoPE at current position
        Tensor freqs_new = rope.get(current_pos, 1);
        nn::functional::apply_rope(q_new, k_new, freqs_new);
        
        // Verify Q and K are properly rotated (no NaN)
        assert(!has_nan_or_inf(q_new));
        assert(!has_nan_or_inf(k_new));
        
        // Add to cache
        kv_cache.push_back(k_new.clone());
        current_pos++;
    }
    
    std::cout << "  Simulated " << kv_cache.size() << " cached K tensors" << std::endl;
    std::cout << "KV cache integration simulation passed!" << std::endl;
}

// Test 25: Attention Score Symmetry
void test_attention_score_symmetry() {
    std::cout << "Testing attention score with RoPE..." << std::endl;
    
    // After RoPE, attention(q_i, k_j) should equal attention(q_j, k_i) 
    // when q and k come from same distribution
    
    nn::RoPEFrequencies rope(128, 64, 10000.0f, Device::CPU);
    
    // Create q and k at different positions
    Tensor q = randn({1, 1, 1, 64}, DType::Float32, Device::CPU);
    Tensor k = q.clone();  // Same content
    
    Tensor q1 = q.clone();
    Tensor k1 = k.clone();
    
    // Apply RoPE: q at pos 5, k at pos 3
    Tensor freqs_5 = rope.get(5, 1);
    Tensor freqs_3 = rope.get(3, 1);
    nn::functional::apply_rope_inplace(q1, freqs_5);
    nn::functional::apply_rope_inplace(k1, freqs_3);
    
    // Attention score q_5 · k_3
    Tensor score1 = ops::sum(ops::mul(q1, k1));
    
    // Now swap: q at pos 3, k at pos 5
    Tensor q2 = q.clone();
    Tensor k2 = k.clone();
    nn::functional::apply_rope_inplace(q2, freqs_3);
    nn::functional::apply_rope_inplace(k2, freqs_5);
    
    // Attention score q_3 · k_5
    Tensor score2 = ops::sum(ops::mul(q2, k2));
    
    // Since we swapped both q<->k and positions, scores should be equal
    float s1 = score1.to(Device::CPU).data_ptr<float>()[0];
    float s2 = score2.to(Device::CPU).data_ptr<float>()[0];
    
    std::cout << "  Score(q@5, k@3): " << s1 << std::endl;
    std::cout << "  Score(q@3, k@5): " << s2 << std::endl;
    
    float rel_diff = std::abs(s1 - s2) / (std::abs(s1) + EPSILON);
    std::cout << "  Relative diff: " << rel_diff << std::endl;
    assert(rel_diff < 1e-5f);
    
    std::cout << "Attention score symmetry passed!" << std::endl;
}

int main() {
    std::cout << "=== Chapter 33.3 RoPE Comprehensive Tests ===" << std::endl;
    
    // Core functionality tests
    test_frequency_computation();
    test_rotation_preserves_norm();
    test_relative_position_property();
    test_inverse_rotation();
    
    // Stress tests
    test_long_context();
    test_batched_processing();
    test_incremental_decoding();
    
    // NTK and configuration tests
    test_ntk_scaling();
    test_update_base();
    test_device_movement();
    
    // Backend consistency
    test_cpu_vs_hip_consistency();
    
    // Different configurations
    test_different_head_dims();
    test_llama3_base();
    
    // API tests
    test_apply_rope_qk();
    test_legacy_api_compatibility();
    test_position_offset();
    test_single_position();
    test_vectorized_kernel();
    
    // Performance
    test_performance_benchmark();
    
    // Additional validation tests
    test_head_dim_2();
    test_very_large_batch();
    test_extreme_positions();
    test_frequency_decay_pattern();
    test_kv_cache_integration();
    test_attention_score_symmetry();
    
    std::cout << "\n=== All Chapter 33.3 RoPE Tests Passed! ===" << std::endl;
    return 0;
}
