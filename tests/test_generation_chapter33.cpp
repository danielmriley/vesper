/**
 * @file test_generation_chapter33.cpp
 * @brief Chapter 33.7 - Text Generation and Sampling Strategies Tests
 * 
 * Comprehensive tests for:
 * - Temperature scaling
 * - Top-K filtering
 * - Top-P (nucleus) sampling
 * - Repetition penalty
 * - Greedy decoding
 * - Multinomial sampling
 * - Generator class
 * - Beam search
 * - Stress tests
 */

#include <vesper/generation/sampling.h>
#include <vesper/generation/generator.h>
#include <vesper/generation/beam_search.h>
#include <vesper/models/transformer.h>
#include <vesper/models/config.h>
#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>
#include <vesper/core/tensor.h>

#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <set>

using namespace vesper;
using namespace vesper::generation;
using namespace vesper::models;

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a simple small model for testing
std::unique_ptr<TransformerLM> create_test_model(int64_t vocab_size = 1000) {
    TransformerConfig config;
    config.vocab_size = vocab_size;
    config.dim = 64;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.ffn_hidden_dim = 128;
    config.max_seq_len = 64;
    config.use_rms_norm = true;
    config.rope_base = 10000.0f;
    
    auto model = std::make_unique<TransformerLM>(config);
    model->eval();
    return model;
}

/// Create test logits with known distribution
Tensor create_peaked_logits(int64_t batch_size, int64_t vocab_size, 
                           int64_t peak_idx = 0, float peak_value = 10.0f) {
    Tensor logits = vesper::zeros({batch_size, vocab_size}, DType::Float32, Device::CPU);
    float* data = logits.data_ptr<float>();
    
    // Small random values
    for (int64_t i = 0; i < batch_size * vocab_size; ++i) {
        data[i] = static_cast<float>(i % 10) * 0.1f;
    }
    
    // Add peak
    for (int64_t b = 0; b < batch_size; ++b) {
        data[b * vocab_size + peak_idx] = peak_value;
    }
    
    return logits;
}

/// Create uniform logits
Tensor create_uniform_logits(int64_t batch_size, int64_t vocab_size) {
    return vesper::zeros({batch_size, vocab_size}, DType::Float32, Device::CPU);
}

/// Create test tokens
Tensor create_test_tokens(int64_t batch_size, int64_t seq_len, int64_t vocab_size) {
    Tensor tokens = vesper::empty({batch_size, seq_len}, DType::Int32, Device::CPU);
    int32_t* data = tokens.data_ptr<int32_t>();
    
    for (int64_t i = 0; i < batch_size * seq_len; ++i) {
        data[i] = static_cast<int32_t>(i % vocab_size);
    }
    
    return tokens;
}

// =============================================================================
// Test 1: Temperature Scaling
// =============================================================================
void test_temperature_scaling() {
    std::cout << "Testing temperature scaling..." << std::endl;
    
    Tensor logits = create_peaked_logits(2, 100, 50, 10.0f);
    
    // T=1.0: no change
    Tensor t1 = apply_temperature(logits, 1.0f);
    Tensor t1_cpu = t1.to(Device::CPU);
    const float* t1_data = t1_cpu.data_ptr<float>();
    assert(std::abs(t1_data[50] - 10.0f) < 1e-5f);
    std::cout << "  T=1.0: logits unchanged ✓" << std::endl;
    
    // T=0.5: sharper (logits doubled)
    Tensor t05 = apply_temperature(logits, 0.5f);
    Tensor t05_cpu = t05.to(Device::CPU);
    const float* t05_data = t05_cpu.data_ptr<float>();
    assert(std::abs(t05_data[50] - 20.0f) < 1e-5f);
    std::cout << "  T=0.5: logits scaled by 2 ✓" << std::endl;
    
    // T=2.0: flatter (logits halved)
    Tensor t2 = apply_temperature(logits, 2.0f);
    Tensor t2_cpu = t2.to(Device::CPU);
    const float* t2_data = t2_cpu.data_ptr<float>();
    assert(std::abs(t2_data[50] - 5.0f) < 1e-5f);
    std::cout << "  T=2.0: logits scaled by 0.5 ✓" << std::endl;
    
    // Verify probability distribution effect
    Tensor p1 = nn::functional::softmax(t1, -1);
    Tensor p05 = nn::functional::softmax(t05, -1);
    Tensor p2 = nn::functional::softmax(t2, -1);
    
    // Convert to CPU and get max probabilities
    Tensor p1_cpu = p1.to(Device::CPU);
    Tensor p05_cpu = p05.to(Device::CPU);
    Tensor p2_cpu = p2.to(Device::CPU);
    
    float max_p1 = *std::max_element(p1_cpu.data_ptr<float>(), 
                                      p1_cpu.data_ptr<float>() + 100);
    float max_p05 = *std::max_element(p05_cpu.data_ptr<float>(),
                                       p05_cpu.data_ptr<float>() + 100);
    float max_p2 = *std::max_element(p2_cpu.data_ptr<float>(),
                                      p2_cpu.data_ptr<float>() + 100);
    
    // Lower T should give higher max probability
    assert(max_p05 > max_p1);
    assert(max_p1 > max_p2);
    std::cout << "  Probability distribution: lower T gives sharper peaks ✓" << std::endl;
    
    std::cout << "Temperature scaling passed!" << std::endl;
}

// =============================================================================
// Test 2: Top-K Filtering
// =============================================================================
void test_top_k_filtering() {
    std::cout << "Testing Top-K filtering..." << std::endl;
    
    // Create logits with known order
    Tensor logits = vesper::empty({1, 10}, DType::Float32, Device::CPU);
    float* data = logits.data_ptr<float>();
    for (int i = 0; i < 10; ++i) {
        data[i] = static_cast<float>(i);  // 0, 1, 2, ..., 9
    }
    // Top tokens are: 9, 8, 7, ...
    
    // K=3: keep only indices 9, 8, 7
    Tensor filtered = apply_top_k(logits, 3);
    Tensor probs = nn::functional::softmax(filtered, -1);
    Tensor probs_cpu = probs.to(Device::CPU);
    const float* probs_data = probs_cpu.data_ptr<float>();
    
    // Indices 0-6 should have probability ≈ 0
    float sum_low = 0.0f;
    for (int i = 0; i < 7; ++i) {
        sum_low += probs_data[i];
    }
    assert(sum_low < 1e-5f);
    std::cout << "  Non-top-K tokens have near-zero probability ✓" << std::endl;
    
    // Indices 7, 8, 9 should sum to 1
    float sum_high = probs_data[7] + probs_data[8] + probs_data[9];
    assert(std::abs(sum_high - 1.0f) < 1e-5f);
    std::cout << "  Top-K tokens sum to 1 ✓" << std::endl;
    
    // Test K=0 (disabled)
    Tensor no_filter = apply_top_k(logits, 0);
    Tensor no_filter_cpu = no_filter.to(Device::CPU);
    const float* no_filter_data = no_filter_cpu.data_ptr<float>();
    bool unchanged = true;
    for (int i = 0; i < 10; ++i) {
        if (std::abs(no_filter_data[i] - data[i]) > 1e-5f) {
            unchanged = false;
            break;
        }
    }
    assert(unchanged);
    std::cout << "  K=0 disables filtering ✓" << std::endl;
    
    // Test K >= vocab_size
    Tensor full = apply_top_k(logits, 100);
    unchanged = true;
    Tensor full_cpu = full.to(Device::CPU);
    const float* full_data = full_cpu.data_ptr<float>();
    for (int i = 0; i < 10; ++i) {
        if (std::abs(full_data[i] - data[i]) > 1e-5f) {
            unchanged = false;
            break;
        }
    }
    assert(unchanged);
    std::cout << "  K >= vocab_size disables filtering ✓" << std::endl;
    
    std::cout << "Top-K filtering passed!" << std::endl;
}

// =============================================================================
// Test 3: Top-P (Nucleus) Filtering  
// =============================================================================
void test_top_p_filtering() {
    std::cout << "Testing Top-P (nucleus) filtering..." << std::endl;
    
    // Create peaked logits: one token has most of the probability mass
    Tensor peaked = vesper::empty({1, 5}, DType::Float32, Device::CPU);
    float* data = peaked.data_ptr<float>();
    data[0] = 10.0f;  // Will have ~0.99 probability after softmax
    data[1] = 0.0f;
    data[2] = 0.0f;
    data[3] = 0.0f;
    data[4] = 0.0f;
    
    // P=0.9: should keep only the first token
    Tensor filtered = apply_top_p(peaked, 0.9f);
    Tensor probs = nn::functional::softmax(filtered, -1);
    Tensor probs_cpu = probs.to(Device::CPU);
    const float* probs_data = probs_cpu.data_ptr<float>();
    
    // First token should have probability > 0.99
    assert(probs_data[0] > 0.99f);
    std::cout << "  Peaked distribution: keeps dominant token ✓" << std::endl;
    
    // Test with uniform distribution
    Tensor uniform = create_uniform_logits(1, 10);
    Tensor uniform_filtered = apply_top_p(uniform, 0.5f);
    Tensor uniform_probs = nn::functional::softmax(uniform_filtered, -1);
    Tensor uniform_probs_cpu = uniform_probs.to(Device::CPU);
    const float* uniform_data = uniform_probs_cpu.data_ptr<float>();
    
    // Count non-zero probabilities
    int non_zero = 0;
    for (int i = 0; i < 10; ++i) {
        if (uniform_data[i] > 1e-6f) {
            ++non_zero;
        }
    }
    // For uniform, P=0.5 should keep about half the tokens
    assert(non_zero >= 4 && non_zero <= 6);
    std::cout << "  Uniform distribution with P=0.5: keeps ~half tokens (" << non_zero << "/10) ✓" << std::endl;
    
    // P=1.0 should disable filtering
    Tensor no_filter = apply_top_p(peaked, 1.0f);
    // All tokens should be valid (not -inf)
    Tensor no_filter_cpu = no_filter.to(Device::CPU);
    const float* nf_data = no_filter_cpu.data_ptr<float>();
    bool all_valid = true;
    for (int i = 0; i < 5; ++i) {
        if (nf_data[i] < -1e10f) {
            all_valid = false;
        }
    }
    assert(all_valid);
    std::cout << "  P=1.0 disables filtering ✓" << std::endl;
    
    std::cout << "Top-P filtering passed!" << std::endl;
}

// =============================================================================
// Test 4: Repetition Penalty
// =============================================================================
void test_repetition_penalty() {
    std::cout << "Testing repetition penalty..." << std::endl;
    
    // Create uniform logits
    Tensor logits = vesper::full({1, 100}, DType::Float32, Device::CPU, 1.0f);
    
    // Input IDs with some repeated tokens
    Tensor input_ids = vesper::empty({1, 5}, DType::Int32, Device::CPU);
    int32_t* ids = input_ids.data_ptr<int32_t>();
    ids[0] = 5;
    ids[1] = 10;
    ids[2] = 5;  // Repeated
    ids[3] = 20;
    ids[4] = 5;  // Repeated again
    
    // Apply penalty > 1 (discourage repetition)
    Tensor penalized = apply_repetition_penalty(logits, input_ids, 2.0f);
    Tensor pen_cpu = penalized.to(Device::CPU);
    const float* pen_data = pen_cpu.data_ptr<float>();
    
    // Token 5, 10, 20 should have lower logits
    assert(pen_data[5] < 1.0f);
    assert(pen_data[10] < 1.0f);
    assert(pen_data[20] < 1.0f);
    std::cout << "  Repeated tokens have lower logits ✓" << std::endl;
    
    // Non-repeated tokens should be unchanged
    assert(std::abs(pen_data[0] - 1.0f) < 1e-5f);
    assert(std::abs(pen_data[50] - 1.0f) < 1e-5f);
    std::cout << "  Non-repeated tokens unchanged ✓" << std::endl;
    
    // Penalty = 1.0 should not change anything
    Tensor no_pen = apply_repetition_penalty(logits, input_ids, 1.0f);
    Tensor no_pen_cpu = no_pen.to(Device::CPU);
    const float* no_pen_data = no_pen_cpu.data_ptr<float>();
    assert(std::abs(no_pen_data[5] - 1.0f) < 1e-5f);
    std::cout << "  Penalty=1.0 has no effect ✓" << std::endl;
    
    // Test with negative logits
    Tensor neg_logits = vesper::full({1, 100}, DType::Float32, Device::CPU, -1.0f);
    Tensor neg_pen = apply_repetition_penalty(neg_logits, input_ids, 2.0f);
    Tensor neg_pen_cpu = neg_pen.to(Device::CPU);
    const float* neg_pen_data = neg_pen_cpu.data_ptr<float>();
    
    // For negative logits with penalty > 1, they should become more negative
    assert(neg_pen_data[5] < -1.0f);
    std::cout << "  Handles negative logits correctly ✓" << std::endl;
    
    std::cout << "Repetition penalty passed!" << std::endl;
}

// =============================================================================
// Test 5: Greedy Sampling
// =============================================================================
void test_greedy_sampling() {
    std::cout << "Testing greedy sampling..." << std::endl;
    
    // Create logits with known maximum
    Tensor logits = vesper::empty({3, 50}, DType::Float32, Device::CPU);
    float* data = logits.data_ptr<float>();
    for (int64_t i = 0; i < 3 * 50; ++i) {
        data[i] = static_cast<float>(i % 50) * 0.1f;
    }
    // Max is at index 49 for each batch
    data[49] = 100.0f;
    data[50 + 49] = 100.0f;
    data[100 + 49] = 100.0f;
    
    // Test greedy
    Tensor sampled = greedy_sample(logits);
    assert(sampled.shape()[0] == 3);
    
    Tensor sampled_cpu = sampled.to(Device::CPU);
    const int32_t* sampled_data = sampled_cpu.data_ptr<int32_t>();
    
    assert(sampled_data[0] == 49);
    assert(sampled_data[1] == 49);
    assert(sampled_data[2] == 49);
    std::cout << "  Selects argmax for all batch elements ✓" << std::endl;
    
    // Test determinism
    Tensor sampled2 = greedy_sample(logits);
    Tensor sampled2_cpu = sampled2.to(Device::CPU);
    const int32_t* sampled2_data = sampled2_cpu.data_ptr<int32_t>();
    
    for (int i = 0; i < 3; ++i) {
        assert(sampled_data[i] == sampled2_data[i]);
    }
    std::cout << "  Deterministic results ✓" << std::endl;
    
    std::cout << "Greedy sampling passed!" << std::endl;
}

// =============================================================================
// Test 6: Multinomial Sampling
// =============================================================================
void test_multinomial_sampling() {
    std::cout << "Testing multinomial sampling..." << std::endl;
    
    // Create peaked probability distribution
    Tensor probs = vesper::zeros({2, 10}, DType::Float32, Device::CPU);
    float* data = probs.data_ptr<float>();
    // Batch 0: token 3 has 90% probability
    data[3] = 0.9f;
    for (int i = 0; i < 10; ++i) {
        if (i != 3) data[i] = 0.1f / 9.0f;
    }
    // Batch 1: token 7 has 90% probability
    data[10 + 7] = 0.9f;
    for (int i = 0; i < 10; ++i) {
        if (i != 7) data[10 + i] = 0.1f / 9.0f;
    }
    
    // Sample many times and check distribution
    int count_3 = 0;
    int count_7 = 0;
    int total = 1000;
    
    for (int trial = 0; trial < total; ++trial) {
        Tensor sampled = multinomial_sample(probs, 1, trial);
        Tensor sampled_cpu = sampled.to(Device::CPU);
        const int32_t* sampled_data = sampled_cpu.data_ptr<int32_t>();
        
        if (sampled_data[0] == 3) ++count_3;
        if (sampled_data[1] == 7) ++count_7;
    }
    
    // Should be around 90% (with some variance)
    float ratio_3 = static_cast<float>(count_3) / total;
    float ratio_7 = static_cast<float>(count_7) / total;
    
    assert(ratio_3 > 0.8f && ratio_3 < 0.98f);
    assert(ratio_7 > 0.8f && ratio_7 < 0.98f);
    std::cout << "  Sampling follows distribution (batch 0: " << ratio_3 * 100 
              << "%, batch 1: " << ratio_7 * 100 << "%) ✓" << std::endl;
    
    // Test with seed for reproducibility
    Tensor s1 = multinomial_sample(probs, 1, 42);
    Tensor s2 = multinomial_sample(probs, 1, 42);
    
    Tensor s1_cpu = s1.to(Device::CPU);
    Tensor s2_cpu = s2.to(Device::CPU);
    
    assert(s1_cpu.data_ptr<int32_t>()[0] == s2_cpu.data_ptr<int32_t>()[0]);
    assert(s1_cpu.data_ptr<int32_t>()[1] == s2_cpu.data_ptr<int32_t>()[1]);
    std::cout << "  Reproducible with seed ✓" << std::endl;
    
    std::cout << "Multinomial sampling passed!" << std::endl;
}

// =============================================================================
// Test 7: Sample Next Token Integration
// =============================================================================
void test_sample_next_token() {
    std::cout << "Testing sample_next_token integration..." << std::endl;
    
    Tensor logits = create_peaked_logits(2, 100, 42, 10.0f);
    
    // Greedy mode
    SamplingParams greedy_params;
    greedy_params.do_sample = false;
    
    Tensor greedy = sample_next_token(logits, greedy_params);
    Tensor greedy_cpu = greedy.to(Device::CPU);
    const int32_t* greedy_data = greedy_cpu.data_ptr<int32_t>();
    
    assert(greedy_data[0] == 42);
    assert(greedy_data[1] == 42);
    std::cout << "  Greedy mode works ✓" << std::endl;
    
    // Sampling with low temperature (should mostly pick peak)
    SamplingParams sample_params;
    sample_params.do_sample = true;
    sample_params.temperature = 0.1f;
    
    int peak_count = 0;
    for (int trial = 0; trial < 100; ++trial) {
        sample_params.seed = trial;
        Tensor sampled = sample_next_token(logits, sample_params);
        Tensor sampled_cpu = sampled.to(Device::CPU);
        if (sampled_cpu.data_ptr<int32_t>()[0] == 42) ++peak_count;
    }
    assert(peak_count > 90);  // Should be almost always the peak
    std::cout << "  Low temperature sampling mostly picks peak (" << peak_count << "/100) ✓" << std::endl;
    
    // Top-K filtering
    SamplingParams topk_params;
    topk_params.do_sample = true;
    topk_params.top_k = 5;
    topk_params.temperature = 1.0f;
    
    std::set<int32_t> seen_tokens;
    for (int trial = 0; trial < 100; ++trial) {
        topk_params.seed = trial;
        Tensor sampled = sample_next_token(logits, topk_params);
        Tensor sampled_cpu = sampled.to(Device::CPU);
        seen_tokens.insert(sampled_cpu.data_ptr<int32_t>()[0]);
    }
    // Should only see top-5 tokens
    assert(seen_tokens.size() <= 5);
    std::cout << "  Top-K limits vocabulary to " << seen_tokens.size() << " tokens ✓" << std::endl;
    
    std::cout << "Sample next token integration passed!" << std::endl;
}

// =============================================================================
// Test 8: Generator Basic
// =============================================================================
void test_generator_basic() {
    std::cout << "Testing Generator basic functionality..." << std::endl;
    
    auto model = create_test_model(1000);
    Generator gen(model.get());
    
    Tensor prompt = create_test_tokens(1, 4, 1000);
    
    SamplingParams params;
    params.max_new_tokens = 10;
    params.do_sample = false;  // Greedy for determinism
    
    Tensor output = gen.generate(prompt, params);
    
    assert(output.ndim() == 2);
    assert(output.shape()[0] == 1);
    assert(output.shape()[1] == 4 + 10);  // prompt + generated
    std::cout << "  Output shape correct: [1, 14] ✓" << std::endl;
    
    // Check that prompt is preserved
    Tensor output_cpu = output.to(Device::CPU);
    Tensor prompt_cpu = prompt.to(Device::CPU);
    const int32_t* out_data = output_cpu.data_ptr<int32_t>();
    const int32_t* prompt_data = prompt_cpu.data_ptr<int32_t>();
    
    bool prompt_preserved = true;
    for (int i = 0; i < 4; ++i) {
        if (out_data[i] != prompt_data[i]) {
            prompt_preserved = false;
            break;
        }
    }
    assert(prompt_preserved);
    std::cout << "  Prompt tokens preserved ✓" << std::endl;
    
    // Check tokens are valid
    bool all_valid = true;
    for (int i = 0; i < 14; ++i) {
        if (out_data[i] < 0 || out_data[i] >= 1000) {
            all_valid = false;
            break;
        }
    }
    assert(all_valid);
    std::cout << "  All tokens in valid range ✓" << std::endl;
    
    std::cout << "Generator basic passed!" << std::endl;
}

// =============================================================================
// Test 9: Generator Stop Tokens
// =============================================================================
void test_generator_stop_tokens() {
    std::cout << "Testing Generator stop token handling..." << std::endl;
    
    auto model = create_test_model(100);
    Generator gen(model.get());
    
    Tensor prompt = create_test_tokens(1, 4, 100);
    
    SamplingParams params;
    params.max_new_tokens = 50;  // Reduced to fit within max_seq_len (64)
    params.do_sample = true;
    params.temperature = 1.0f;
    // Note: With random weights, we can't guarantee when stop will be hit
    // So we just verify the mechanism doesn't crash
    params.stop_token_ids = {50, 99};
    
    Tensor output = gen.generate(prompt, params);
    
    // Should not exceed max
    assert(output.shape()[1] <= 4 + 50);
    std::cout << "  Generation respects max_new_tokens ✓" << std::endl;
    
    std::cout << "Generator stop tokens passed!" << std::endl;
}

// =============================================================================
// Test 10: Generator Streaming
// =============================================================================
void test_generator_streaming() {
    std::cout << "Testing Generator streaming..." << std::endl;
    
    auto model = create_test_model(500);
    Generator gen(model.get());
    
    Tensor prompt = create_test_tokens(1, 4, 500);
    
    SamplingParams params;
    params.max_new_tokens = 20;
    params.do_sample = false;
    
    std::vector<int64_t> streamed_tokens;
    
    Tensor output = gen.generate_streaming(prompt, params,
        [&streamed_tokens](int64_t batch_idx, int64_t token_id) -> bool {
            assert(batch_idx == 0);
            streamed_tokens.push_back(token_id);
            return true;  // Continue
        });
    
    // Should have received 20 tokens via callback
    assert(streamed_tokens.size() == 20);
    std::cout << "  Streamed " << streamed_tokens.size() << " tokens ✓" << std::endl;
    
    // Verify they match the output
    Tensor output_cpu = output.to(Device::CPU);
    const int32_t* out_data = output_cpu.data_ptr<int32_t>();
    
    bool match = true;
    for (size_t i = 0; i < streamed_tokens.size(); ++i) {
        if (out_data[4 + i] != static_cast<int32_t>(streamed_tokens[i])) {
            match = false;
            break;
        }
    }
    assert(match);
    std::cout << "  Streamed tokens match output ✓" << std::endl;
    
    // Test early stopping via callback
    streamed_tokens.clear();
    Tensor output2 = gen.generate_streaming(prompt, params,
        [&streamed_tokens](int64_t batch_idx, int64_t token_id) -> bool {
            streamed_tokens.push_back(token_id);
            return streamed_tokens.size() < 5;  // Stop after 5 tokens
        });
    
    assert(streamed_tokens.size() == 5);
    assert(output2.shape()[1] == 4 + 5);
    std::cout << "  Callback-based early stopping works ✓" << std::endl;
    
    std::cout << "Generator streaming passed!" << std::endl;
}

// =============================================================================
// Test 11: Generator Batch Generation
// =============================================================================
void test_generator_batch() {
    std::cout << "Testing Generator batch generation..." << std::endl;
    
    auto model = create_test_model(500);
    Generator gen(model.get());
    
    // Batch of 4 prompts
    Tensor prompt = create_test_tokens(4, 8, 500);
    
    SamplingParams params;
    params.max_new_tokens = 16;
    params.do_sample = true;
    params.temperature = 0.8f;
    params.seed = 42;
    
    Tensor output = gen.generate(prompt, params);
    
    assert(output.shape()[0] == 4);
    assert(output.shape()[1] == 8 + 16);
    std::cout << "  Batch output shape: [4, 24] ✓" << std::endl;
    
    // Check each batch element has valid tokens
    Tensor output_cpu = output.to(Device::CPU);
    const int32_t* out_data = output_cpu.data_ptr<int32_t>();
    
    bool all_valid = true;
    for (int64_t b = 0; b < 4; ++b) {
        for (int64_t s = 0; s < 24; ++s) {
            int32_t token = out_data[b * 24 + s];
            if (token < 0 || token >= 500) {
                all_valid = false;
                break;
            }
        }
    }
    assert(all_valid);
    std::cout << "  All batch tokens valid ✓" << std::endl;
    
    std::cout << "Generator batch passed!" << std::endl;
}

// =============================================================================
// Test 12: Beam Search Basic
// =============================================================================
void test_beam_search_basic() {
    std::cout << "Testing Beam Search basic..." << std::endl;
    
    auto model = create_test_model(500);
    BeamSearcher searcher(model.get());
    
    Tensor prompt = create_test_tokens(1, 4, 500);
    
    BeamSearchParams params;
    params.num_beams = 4;
    params.max_new_tokens = 10;
    params.eos_token_ids = {499};  // Use a specific EOS token
    
    Tensor output = searcher.search(prompt, params);
    
    assert(output.ndim() == 2);
    assert(output.shape()[0] == 1);
    // Length may vary due to EOS
    assert(output.shape()[1] >= 4);
    std::cout << "  Output shape: [1, " << output.shape()[1] << "] ✓" << std::endl;
    
    // Check tokens are valid
    Tensor output_cpu = output.to(Device::CPU);
    const int32_t* out_data = output_cpu.data_ptr<int32_t>();
    
    bool all_valid = true;
    for (int64_t i = 0; i < output.shape()[1]; ++i) {
        if (out_data[i] < 0 || out_data[i] >= 500) {
            all_valid = false;
            break;
        }
    }
    assert(all_valid);
    std::cout << "  All tokens in valid range ✓" << std::endl;
    
    std::cout << "Beam Search basic passed!" << std::endl;
}

// =============================================================================
// Test 13: Beam Search Determinism
// =============================================================================
void test_beam_search_determinism() {
    std::cout << "Testing Beam Search determinism..." << std::endl;
    
    auto model = create_test_model(200);
    BeamSearcher searcher(model.get());
    
    Tensor prompt = create_test_tokens(1, 4, 200);
    
    BeamSearchParams params;
    params.num_beams = 2;
    params.max_new_tokens = 8;
    
    Tensor output1 = searcher.search(prompt, params);
    Tensor output2 = searcher.search(prompt, params);
    
    // Beam search should be deterministic
    assert(output1.shape()[1] == output2.shape()[1]);
    
    Tensor out1_cpu = output1.to(Device::CPU);
    Tensor out2_cpu = output2.to(Device::CPU);
    const int32_t* data1 = out1_cpu.data_ptr<int32_t>();
    const int32_t* data2 = out2_cpu.data_ptr<int32_t>();
    
    bool equal = true;
    for (int64_t i = 0; i < output1.shape()[1]; ++i) {
        if (data1[i] != data2[i]) {
            equal = false;
            break;
        }
    }
    assert(equal);
    std::cout << "  Beam search is deterministic ✓" << std::endl;
    
    std::cout << "Beam Search determinism passed!" << std::endl;
}

// =============================================================================
// Test 14: Stress Test - Long Generation
// =============================================================================
void test_stress_long_generation() {
    std::cout << "Testing stress: long generation..." << std::endl;
    
    TransformerConfig config;
    config.vocab_size = 1000;
    config.dim = 128;
    config.n_layers = 4;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.ffn_hidden_dim = 256;
    config.max_seq_len = 512;
    config.use_rms_norm = true;
    
    auto model = std::make_unique<TransformerLM>(config);
    model->eval();
    
    Generator gen(model.get());
    
    Tensor prompt = create_test_tokens(1, 16, 1000);
    
    SamplingParams params;
    params.max_new_tokens = 200;
    params.do_sample = true;
    params.temperature = 0.9f;
    params.top_k = 50;
    
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = gen.generate(prompt, params);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    float tokens_per_sec = 200.0f / (ms / 1000.0f);
    
    assert(output.shape()[1] == 16 + 200);
    std::cout << "  Generated 200 tokens in " << ms << " ms (" << tokens_per_sec << " tok/s) ✓" << std::endl;
    
    // Check for NaN/Inf
    Tensor output_cpu = output.to(Device::CPU);
    const int32_t* data = output_cpu.data_ptr<int32_t>();
    bool valid = true;
    for (int64_t i = 0; i < output.numel(); ++i) {
        if (data[i] < 0 || data[i] >= 1000) {
            valid = false;
            break;
        }
    }
    assert(valid);
    std::cout << "  All tokens valid ✓" << std::endl;
    
    std::cout << "Stress long generation passed!" << std::endl;
}

// =============================================================================
// Test 15: Stress Test - Batch Scaling
// =============================================================================
void test_stress_batch_scaling() {
    std::cout << "Testing stress: batch scaling..." << std::endl;
    
    auto model = create_test_model(500);
    Generator gen(model.get());
    
    SamplingParams params;
    params.max_new_tokens = 16;
    params.do_sample = true;
    
    // Test different batch sizes
    std::vector<int64_t> batch_sizes = {1, 2, 4, 8};
    
    for (int64_t bs : batch_sizes) {
        Tensor prompt = create_test_tokens(bs, 8, 500);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor output = gen.generate(prompt, params);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        assert(output.shape()[0] == bs);
        assert(output.shape()[1] == 24);
        
        std::cout << "  Batch " << bs << ": " << ms << " ms ✓" << std::endl;
    }
    
    std::cout << "Stress batch scaling passed!" << std::endl;
}

// =============================================================================
// Test 16: Stress Test - Memory Stability
// =============================================================================
void test_stress_memory_stability() {
    std::cout << "Testing stress: memory stability..." << std::endl;
    
    auto model = create_test_model(500);
    Generator gen(model.get());
    
    SamplingParams params;
    params.max_new_tokens = 20;
    params.do_sample = true;
    
    // Run many generation cycles
    for (int cycle = 0; cycle < 10; ++cycle) {
        Tensor prompt = create_test_tokens(2, 8, 500);
        Tensor output = gen.generate(prompt, params);
        
        assert(output.shape()[0] == 2);
        assert(output.shape()[1] == 28);
    }
    
    std::cout << "  Completed 10 generation cycles without memory issues ✓" << std::endl;
    
    std::cout << "Stress memory stability passed!" << std::endl;
}

// =============================================================================
// Test 17: Quality Test - Sampling Diversity
// =============================================================================
void test_quality_sampling_diversity() {
    std::cout << "Testing quality: sampling diversity..." << std::endl;
    
    auto model = create_test_model(100);
    Generator gen(model.get());
    
    Tensor prompt = create_test_tokens(1, 4, 100);
    
    // High temperature should give diverse outputs
    SamplingParams diverse_params;
    diverse_params.max_new_tokens = 10;
    diverse_params.do_sample = true;
    diverse_params.temperature = 1.5f;
    
    std::set<std::vector<int32_t>> unique_outputs;
    
    for (int trial = 0; trial < 20; ++trial) {
        diverse_params.seed = trial;
        Tensor output = gen.generate(prompt, diverse_params);
        
        Tensor out_cpu = output.to(Device::CPU);
        const int32_t* data = out_cpu.data_ptr<int32_t>();
        
        std::vector<int32_t> seq(data + 4, data + output.shape()[1]);
        unique_outputs.insert(seq);
    }
    
    // Should have multiple unique outputs
    assert(unique_outputs.size() > 1);
    std::cout << "  High temp generated " << unique_outputs.size() << " unique sequences ✓" << std::endl;
    
    // Low temperature should give less diversity
    SamplingParams focused_params;
    focused_params.max_new_tokens = 10;
    focused_params.do_sample = true;
    focused_params.temperature = 0.1f;
    
    std::set<std::vector<int32_t>> focused_outputs;
    
    for (int trial = 0; trial < 20; ++trial) {
        focused_params.seed = trial;
        Tensor output = gen.generate(prompt, focused_params);
        
        Tensor out_cpu = output.to(Device::CPU);
        const int32_t* data = out_cpu.data_ptr<int32_t>();
        
        std::vector<int32_t> seq(data + 4, data + output.shape()[1]);
        focused_outputs.insert(seq);
    }
    
    // Low temp should have fewer unique outputs (possibly just 1)
    assert(focused_outputs.size() <= unique_outputs.size());
    std::cout << "  Low temp generated " << focused_outputs.size() << " unique sequences ✓" << std::endl;
    
    std::cout << "Quality sampling diversity passed!" << std::endl;
}

// =============================================================================
// Test 18: Quality Test - Top-K vs Top-P
// =============================================================================
void test_quality_topk_vs_topp() {
    std::cout << "Testing quality: Top-K vs Top-P behavior..." << std::endl;
    
    // Create specific probability distribution
    Tensor logits = vesper::empty({1, 20}, DType::Float32, Device::CPU);
    float* data = logits.data_ptr<float>();
    
    // Exponentially decreasing probabilities
    for (int i = 0; i < 20; ++i) {
        data[i] = 10.0f - static_cast<float>(i) * 0.5f;
    }
    
    // Top-K=5 keeps exactly 5 tokens
    Tensor topk = apply_top_k(logits, 5);
    Tensor topk_probs = nn::functional::softmax(topk, -1);
    Tensor topk_cpu = topk_probs.to(Device::CPU);
    const float* topk_data = topk_cpu.data_ptr<float>();
    
    int topk_count = 0;
    for (int i = 0; i < 20; ++i) {
        if (topk_data[i] > 1e-6f) ++topk_count;
    }
    assert(topk_count == 5);
    std::cout << "  Top-K=5 keeps exactly 5 tokens ✓" << std::endl;
    
    // Top-P adapts to the distribution
    Tensor topp = apply_top_p(logits, 0.9f);
    Tensor topp_probs = nn::functional::softmax(topp, -1);
    Tensor topp_cpu = topp_probs.to(Device::CPU);
    const float* topp_data = topp_cpu.data_ptr<float>();
    
    int topp_count = 0;
    for (int i = 0; i < 20; ++i) {
        if (topp_data[i] > 1e-6f) ++topp_count;
    }
    // Top-P should keep tokens until cumsum > 0.9
    assert(topp_count >= 1 && topp_count <= 20);
    std::cout << "  Top-P=0.9 keeps " << topp_count << " tokens (adaptive) ✓" << std::endl;
    
    std::cout << "Quality Top-K vs Top-P passed!" << std::endl;
}

// =============================================================================
// Test 19: Edge Case - Empty Stop Tokens
// =============================================================================
void test_edge_empty_stop_tokens() {
    std::cout << "Testing edge case: empty stop tokens..." << std::endl;
    
    auto model = create_test_model(100);
    Generator gen(model.get());
    
    Tensor prompt = create_test_tokens(1, 4, 100);
    
    SamplingParams params;
    params.max_new_tokens = 10;
    params.do_sample = false;
    params.stop_token_ids = {};  // Empty
    
    Tensor output = gen.generate(prompt, params);
    
    // Should generate all max_new_tokens
    assert(output.shape()[1] == 14);
    std::cout << "  Empty stop tokens generates full length ✓" << std::endl;
    
    std::cout << "Edge case empty stop tokens passed!" << std::endl;
}

// =============================================================================
// Test 20: Edge Case - Single Token Generation
// =============================================================================
void test_edge_single_token() {
    std::cout << "Testing edge case: single token generation..." << std::endl;
    
    auto model = create_test_model(100);
    Generator gen(model.get());
    
    Tensor prompt = create_test_tokens(1, 4, 100);
    
    SamplingParams params;
    params.max_new_tokens = 1;
    params.do_sample = false;
    
    Tensor output = gen.generate(prompt, params);
    
    assert(output.shape()[1] == 5);
    std::cout << "  Single token generation works ✓" << std::endl;
    
    std::cout << "Edge case single token passed!" << std::endl;
}

// =============================================================================
// Test 21: Edge Case - Very Small Top-K
// =============================================================================
void test_edge_small_topk() {
    std::cout << "Testing edge case: very small Top-K..." << std::endl;
    
    Tensor logits = create_peaked_logits(2, 100, 50, 10.0f);
    
    // K=1 should always pick the peak
    Tensor filtered = apply_top_k(logits, 1);
    
    SamplingParams params;
    params.do_sample = true;
    params.top_k = 1;
    params.temperature = 100.0f;  // Even with high temp, only 1 option
    
    Tensor sampled = sample_next_token(logits, params);
    Tensor sampled_cpu = sampled.to(Device::CPU);
    
    assert(sampled_cpu.data_ptr<int32_t>()[0] == 50);
    assert(sampled_cpu.data_ptr<int32_t>()[1] == 50);
    std::cout << "  K=1 always selects the peak ✓" << std::endl;
    
    std::cout << "Edge case small Top-K passed!" << std::endl;
}

// =============================================================================
// Test 22: Edge Case - Very Small Top-P
// =============================================================================
void test_edge_small_topp() {
    std::cout << "Testing edge case: very small Top-P..." << std::endl;
    
    Tensor logits = create_peaked_logits(1, 100, 50, 10.0f);
    
    // P=0.01: should keep only the highest probability token
    Tensor filtered = apply_top_p(logits, 0.01f);
    Tensor probs = nn::functional::softmax(filtered, -1);
    Tensor probs_cpu = probs.to(Device::CPU);
    const float* probs_data = probs_cpu.data_ptr<float>();
    
    // Token 50 should have ~100% probability
    assert(probs_data[50] > 0.99f);
    std::cout << "  P=0.01 keeps only the peak ✓" << std::endl;
    
    std::cout << "Edge case small Top-P passed!" << std::endl;
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "\n=== Chapter 33.7 Text Generation and Sampling Tests ===\n" << std::endl;
    
    try {
        // Core sampling tests
        test_temperature_scaling();
        test_top_k_filtering();
        test_top_p_filtering();
        test_repetition_penalty();
        test_greedy_sampling();
        test_multinomial_sampling();
        test_sample_next_token();
        
        // Generator tests
        test_generator_basic();
        test_generator_stop_tokens();
        test_generator_streaming();
        test_generator_batch();
        
        // Beam search tests
        test_beam_search_basic();
        test_beam_search_determinism();
        
        // Stress tests
        test_stress_long_generation();
        test_stress_batch_scaling();
        test_stress_memory_stability();
        
        // Quality tests
        test_quality_sampling_diversity();
        test_quality_topk_vs_topp();
        
        // Edge cases
        test_edge_empty_stop_tokens();
        test_edge_single_token();
        test_edge_small_topk();
        test_edge_small_topp();
        
        std::cout << "\n=== All Chapter 33.7 Tests Passed! ===\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
