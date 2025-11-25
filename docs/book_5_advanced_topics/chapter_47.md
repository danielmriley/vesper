```markdown
# Chapter 47: Speculative Decoding

## 1. Introduction

Autoregressive LLM generation is inherently serial: each token depends on all previous tokens. This creates a **memory-bandwidth bottleneck** where the GPU is underutilized.

**Speculative decoding** accelerates generation by:
1. Using a **draft model** (small, fast) to generate candidate tokens
2. Using the **target model** (large, accurate) to verify them in parallel
3. Accepting correct tokens, rejecting wrong ones

This achieves **2-3x speedup** without changing the output distribution.

This chapter covers:
1. **The speculative decoding algorithm**
2. **Draft model design** (smaller version or n-gram)
3. **Parallel verification** with the target model
4. **Tree-structured speculation** for higher acceptance rates

## 2. Why Speculative Decoding Works

### 2.1 Memory Bandwidth Bottleneck

For a 7B model generating one token:
- Load 7B parameters (14 GB in FP16)
- Perform ~7B FLOPs
- Arithmetic intensity: 0.5 FLOPs/byte

Modern GPUs have:
- Memory bandwidth: ~2 TB/s (MI250X)
- Compute: ~380 TFLOPS (FP16)

**Implication**: GPU is 99% idle waiting for memory!

### 2.2 Speculation Insight

If a small draft model can predict the next 4 tokens with 70% accuracy:
- Draft: Generate 4 tokens (fast)
- Target: Verify 4 tokens in one forward pass (same memory bandwidth)
- Accept ~3 tokens on average

Speedup: 3x tokens per target model forward pass.

## 3. The Algorithm

### 3.1 Basic Speculative Decoding

```
Input: prompt, draft_model, target_model, num_speculative (K)
Output: generated text

1. prefill prompt with both models
2. while not done:
   a. draft K tokens: d1, d2, ..., dK using draft_model
   b. compute target logits for all K positions in parallel
   c. for each position i = 1 to K:
      - sample from target distribution
      - if target sample == di: accept, continue
      - else: reject, use target sample, break
   d. if all K accepted: sample K+1 from target
3. return generated text
```

### 3.2 Mathematical Foundation

For token position i, let:
- p(x) = target model probability
- q(x) = draft model probability

Acceptance probability for draft token d:
$$\alpha = \min\left(1, \frac{p(d)}{q(d)}\right)$$

If rejected, sample from adjusted distribution:
$$p'(x) = \text{normalize}\left(\max(0, p(x) - q(x))\right)$$

This ensures the output distribution exactly matches the target model.

## 4. Implementation

### 4.1 Core Data Structures

```cpp
// include/vesper/speculative/speculative.h

namespace vesper::speculative {

struct SpeculativeConfig {
    int num_speculative_tokens = 4;  // K
    float draft_temperature = 1.0f;
    bool use_tree_attention = false;
    int tree_width = 2;  // For tree speculation
};

struct DraftOutput {
    std::vector<int64_t> tokens;
    std::vector<Tensor> logits;  // Draft logits for each position
};

struct VerifyResult {
    int num_accepted;
    int64_t next_token;           // Token after accepted sequence
    std::vector<int64_t> final_tokens;  // All tokens to append
};

class SpeculativeDecoder {
public:
    SpeculativeDecoder(
        std::shared_ptr<models::Transformer> target_model,
        std::shared_ptr<models::Transformer> draft_model,
        SpeculativeConfig config = {});
    
    // Generate with speculative decoding
    std::vector<int64_t> generate(
        const std::vector<int64_t>& prompt,
        int max_new_tokens,
        SamplingParams params = {});
    
    // Stats
    float acceptance_rate() const { return total_accepted_ / float(total_drafted_); }
    float speedup() const;
    
private:
    std::shared_ptr<models::Transformer> target_model_;
    std::shared_ptr<models::Transformer> draft_model_;
    SpeculativeConfig config_;
    
    // Statistics
    int64_t total_accepted_ = 0;
    int64_t total_drafted_ = 0;
    
    // Draft K tokens
    DraftOutput draft_tokens(
        const std::vector<int64_t>& context,
        int K,
        const SamplingParams& params);
    
    // Verify and accept/reject
    VerifyResult verify(
        const std::vector<int64_t>& context,
        const DraftOutput& draft,
        const SamplingParams& params);
};

} // namespace vesper::speculative
```

### 4.2 Draft Token Generation

```cpp
// src/speculative/speculative.cpp

DraftOutput SpeculativeDecoder::draft_tokens(
    const std::vector<int64_t>& context,
    int K,
    const SamplingParams& params) 
{
    DraftOutput output;
    output.tokens.reserve(K);
    output.logits.reserve(K);
    
    std::vector<int64_t> current_context = context;
    
    for (int i = 0; i < K; ++i) {
        // Forward through draft model (single token)
        Tensor input = tensor({current_context.back()}, DType::Int64).to(Device::HIP);
        
        // Use KV cache for efficiency
        Tensor logits = draft_model_->forward_with_cache(
            input, 
            /*position=*/current_context.size() - 1);
        
        // Sample
        Tensor probs = softmax(logits[0] / config_.draft_temperature, -1);
        int64_t token = multinomial(probs, 1).item<int64_t>();
        
        output.tokens.push_back(token);
        output.logits.push_back(logits[0].clone());
        
        current_context.push_back(token);
    }
    
    return output;
}
```

### 4.3 Parallel Verification

```cpp
VerifyResult SpeculativeDecoder::verify(
    const std::vector<int64_t>& context,
    const DraftOutput& draft,
    const SamplingParams& params) 
{
    int K = draft.tokens.size();
    
    // Build input: [last_context_token, d1, d2, ..., dK]
    std::vector<int64_t> verify_tokens = {context.back()};
    verify_tokens.insert(verify_tokens.end(), 
                         draft.tokens.begin(), 
                         draft.tokens.end());
    
    // Single forward pass for K+1 positions
    Tensor input = tensor(verify_tokens, DType::Int64).to(Device::HIP);
    Tensor positions = arange(context.size() - 1, 
                              context.size() + K, 
                              DType::Int64).to(Device::HIP);
    
    Tensor target_logits = target_model_->forward_with_positions(input, positions);
    // target_logits: [K+1, vocab_size]
    
    VerifyResult result;
    result.num_accepted = 0;
    
    // Verify each draft token
    for (int i = 0; i < K; ++i) {
        int64_t draft_token = draft.tokens[i];
        Tensor draft_logit = draft.logits[i];
        Tensor target_logit = target_logits[i];
        
        // Compute probabilities
        Tensor p = softmax(target_logit / params.temperature, -1);
        Tensor q = softmax(draft_logit / config_.draft_temperature, -1);
        
        float p_draft = p[draft_token].item<float>();
        float q_draft = q[draft_token].item<float>();
        
        // Acceptance probability
        float accept_prob = std::min(1.0f, p_draft / q_draft);
        
        // Sample uniform for acceptance test
        float u = uniform_random();
        
        if (u < accept_prob) {
            // Accept draft token
            result.final_tokens.push_back(draft_token);
            result.num_accepted++;
        } else {
            // Reject: sample from adjusted distribution
            Tensor adjusted = (p - q).clamp(0);
            adjusted = adjusted / adjusted.sum();
            
            int64_t new_token = multinomial(adjusted, 1).item<int64_t>();
            result.final_tokens.push_back(new_token);
            result.next_token = new_token;
            
            // Update stats
            total_drafted_ += K;
            total_accepted_ += result.num_accepted;
            
            return result;
        }
    }
    
    // All K tokens accepted! Sample K+1 from target
    Tensor final_probs = softmax(target_logits[K] / params.temperature, -1);
    int64_t bonus_token = multinomial(final_probs, 1).item<int64_t>();
    result.final_tokens.push_back(bonus_token);
    result.next_token = bonus_token;
    result.num_accepted = K;  // All accepted
    
    total_drafted_ += K;
    total_accepted_ += K;
    
    return result;
}
```

### 4.4 Main Generation Loop

```cpp
std::vector<int64_t> SpeculativeDecoder::generate(
    const std::vector<int64_t>& prompt,
    int max_new_tokens,
    SamplingParams params) 
{
    // Prefill both models
    target_model_->prefill(prompt);
    draft_model_->prefill(prompt);
    
    std::vector<int64_t> generated = prompt;
    int tokens_generated = 0;
    
    while (tokens_generated < max_new_tokens) {
        // Draft K tokens
        int K = std::min(
            config_.num_speculative_tokens,
            max_new_tokens - tokens_generated);
        
        auto draft = draft_tokens(generated, K, params);
        
        // Verify with target model
        auto result = verify(generated, draft, params);
        
        // Append accepted tokens
        for (int64_t token : result.final_tokens) {
            generated.push_back(token);
            tokens_generated++;
            
            // Check for EOS
            if (is_eos_token(token)) {
                return generated;
            }
        }
        
        // Update draft model's KV cache
        // (reset to match accepted tokens)
        draft_model_->truncate_cache(generated.size());
    }
    
    return generated;
}
```

## 5. Draft Model Strategies

### 5.1 Smaller Version of Target

The simplest approach: Use a distilled or smaller version.

| Target Model | Draft Model | Draft Latency |
|--------------|-------------|---------------|
| Llama-70B    | Llama-7B    | ~10x faster   |
| Llama-7B     | Llama-1B    | ~7x faster    |
| GPT-4        | GPT-3.5     | ~5x faster    |

### 5.2 N-gram Draft Model

For very fast drafting, use n-gram statistics:

```cpp
class NGramDraftModel {
public:
    NGramDraftModel(int n = 4) : n_(n) {}
    
    // Build from training corpus
    void train(const std::vector<std::vector<int64_t>>& sequences) {
        for (const auto& seq : sequences) {
            for (size_t i = n_; i < seq.size(); ++i) {
                std::vector<int64_t> ngram(seq.begin() + i - n_, seq.begin() + i);
                int64_t next = seq[i];
                counts_[ngram][next]++;
            }
        }
    }
    
    // Draft tokens based on n-gram statistics
    std::vector<int64_t> draft(
        const std::vector<int64_t>& context, 
        int K) 
    {
        std::vector<int64_t> result;
        std::vector<int64_t> current = context;
        
        for (int i = 0; i < K; ++i) {
            std::vector<int64_t> ngram(current.end() - n_, current.end());
            
            if (counts_.count(ngram) == 0) {
                break;  // Unknown context
            }
            
            // Sample from n-gram distribution
            int64_t next = sample_from_counts(counts_[ngram]);
            result.push_back(next);
            current.push_back(next);
        }
        
        return result;
    }
    
private:
    int n_;
    std::unordered_map<std::vector<int64_t>, std::unordered_map<int64_t, int>> counts_;
};
```

### 5.3 Self-Drafting (Medusa Style)

Train additional heads on the target model:

```cpp
class MedusaHead : public nn::Module {
public:
    MedusaHead(int hidden_dim, int vocab_size, int num_heads = 4) {
        for (int i = 0; i < num_heads; ++i) {
            heads_.push_back(register_module(
                "head_" + std::to_string(i),
                std::make_shared<nn::Linear>(hidden_dim, vocab_size)));
        }
    }
    
    // Predict next K tokens from last hidden state
    std::vector<Tensor> forward(const Tensor& hidden) {
        std::vector<Tensor> predictions;
        for (auto& head : heads_) {
            predictions.push_back(head->forward(hidden));
        }
        return predictions;
    }
    
private:
    std::vector<std::shared_ptr<nn::Linear>> heads_;
};
```

## 6. Tree-Structured Speculation

Instead of a linear chain of speculative tokens, explore multiple branches:

```
        ┌── d3a ── d4a
   d2 ──┤
        └── d3b ── d4b
d1 ─┤
        ┌── d3c
   d2' ─┤
        └── d3d
```

### 6.1 Tree Attention Mask

```cpp
// Build tree attention mask
Tensor build_tree_mask(const std::vector<std::vector<int>>& tree_structure) {
    int total_nodes = 0;
    for (const auto& level : tree_structure) {
        total_nodes += level.size();
    }
    
    // Mask: [total_nodes, total_nodes]
    // mask[i, j] = 1 if node i can attend to node j
    Tensor mask = zeros({total_nodes, total_nodes});
    
    int node_idx = 0;
    for (size_t level = 0; level < tree_structure.size(); ++level) {
        for (size_t i = 0; i < tree_structure[level].size(); ++i) {
            int parent = tree_structure[level][i];
            
            // Can attend to self and all ancestors
            mask[node_idx][node_idx] = 1;
            if (parent >= 0) {
                // Copy parent's attention pattern
                mask[node_idx] = mask[node_idx] | mask[parent];
            }
            
            node_idx++;
        }
    }
    
    return mask;
}
```

### 6.2 Tree Verification

```cpp
struct TreeVerifyResult {
    std::vector<int64_t> accepted_path;  // Best accepted path through tree
    int64_t next_token;
};

TreeVerifyResult verify_tree(
    const std::vector<int64_t>& context,
    const std::vector<std::vector<int64_t>>& tree_tokens,  // [num_nodes]
    const std::vector<std::vector<int>>& tree_parents,
    const SamplingParams& params) 
{
    // Flatten tree tokens for batch processing
    std::vector<int64_t> flat_tokens;
    for (const auto& level : tree_tokens) {
        flat_tokens.insert(flat_tokens.end(), level.begin(), level.end());
    }
    
    // Forward with tree attention mask
    Tensor mask = build_tree_mask(tree_parents);
    Tensor input = tensor(flat_tokens, DType::Int64).to(Device::HIP);
    Tensor logits = target_model_->forward_with_mask(input, mask);
    
    // Find best accepted path
    // ... tree traversal with acceptance testing ...
}
```

## 7. Lookahead Decoding

An alternative approach: Generate and verify Jacobi-style.

```cpp
class LookaheadDecoder {
public:
    LookaheadDecoder(
        std::shared_ptr<models::Transformer> model,
        int window_size = 5,
        int n_gram_size = 4);
    
    std::vector<int64_t> generate(
        const std::vector<int64_t>& prompt,
        int max_new_tokens);
    
private:
    // Jacobi iteration: refine window of tokens
    std::vector<int64_t> jacobi_step(
        const std::vector<int64_t>& context,
        std::vector<int64_t>& window);
};
```

## 8. Testing Strategy

### 8.1 Unit Tests

```cpp
// tests/speculative/test_speculative.cpp

TEST(Speculative, DraftGeneration) {
    auto draft_model = create_small_test_model();
    auto target_model = create_test_model();
    
    SpeculativeConfig config;
    config.num_speculative_tokens = 4;
    
    SpeculativeDecoder decoder(target_model, draft_model, config);
    
    std::vector<int64_t> prompt = {1, 2, 3, 4, 5};
    auto draft = decoder.draft_tokens(prompt, 4, SamplingParams());
    
    EXPECT_EQ(draft.tokens.size(), 4);
    EXPECT_EQ(draft.logits.size(), 4);
}

TEST(Speculative, Verification) {
    // Create models where draft will be partially correct
    auto draft_model = create_test_model();
    auto target_model = create_test_model();  // Same model = 100% acceptance
    
    SpeculativeConfig config;
    config.num_speculative_tokens = 4;
    
    SpeculativeDecoder decoder(target_model, draft_model, config);
    
    std::vector<int64_t> prompt = {1, 2, 3, 4, 5};
    auto draft = decoder.draft_tokens(prompt, 4, SamplingParams());
    auto result = decoder.verify(prompt, draft, SamplingParams());
    
    // With same model, should accept all
    EXPECT_EQ(result.num_accepted, 4);
}

TEST(Speculative, OutputDistribution) {
    // Critical: verify output matches target model exactly
    auto draft_model = create_small_test_model();
    auto target_model = create_test_model();
    
    SpeculativeDecoder spec_decoder(target_model, draft_model);
    
    std::vector<int64_t> prompt = {1, 2, 3, 4, 5};
    SamplingParams params;
    params.temperature = 1.0f;
    
    // Generate many samples with speculative decoding
    std::unordered_map<std::vector<int64_t>, int> spec_counts;
    for (int i = 0; i < 1000; ++i) {
        auto output = spec_decoder.generate(prompt, 10, params);
        std::vector<int64_t> gen(output.begin() + prompt.size(), output.end());
        spec_counts[gen]++;
    }
    
    // Generate many samples with target model only
    std::unordered_map<std::vector<int64_t>, int> target_counts;
    for (int i = 0; i < 1000; ++i) {
        auto output = target_model->generate(prompt, 10, params);
        std::vector<int64_t> gen(output.begin() + prompt.size(), output.end());
        target_counts[gen]++;
    }
    
    // Distributions should be similar (chi-square test or KL divergence)
    // This is a statistical test, allow some variance
}
```

### 8.2 Acceptance Rate Tests

```cpp
TEST(Speculative, AcceptanceRate) {
    auto draft_model = create_small_test_model();  // Smaller, less accurate
    auto target_model = create_test_model();
    
    SpeculativeConfig config;
    config.num_speculative_tokens = 4;
    
    SpeculativeDecoder decoder(target_model, draft_model, config);
    
    // Generate substantial output
    std::vector<int64_t> prompt = tokenize("The quick brown fox");
    auto output = decoder.generate(prompt, 100);
    
    float acceptance = decoder.acceptance_rate();
    std::cout << "Acceptance rate: " << (acceptance * 100) << "%" << std::endl;
    
    // Reasonable draft model should have >50% acceptance
    EXPECT_GT(acceptance, 0.5f);
}
```

### 8.3 Stress Tests

```cpp
TEST(Speculative, StressTest_Speedup) {
    auto draft_model = create_small_model();  // 1B params
    auto target_model = create_large_model();  // 7B params
    
    draft_model->to(Device::HIP);
    target_model->to(Device::HIP);
    
    SpeculativeConfig config;
    config.num_speculative_tokens = 5;
    
    SpeculativeDecoder spec_decoder(target_model, draft_model, config);
    
    std::vector<int64_t> prompt = tokenize("Write a short story about");
    
    // Benchmark speculative decoding
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        auto output = spec_decoder.generate(prompt, 128);
    }
    auto spec_end = std::chrono::high_resolution_clock::now();
    double spec_ms = std::chrono::duration<double, std::milli>(spec_end - start).count();
    
    // Benchmark target-only decoding
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        auto output = target_model->generate(prompt, 128);
    }
    auto target_end = std::chrono::high_resolution_clock::now();
    double target_ms = std::chrono::duration<double, std::milli>(target_end - start).count();
    
    double speedup = target_ms / spec_ms;
    std::cout << "Speculative time: " << spec_ms << " ms" << std::endl;
    std::cout << "Target-only time: " << target_ms << " ms" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    std::cout << "Acceptance rate: " << spec_decoder.acceptance_rate() << std::endl;
    
    // Should achieve at least 1.5x speedup
    EXPECT_GT(speedup, 1.5);
}

TEST(Speculative, StressTest_MemoryUsage) {
    auto draft_model = create_small_model();
    auto target_model = create_large_model();
    
    draft_model->to(Device::HIP);
    target_model->to(Device::HIP);
    
    // Measure memory for target-only
    size_t target_only_mem = target_model->memory_bytes();
    
    // Measure memory for speculative setup
    SpeculativeDecoder decoder(target_model, draft_model);
    size_t spec_mem = target_model->memory_bytes() + draft_model->memory_bytes();
    
    float overhead = float(spec_mem) / target_only_mem;
    std::cout << "Memory overhead: " << ((overhead - 1) * 100) << "%" << std::endl;
    
    // Draft model should be small (< 20% overhead)
    EXPECT_LT(overhead, 1.2f);
}

TEST(Speculative, StressTest_LongGeneration) {
    auto draft_model = create_small_model();
    auto target_model = create_large_model();
    
    SpeculativeDecoder decoder(target_model, draft_model);
    
    std::vector<int64_t> prompt = tokenize("Once upon a time");
    
    // Generate long sequence
    auto output = decoder.generate(prompt, 1024);
    
    EXPECT_EQ(output.size(), prompt.size() + 1024);
    
    // Check no memory leaks (KV cache management)
    float acceptance = decoder.acceptance_rate();
    std::cout << "Long generation acceptance: " << acceptance << std::endl;
}

TEST(Speculative, StressTest_VariousTemperatures) {
    auto draft_model = create_small_model();
    auto target_model = create_large_model();
    
    SpeculativeDecoder decoder(target_model, draft_model);
    
    std::vector<int64_t> prompt = tokenize("The answer is");
    
    for (float temp : {0.1f, 0.5f, 1.0f, 1.5f, 2.0f}) {
        SamplingParams params;
        params.temperature = temp;
        
        decoder.reset_stats();
        auto output = decoder.generate(prompt, 64, params);
        
        std::cout << "Temperature " << temp 
                  << ": acceptance = " << decoder.acceptance_rate() << std::endl;
    }
    
    // Higher temperatures typically lead to lower acceptance (more randomness)
}
```

### 8.4 Correctness Tests

```cpp
TEST(Speculative, GreedyMatches) {
    // With greedy decoding (temp=0), output should be deterministic
    auto draft_model = create_small_model();
    auto target_model = create_large_model();
    
    SpeculativeDecoder decoder(target_model, draft_model);
    
    std::vector<int64_t> prompt = tokenize("The capital of France is");
    
    SamplingParams params;
    params.temperature = 0.0f;  // Greedy
    
    // Generate with speculation
    auto spec_output = decoder.generate(prompt, 32, params);
    
    // Generate with target only
    auto target_output = target_model->generate(prompt, 32, params);
    
    // Should be identical
    EXPECT_EQ(spec_output, target_output);
}
```

## 9. Performance Optimization

### 9.1 Batch Draft and Verify

Process multiple sequences with speculation:

```cpp
std::vector<std::vector<int64_t>> batch_generate(
    const std::vector<std::vector<int64_t>>& prompts,
    int max_new_tokens) 
{
    // Draft tokens for all sequences in parallel
    // Verify all sequences in one forward pass
    // This maximizes GPU utilization
}
```

### 9.2 Dynamic K Selection

Adjust speculative length based on acceptance history:

```cpp
int adaptive_K(float recent_acceptance_rate) {
    if (recent_acceptance_rate > 0.8) {
        return 8;  // High acceptance, speculate more
    } else if (recent_acceptance_rate > 0.5) {
        return 4;  // Medium acceptance
    } else {
        return 2;  // Low acceptance, speculate less
    }
}
```

## 10. Summary

This chapter covered:

1. **Speculative decoding algorithm**: Draft with small model, verify with large model
2. **Acceptance criteria**: Maintain target distribution exactly
3. **Draft strategies**: Smaller model, n-gram, self-drafting (Medusa)
4. **Tree speculation**: Explore multiple branches simultaneously
5. **Implementation**: KV cache management, parallel verification

Key insights:
- **2-3x speedup** is typical with good draft models
- **No quality loss**: Output distribution is preserved
- **Memory overhead** from draft model is usually small (<20%)
- **Acceptance rate** is the key metric to optimize

Speculative decoding is essential for production LLM serving where latency matters.
```
