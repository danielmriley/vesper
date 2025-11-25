```markdown
# Chapter 33.7: Text Generation and Sampling Strategies

## 1. Introduction

A language model produces a probability distribution over the vocabulary for the next token. **Sampling** is the process of selecting a token from this distribution. The choice of sampling strategy dramatically affects the quality, diversity, and coherence of generated text.

### Why Sampling Matters

| Strategy | Determinism | Quality | Diversity | Use Case |
|----------|-------------|---------|-----------|----------|
| Greedy | 100% | High | None | Factual Q&A |
| Temperature | Low | Variable | Variable | General |
| Top-K | Medium | High | Medium | Creative |
| Top-P (Nucleus) | Medium | High | High | Open-ended |
| Beam Search | High | Very High | Low | Translation |

## 2. Mathematical Foundation

### 2.1 Logits to Probabilities

The model outputs **logits** $z \in \mathbb{R}^V$ (raw scores). We convert to probabilities using **softmax**:

$$
P(token_i) = \frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}}
$$

### 2.2 Temperature Scaling

Temperature $T$ controls the "sharpness" of the distribution:

$$
P(token_i | T) = \frac{e^{z_i / T}}{\sum_{j=1}^V e^{z_j / T}}
$$

- $T \to 0$: Approaches greedy (always pick max)
- $T = 1$: Standard softmax
- $T > 1$: Flatter distribution (more random)

### 2.3 Top-K Sampling

Only consider the $K$ tokens with highest probability:

1. Sort tokens by probability
2. Keep top $K$
3. Re-normalize probabilities
4. Sample from the truncated distribution

### 2.4 Top-P (Nucleus) Sampling

Keep the smallest set of tokens whose cumulative probability exceeds $P$:

1. Sort tokens by probability (descending)
2. Compute cumulative sum
3. Keep tokens until cumsum > $P$
4. Re-normalize and sample

**Advantage**: Dynamically adjusts the candidate set size based on the distribution's entropy.

### 2.5 Beam Search

Maintain $B$ candidate sequences (beams) and expand each:

1. For each beam, compute next token probabilities
2. Keep top $B$ total candidates (across all beams)
3. Repeat until EOS or max length
4. Return highest-scoring complete sequence

## 3. Implementation Plan

### 3.1 Core Sampling Functions

```cpp
// include/vesper/generation/sampling.h

namespace vesper::generation {

struct SamplingParams {
    float temperature = 1.0f;
    int64_t top_k = 0;          // 0 = disabled
    float top_p = 1.0f;         // 1.0 = disabled
    float repetition_penalty = 1.0f;
    int64_t min_new_tokens = 0;
    int64_t max_new_tokens = 256;
    std::vector<int64_t> stop_token_ids;
    
    bool do_sample = true;      // false = greedy
    int64_t seed = -1;          // -1 = random
};

// Sample next token from logits
// logits: [Batch, VocabSize]
// Returns: [Batch] token IDs
Tensor sample_next_token(const Tensor& logits, const SamplingParams& params);

// Apply temperature scaling
Tensor apply_temperature(const Tensor& logits, float temperature);

// Apply Top-K filtering
Tensor apply_top_k(const Tensor& logits, int64_t k);

// Apply Top-P (nucleus) filtering
Tensor apply_top_p(const Tensor& logits, float p);

// Apply repetition penalty
Tensor apply_repetition_penalty(const Tensor& logits, 
                                 const Tensor& input_ids,
                                 float penalty);

} // namespace vesper::generation
```

### 3.2 Temperature Implementation

```cpp
// src/generation/sampling.cpp

Tensor apply_temperature(const Tensor& logits, float temperature) {
    if (temperature == 1.0f) {
        return logits;
    }
    
    VESPER_CHECK(temperature > 0, "Temperature must be positive");
    return logits / temperature;
}
```

### 3.3 Top-K Implementation

```cpp
Tensor apply_top_k(const Tensor& logits, int64_t k) {
    if (k <= 0 || k >= logits.shape(-1)) {
        return logits;
    }
    
    // Get top-k values and indices
    auto [values, indices] = ops::topk(logits, k, /*dim=*/-1);
    
    // Create mask: set non-top-k to -inf
    Tensor mask = full_like(logits, -std::numeric_limits<float>::infinity());
    mask.scatter_(-1, indices, values);
    
    return mask;
}
```

### 3.4 Top-P Implementation

```cpp
Tensor apply_top_p(const Tensor& logits, float p) {
    if (p >= 1.0f) {
        return logits;
    }
    
    VESPER_CHECK(p > 0, "Top-P must be positive");
    
    // Sort in descending order
    auto [sorted_logits, sorted_indices] = ops::sort(logits, /*dim=*/-1, /*descending=*/true);
    
    // Compute cumulative probabilities
    Tensor sorted_probs = ops::softmax(sorted_logits, /*dim=*/-1);
    Tensor cumsum = ops::cumsum(sorted_probs, /*dim=*/-1);
    
    // Create mask: keep tokens until cumsum > p
    // Shift cumsum right by 1 (we want to include the token that pushes over p)
    Tensor shifted_cumsum = ops::cat({
        zeros({logits.shape(0), 1}, logits.dtype(), logits.device()),
        cumsum.slice(-1, 0, -1)
    }, /*dim=*/-1);
    
    Tensor sorted_mask = shifted_cumsum < p;
    
    // Set filtered logits to -inf
    sorted_logits = ops::where(sorted_mask, sorted_logits, 
                                full_like(sorted_logits, -INFINITY));
    
    // Unsort
    Tensor result = empty_like(logits);
    result.scatter_(-1, sorted_indices, sorted_logits);
    
    return result;
}
```

### 3.5 GPU Kernel for Top-P

Top-P requires sorting, which is expensive. Here's an optimized kernel:

```cpp
// src/ops/hip/top_p.hip

// Parallel Top-P filtering
// Each block handles one batch element
__global__ void top_p_kernel(
    const float* __restrict__ logits,    // [Batch, Vocab]
    float* __restrict__ output,           // [Batch, Vocab]
    int vocab_size, float p)
{
    extern __shared__ float shared[];
    float* s_logits = shared;
    int* s_indices = (int*)(shared + vocab_size);
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* batch_logits = logits + batch_idx * vocab_size;
    float* batch_output = output + batch_idx * vocab_size;
    
    // 1. Load logits and compute softmax (stable)
    float max_val = -INFINITY;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float v = batch_logits[i];
        s_logits[i] = v;
        max_val = fmaxf(max_val, v);
    }
    
    // Reduce max across block
    __shared__ float block_max;
    // ... warp reduction ...
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        s_logits[i] = expf(s_logits[i] - block_max);
        sum += s_logits[i];
    }
    
    // Reduce sum
    __shared__ float block_sum;
    // ... warp reduction ...
    
    // Normalize to get probabilities
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        s_logits[i] /= block_sum;
        s_indices[i] = i;
    }
    
    __syncthreads();
    
    // 2. Parallel sort (bitonic sort or radix sort)
    // For simplicity, use thrust on device
    // In production, use CUB/rocPRIM
    
    // 3. Compute cumsum and find cutoff
    // ...
    
    // 4. Write output (masked logits)
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        // If token is in nucleus, keep original logit; else -inf
        batch_output[i] = /* masked value */;
    }
}
```

### 3.6 Repetition Penalty

Discourages repeating tokens from the context:

```cpp
Tensor apply_repetition_penalty(const Tensor& logits, 
                                 const Tensor& input_ids,
                                 float penalty) {
    if (penalty == 1.0f) {
        return logits;
    }
    
    Tensor result = logits.clone();
    
    // For each token in input_ids, apply penalty
    // penalty > 1: decrease probability of repeated tokens
    // penalty < 1: increase probability (not common)
    
    for (int64_t b = 0; b < input_ids.shape(0); ++b) {
        for (int64_t s = 0; s < input_ids.shape(1); ++s) {
            int64_t token_id = input_ids[{b, s}].item<int64_t>();
            float logit = result[{b, token_id}].item<float>();
            
            if (logit > 0) {
                result[{b, token_id}] = logit / penalty;
            } else {
                result[{b, token_id}] = logit * penalty;
            }
        }
    }
    
    return result;
}
```

### 3.7 Complete Sampling Function

```cpp
Tensor sample_next_token(const Tensor& logits, const SamplingParams& params) {
    Tensor processed = logits;
    
    // 1. Repetition penalty (if applicable, needs input_ids)
    // Handled externally
    
    // 2. Temperature
    processed = apply_temperature(processed, params.temperature);
    
    // 3. Top-K
    if (params.top_k > 0) {
        processed = apply_top_k(processed, params.top_k);
    }
    
    // 4. Top-P
    if (params.top_p < 1.0f) {
        processed = apply_top_p(processed, params.top_p);
    }
    
    // 5. Convert to probabilities
    Tensor probs = ops::softmax(processed, /*dim=*/-1);
    
    // 6. Sample or argmax
    if (params.do_sample) {
        return ops::multinomial(probs, /*num_samples=*/1).squeeze(-1);
    } else {
        return ops::argmax(probs, /*dim=*/-1);
    }
}
```

## 4. The Generator Class

### 4.1 Definition

```cpp
// include/vesper/generation/generator.h

namespace vesper::generation {

class Generator {
public:
    Generator(models::Transformer* model);
    
    // Generate tokens autoregressively
    // prompt_ids: [Batch, PromptLen]
    // Returns: [Batch, PromptLen + GeneratedLen]
    Tensor generate(const Tensor& prompt_ids, const SamplingParams& params);
    
    // Streaming generation with callback
    using TokenCallback = std::function<void(int64_t batch_idx, int64_t token_id)>;
    void generate_streaming(const Tensor& prompt_ids, 
                             const SamplingParams& params,
                             TokenCallback on_token);
    
private:
    models::Transformer* model_;
    
    bool should_stop(int64_t token_id, const SamplingParams& params) const;
};

} // namespace vesper::generation
```

### 4.2 Implementation

```cpp
// src/generation/generator.cpp

namespace vesper::generation {

Generator::Generator(models::Transformer* model) : model_(model) {}

Tensor Generator::generate(const Tensor& prompt_ids, const SamplingParams& params) {
    auto [batch_size, prompt_len] = prompt_ids.sizes2d();
    Device device = prompt_ids.device();
    
    // Initialize KV cache
    model_->init_cache(batch_size, device);
    
    // Start with prompt
    std::vector<Tensor> all_tokens;
    all_tokens.push_back(prompt_ids);
    
    // Prefill: process entire prompt
    Tensor logits = model_->forward_with_cache(prompt_ids, 0);
    // logits: [Batch, 1, Vocab] (last token only)
    logits = logits.squeeze(1);  // [Batch, Vocab]
    
    int64_t generated = 0;
    int64_t pos = prompt_len;
    
    // Track which sequences are done
    Tensor done = zeros({batch_size}, DType::Bool, device);
    
    while (generated < params.max_new_tokens) {
        // Sample next token
        Tensor next_token = sample_next_token(logits, params);  // [Batch]
        all_tokens.push_back(next_token.unsqueeze(1));
        
        // Check stopping conditions
        for (int64_t b = 0; b < batch_size; ++b) {
            if (done[b].item<bool>()) continue;
            
            int64_t token = next_token[b].item<int64_t>();
            if (should_stop(token, params)) {
                done[b] = true;
            }
        }
        
        // All sequences done?
        if (done.all().item<bool>()) {
            break;
        }
        
        // Forward with new token
        logits = model_->forward_with_cache(next_token.unsqueeze(1), pos);
        logits = logits.squeeze(1);
        
        ++generated;
        ++pos;
    }
    
    // Concatenate all tokens
    return ops::cat(all_tokens, /*dim=*/1);
}

void Generator::generate_streaming(const Tensor& prompt_ids,
                                     const SamplingParams& params,
                                     TokenCallback on_token) {
    auto [batch_size, prompt_len] = prompt_ids.sizes2d();
    Device device = prompt_ids.device();
    
    model_->init_cache(batch_size, device);
    
    // Prefill
    Tensor logits = model_->forward_with_cache(prompt_ids, 0);
    logits = logits.squeeze(1);
    
    int64_t pos = prompt_len;
    Tensor done = zeros({batch_size}, DType::Bool, device);
    
    for (int64_t gen = 0; gen < params.max_new_tokens; ++gen) {
        Tensor next_token = sample_next_token(logits, params);
        
        // Call callback for each batch item
        for (int64_t b = 0; b < batch_size; ++b) {
            if (!done[b].item<bool>()) {
                int64_t token = next_token[b].item<int64_t>();
                on_token(b, token);
                
                if (should_stop(token, params)) {
                    done[b] = true;
                }
            }
        }
        
        if (done.all().item<bool>()) break;
        
        logits = model_->forward_with_cache(next_token.unsqueeze(1), pos);
        logits = logits.squeeze(1);
        ++pos;
    }
}

bool Generator::should_stop(int64_t token_id, const SamplingParams& params) const {
    for (int64_t stop_id : params.stop_token_ids) {
        if (token_id == stop_id) return true;
    }
    return false;
}

} // namespace vesper::generation
```

## 5. Beam Search

### 5.1 Beam Search Implementation

```cpp
// include/vesper/generation/beam_search.h

namespace vesper::generation {

struct BeamSearchParams {
    int64_t num_beams = 4;
    int64_t max_new_tokens = 256;
    float length_penalty = 1.0f;  // > 1 favors longer sequences
    bool early_stopping = false;
    std::vector<int64_t> eos_token_ids;
};

class BeamSearcher {
public:
    BeamSearcher(models::Transformer* model);
    
    // Returns the best sequence for each batch item
    Tensor search(const Tensor& prompt_ids, const BeamSearchParams& params);
    
private:
    struct Beam {
        Tensor token_ids;  // [SeqLen]
        float score;
        bool is_done;
    };
    
    models::Transformer* model_;
};

} // namespace vesper::generation
```

### 5.2 Beam Search Logic

```cpp
Tensor BeamSearcher::search(const Tensor& prompt_ids, const BeamSearchParams& params) {
    auto [batch_size, prompt_len] = prompt_ids.sizes2d();
    int64_t num_beams = params.num_beams;
    int64_t vocab_size = model_->config().vocab_size;
    Device device = prompt_ids.device();
    
    // Expand prompt for beam search: [B, S] -> [B * num_beams, S]
    Tensor expanded_prompt = prompt_ids.unsqueeze(1)
        .expand({batch_size, num_beams, prompt_len})
        .reshape({batch_size * num_beams, prompt_len});
    
    model_->init_cache(batch_size * num_beams, device);
    
    // Beam scores: [B * num_beams]
    Tensor beam_scores = zeros({batch_size * num_beams}, DType::Float32, device);
    // Only first beam per batch should be active initially
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t beam = 1; beam < num_beams; ++beam) {
            beam_scores[b * num_beams + beam] = -INFINITY;
        }
    }
    
    std::vector<Tensor> all_tokens;
    all_tokens.push_back(expanded_prompt);
    
    // Prefill
    Tensor logits = model_->forward_with_cache(expanded_prompt, 0);
    logits = logits.squeeze(1);  // [B * num_beams, Vocab]
    
    int64_t pos = prompt_len;
    Tensor done = zeros({batch_size}, DType::Bool, device);
    
    for (int64_t gen = 0; gen < params.max_new_tokens; ++gen) {
        // Log probabilities
        Tensor log_probs = ops::log_softmax(logits, -1);  // [B * beams, V]
        
        // Add beam scores: [B * beams, V]
        Tensor next_scores = log_probs + beam_scores.unsqueeze(-1);
        
        // Reshape to [B, beams * V] for top-k selection
        next_scores = next_scores.reshape({batch_size, num_beams * vocab_size});
        
        // Get top 2*beams candidates (for diversity)
        auto [top_scores, top_indices] = ops::topk(next_scores, 2 * num_beams, -1);
        
        // Compute beam and token indices
        Tensor next_beam_indices = top_indices / vocab_size;  // [B, 2*beams]
        Tensor next_tokens = top_indices % vocab_size;        // [B, 2*beams]
        
        // Select top num_beams for each batch
        // ... complex selection logic for handling EOS ...
        
        // Update beam_scores and token sequence
        // ...
        
        // Forward with next tokens
        logits = model_->forward_with_cache(/* selected tokens */, pos);
        logits = logits.squeeze(1);
        ++pos;
        
        if (done.all().item<bool>() && params.early_stopping) {
            break;
        }
    }
    
    // Return best sequence per batch
    // ...
}
```

## 6. Usage Examples

### 6.1 Basic Generation

```cpp
#include <vesper/models/transformer.h>
#include <vesper/generation/generator.h>

int main() {
    auto model = models::create_model("llama2-7b");
    model->to(Device::HIP);
    model->eval();
    
    Generator gen(model.get());
    
    SamplingParams params;
    params.temperature = 0.7f;
    params.top_p = 0.9f;
    params.max_new_tokens = 100;
    params.stop_token_ids = {2};  // EOS token
    
    // Tokenized prompt: "Once upon a time"
    Tensor prompt = tensor({{1, 9038, 2501, 263, 931}}, DType::Int64, Device::HIP);
    
    Tensor output = gen.generate(prompt, params);
    // Decode and print...
    
    return 0;
}
```

### 6.2 Streaming Generation

```cpp
Generator gen(model.get());

SamplingParams params;
params.temperature = 0.8f;
params.max_new_tokens = 50;

Tensor prompt = tokenize("The quick brown fox");

gen.generate_streaming(prompt, params, [&tokenizer](int64_t batch, int64_t token) {
    std::cout << tokenizer.decode({token}) << std::flush;
});
std::cout << std::endl;
```

### 6.3 Beam Search

```cpp
BeamSearcher searcher(model.get());

BeamSearchParams params;
params.num_beams = 4;
params.max_new_tokens = 50;
params.length_penalty = 0.6f;
params.eos_token_ids = {2};

Tensor prompt = tokenize("Translate to French: Hello");
Tensor result = searcher.search(prompt, params);
```

## 7. Testing Strategy

### 7.1 Unit Tests

```cpp
// tests/generation/test_sampling.cpp

TEST(Sampling, TemperatureScaling) {
    Tensor logits = tensor({{1.0f, 2.0f, 3.0f}});
    
    // T=1: standard softmax
    Tensor p1 = ops::softmax(apply_temperature(logits, 1.0f), -1);
    
    // T=0.5: sharper
    Tensor p05 = ops::softmax(apply_temperature(logits, 0.5f), -1);
    
    // T=2.0: flatter
    Tensor p2 = ops::softmax(apply_temperature(logits, 2.0f), -1);
    
    // Max probability should increase as T decreases
    EXPECT_GT(p05.max().item<float>(), p1.max().item<float>());
    EXPECT_GT(p1.max().item<float>(), p2.max().item<float>());
}

TEST(Sampling, TopK) {
    Tensor logits = tensor({{1.0f, 5.0f, 2.0f, 4.0f, 3.0f}});
    
    Tensor filtered = apply_top_k(logits, 2);
    Tensor probs = ops::softmax(filtered, -1);
    
    // Only indices 1 and 3 should have non-zero probability
    EXPECT_GT(probs[{0, 1}].item<float>(), 0);
    EXPECT_GT(probs[{0, 3}].item<float>(), 0);
    EXPECT_NEAR(probs[{0, 0}].item<float>(), 0, 1e-6);
    EXPECT_NEAR(probs[{0, 2}].item<float>(), 0, 1e-6);
    EXPECT_NEAR(probs[{0, 4}].item<float>(), 0, 1e-6);
}

TEST(Sampling, TopP) {
    // Uniform logits -> all tokens in nucleus
    Tensor uniform = zeros({1, 10});
    Tensor filtered_uniform = apply_top_p(uniform, 0.9f);
    Tensor probs_uniform = ops::softmax(filtered_uniform, -1);
    
    // Should sum to 1 (all tokens included for uniform)
    EXPECT_NEAR(probs_uniform.sum().item<float>(), 1.0f, 1e-5);
    
    // Peaked distribution
    Tensor peaked = tensor({{10.0f, 0.0f, 0.0f, 0.0f, 0.0f}});
    Tensor filtered_peaked = apply_top_p(peaked, 0.9f);
    Tensor probs_peaked = ops::softmax(filtered_peaked, -1);
    
    // Only first token should be included
    EXPECT_GT(probs_peaked[{0, 0}].item<float>(), 0.99f);
}

TEST(Sampling, GreedyDeterminism) {
    Tensor logits = randn({1, 1000});
    
    SamplingParams params;
    params.do_sample = false;  // Greedy
    
    Tensor token1 = sample_next_token(logits, params);
    Tensor token2 = sample_next_token(logits, params);
    
    EXPECT_EQ(token1.item<int64_t>(), token2.item<int64_t>());
    EXPECT_EQ(token1.item<int64_t>(), ops::argmax(logits, -1).item<int64_t>());
}

TEST(Sampling, SamplingWithSeed) {
    Tensor logits = randn({1, 1000});
    
    SamplingParams params;
    params.do_sample = true;
    params.temperature = 1.0f;
    params.seed = 42;
    
    // Same seed should give same result
    set_seed(42);
    Tensor token1 = sample_next_token(logits, params);
    
    set_seed(42);
    Tensor token2 = sample_next_token(logits, params);
    
    EXPECT_EQ(token1.item<int64_t>(), token2.item<int64_t>());
}
```

### 7.2 Stress Tests

```cpp
TEST(Generator, StressTest_LongGeneration) {
    auto config = TransformerConfig::llama2_7b();
    config.n_layers = 2;
    
    auto model = std::make_unique<Transformer>(config);
    model->to(Device::HIP);
    model->eval();
    
    Generator gen(model.get());
    
    SamplingParams params;
    params.max_new_tokens = 2000;
    params.temperature = 0.8f;
    
    Tensor prompt = randint(0, 32000, {1, 32}, DType::Int64, Device::HIP);
    
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = gen.generate(prompt, params);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    float tokens_per_sec = 2000.0f / (ms / 1000.0f);
    
    std::cout << "Generated 2000 tokens in " << ms << " ms" << std::endl;
    std::cout << "Speed: " << tokens_per_sec << " tokens/sec" << std::endl;
    
    EXPECT_EQ(output.shape(1), 32 + 2000);
}

TEST(Generator, StressTest_BatchGeneration) {
    auto config = TransformerConfig::llama2_7b();
    config.n_layers = 2;
    
    auto model = std::make_unique<Transformer>(config);
    model->to(Device::HIP);
    model->eval();
    
    Generator gen(model.get());
    
    SamplingParams params;
    params.max_new_tokens = 100;
    
    // Batch of 8
    Tensor prompt = randint(0, 32000, {8, 64}, DType::Int64, Device::HIP);
    
    Tensor output = gen.generate(prompt, params);
    
    EXPECT_EQ(output.shape(0), 8);
    EXPECT_EQ(output.shape(1), 64 + 100);
    EXPECT_FALSE(output.isnan().any().item<bool>());
}

TEST(Generator, StressTest_MemoryStability) {
    auto config = TransformerConfig::gpt2_small();
    config.n_layers = 4;
    
    auto model = std::make_unique<Transformer>(config);
    model->to(Device::HIP);
    model->eval();
    
    Generator gen(model.get());
    
    size_t initial_mem = get_hip_memory_usage();
    
    for (int i = 0; i < 20; ++i) {
        SamplingParams params;
        params.max_new_tokens = 50;
        
        Tensor prompt = randint(0, 50257, {2, 32}, DType::Int64, Device::HIP);
        Tensor output = gen.generate(prompt, params);
        
        // Clear cache between generations
        model->clear_cache();
    }
    
    hipDeviceSynchronize();
    size_t final_mem = get_hip_memory_usage();
    
    // Memory should not grow significantly
    EXPECT_LT(final_mem, initial_mem * 1.2);
}
```

### 7.3 Quality Tests

```cpp
TEST(Sampling, RepetitionPenaltyEffect) {
    Tensor logits = ones({1, 100});  // Uniform
    
    // Token 5 appears in input
    Tensor input_ids = tensor({{1, 5, 10, 5, 20}}, DType::Int64);
    
    // Without penalty
    Tensor probs_no_penalty = ops::softmax(logits, -1);
    float prob_5_no_penalty = probs_no_penalty[{0, 5}].item<float>();
    
    // With penalty
    Tensor penalized = apply_repetition_penalty(logits, input_ids, 1.5f);
    Tensor probs_penalty = ops::softmax(penalized, -1);
    float prob_5_penalty = probs_penalty[{0, 5}].item<float>();
    
    // Token 5 should be less likely with penalty
    EXPECT_LT(prob_5_penalty, prob_5_no_penalty);
}

TEST(Generator, StopTokenRespected) {
    auto config = TransformerConfig::gpt2_small();
    config.n_layers = 1;
    
    auto model = std::make_unique<Transformer>(config);
    model->eval();
    
    Generator gen(model.get());
    
    SamplingParams params;
    params.max_new_tokens = 1000;
    params.stop_token_ids = {50256};  // EOS for GPT-2
    
    Tensor prompt = tensor({{1, 2, 3}}, DType::Int64);
    Tensor output = gen.generate(prompt, params);
    
    // Should stop before max (unless EOS never generated)
    // Check that if EOS is in output, it's at the end
    int64_t eos_count = (output == 50256).sum().item<int64_t>();
    if (eos_count > 0) {
        // EOS should only appear once (at the end)
        EXPECT_EQ(eos_count, 1);
        EXPECT_EQ(output[{0, output.shape(1) - 1}].item<int64_t>(), 50256);
    }
}
```

## 8. Performance Optimization

### 8.1 Fused Softmax Sampling

Combine softmax, multinomial, and sampling into one kernel:

```cpp
// Fused kernel: logits -> sampled token
__global__ void fused_sample_kernel(
    const float* __restrict__ logits,
    int64_t* __restrict__ output,
    const float* __restrict__ random,  // Pre-generated random values
    int batch, int vocab, float temperature)
{
    // 1. Compute softmax (online)
    // 2. Compute cumulative sum
    // 3. Binary search for random value
    // All in one pass!
}
```

### 8.2 Speculative Sampling (Preview)

Accept/reject multiple tokens at once using a draft model (see Chapter 47).

## 9. Summary

Text generation combines:

1. **Temperature**: Controls randomness
2. **Top-K/Top-P**: Truncates distribution for quality
3. **Repetition Penalty**: Prevents loops
4. **Beam Search**: For deterministic, high-quality output

Key implementation points:
- Use GPU kernels for softmax and sampling
- Support streaming for real-time output
- Handle batch generation with variable stopping
- Clear KV cache between generations

With sampling implemented, Vesper can now generate coherent text from trained or pre-trained models.

```
