#pragma once

#include <vesper/core/tensor.h>
#include <vector>
#include <cstdint>

namespace vesper::generation {

/// Parameters controlling the sampling strategy
struct SamplingParams {
    float temperature = 1.0f;       ///< Temperature scaling (>0). Lower = sharper distribution
    int64_t top_k = 0;              ///< Top-K filtering (0 = disabled)
    float top_p = 1.0f;             ///< Top-P (nucleus) filtering (1.0 = disabled)
    float repetition_penalty = 1.0f; ///< Penalty for repeated tokens (1.0 = no penalty)
    int64_t min_new_tokens = 0;     ///< Minimum tokens to generate before stopping
    int64_t max_new_tokens = 256;   ///< Maximum tokens to generate
    std::vector<int64_t> stop_token_ids; ///< Stop generation when these tokens are produced
    
    bool do_sample = true;          ///< If false, use greedy decoding (always pick max)
    int64_t seed = -1;              ///< Random seed (-1 = use random seed)
};

/// Apply temperature scaling to logits
/// @param logits: [Batch, VocabSize] raw logits
/// @param temperature: scaling factor (>0)
/// @returns scaled logits
Tensor apply_temperature(const Tensor& logits, float temperature);

/// Apply Top-K filtering: keep only the K highest probability tokens
/// @param logits: [Batch, VocabSize] raw logits  
/// @param k: number of top tokens to keep
/// @returns filtered logits (non-top-k set to -inf)
Tensor apply_top_k(const Tensor& logits, int64_t k);

/// Apply Top-P (nucleus) filtering: keep smallest set with cumulative prob > p
/// @param logits: [Batch, VocabSize] raw logits
/// @param p: cumulative probability threshold (0 < p <= 1)
/// @returns filtered logits
Tensor apply_top_p(const Tensor& logits, float p);

/// Apply repetition penalty to discourage repeating tokens
/// @param logits: [Batch, VocabSize] raw logits
/// @param input_ids: [Batch, SeqLen] previously generated token IDs
/// @param penalty: penalty factor (>1 discourages repetition)
/// @returns penalized logits
Tensor apply_repetition_penalty(const Tensor& logits, 
                                const Tensor& input_ids,
                                float penalty);

/// Sample next token from logits using the specified parameters
/// @param logits: [Batch, VocabSize] raw logits
/// @param params: sampling configuration
/// @param input_ids: optional previous tokens for repetition penalty [Batch, SeqLen]
/// @returns [Batch] sampled token IDs
Tensor sample_next_token(const Tensor& logits, 
                         const SamplingParams& params,
                         const Tensor* input_ids = nullptr);

/// Greedy sampling: select the token with highest probability
/// @param logits: [Batch, VocabSize]
/// @returns [Batch] token IDs
Tensor greedy_sample(const Tensor& logits);

/// Multinomial sampling from probability distribution
/// @param probs: [Batch, VocabSize] probability distribution (must sum to 1)
/// @param num_samples: number of samples per batch element
/// @param seed: random seed (-1 for random)
/// @returns [Batch, num_samples] sampled indices
Tensor multinomial_sample(const Tensor& probs, int64_t num_samples = 1, int64_t seed = -1);

} // namespace vesper::generation
