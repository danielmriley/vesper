#pragma once

#include <vesper/core/tensor.h>
#include <cstdint>

namespace vesper::ops {

// =============================================================================
// Top-K Filtering
// =============================================================================

/// Apply top-k filtering on GPU: keep only k highest values, set rest to -inf
/// @param logits Input logits [Batch, Vocab]
/// @param k Number of top values to keep
/// @return Filtered logits
void top_k_filter_hip_dispatch(const Tensor& logits, Tensor& output, int64_t k);
void top_k_filter_cuda_dispatch(const Tensor& logits, Tensor& output, int64_t k);
void top_k_filter_cpu_dispatch(const Tensor& logits, Tensor& output, int64_t k);

Tensor top_k_filter(const Tensor& logits, int64_t k);

// =============================================================================
// Top-P (Nucleus) Filtering
// =============================================================================

/// Apply top-p filtering on GPU: keep smallest set with cumulative prob > p
/// @param logits Input logits [Batch, Vocab]
/// @param p Cumulative probability threshold
/// @return Filtered logits
void top_p_filter_hip_dispatch(const Tensor& logits, Tensor& output, float p);
void top_p_filter_cuda_dispatch(const Tensor& logits, Tensor& output, float p);
void top_p_filter_cpu_dispatch(const Tensor& logits, Tensor& output, float p);

Tensor top_p_filter(const Tensor& logits, float p);

// =============================================================================
// Repetition Penalty
// =============================================================================

/// Apply repetition penalty to logits based on previously generated tokens
/// @param logits Input logits [Batch, Vocab]
/// @param input_ids Previously generated token IDs [Batch, SeqLen]
/// @param penalty Penalty factor (>1 discourages repetition)
/// @return Penalized logits
void repetition_penalty_hip_dispatch(const Tensor& logits, Tensor& output,
                                      const Tensor& input_ids, float penalty);
void repetition_penalty_cuda_dispatch(const Tensor& logits, Tensor& output,
                                       const Tensor& input_ids, float penalty);
void repetition_penalty_cpu_dispatch(const Tensor& logits, Tensor& output,
                                      const Tensor& input_ids, float penalty);

Tensor apply_repetition_penalty(const Tensor& logits, const Tensor& input_ids, float penalty);

// =============================================================================
// Argmax (Greedy Sampling)
// =============================================================================

/// Find argmax along last dimension (for greedy decoding)
/// @param input Input tensor [Batch, Vocab]
/// @return Indices of maximum values [Batch]
void argmax_hip_dispatch(const Tensor& input, Tensor& output);
void argmax_cuda_dispatch(const Tensor& input, Tensor& output);
void argmax_cpu_dispatch(const Tensor& input, Tensor& output);

Tensor argmax(const Tensor& input, int64_t dim = -1);

// =============================================================================
// Multinomial Sampling
// =============================================================================

/// Sample from a multinomial distribution on GPU
/// @param probs Probability distribution [Batch, Vocab] (must sum to 1)
/// @param num_samples Number of samples per batch element
/// @param seed Random seed for reproducibility
/// @return Sampled indices [Batch, NumSamples]
void multinomial_hip_dispatch(const Tensor& probs, Tensor& output, 
                               int64_t num_samples, uint64_t seed);
void multinomial_cuda_dispatch(const Tensor& probs, Tensor& output,
                                int64_t num_samples, uint64_t seed);
void multinomial_cpu_dispatch(const Tensor& probs, Tensor& output,
                               int64_t num_samples, uint64_t seed);

Tensor multinomial(const Tensor& probs, int64_t num_samples = 1, int64_t seed = -1);

// =============================================================================
// Softmax (used for sampling)
// =============================================================================

/// Apply softmax along last dimension on GPU
/// Assumes 2D input [Batch, Vocab]
void softmax_2d_hip_dispatch(const Tensor& input, Tensor& output);
void softmax_2d_cuda_dispatch(const Tensor& input, Tensor& output);
void softmax_2d_cpu_dispatch(const Tensor& input, Tensor& output);

// =============================================================================
// Top-K with Values and Indices
// =============================================================================

/// Get top-k values and indices (used internally for top-k and top-p)
/// @param input Input tensor [Batch, N]
/// @param k Number of top elements
/// @param values Output for top-k values [Batch, K]
/// @param indices Output for top-k indices [Batch, K]
void topk_hip_dispatch(const Tensor& input, int64_t k, Tensor& values, Tensor& indices);
void topk_cuda_dispatch(const Tensor& input, int64_t k, Tensor& values, Tensor& indices);
void topk_cpu_dispatch(const Tensor& input, int64_t k, Tensor& values, Tensor& indices);

std::pair<Tensor, Tensor> topk(const Tensor& input, int64_t k, int64_t dim = -1);

// =============================================================================
// Stop Token Checking (GPU-accelerated)
// =============================================================================

/// Check if tokens match any stop token IDs on GPU
/// Returns a mask (int32_t) indicating which batch elements should stop (1 = stop, 0 = continue)
/// @param tokens Token tensor [Batch] or [Batch, 1]
/// @param stop_token_ids Vector of stop token IDs to check against
/// @param min_generated Minimum number of tokens that must be generated before stopping
/// @param generated_count Current count of generated tokens
/// @return Mask [Batch] where 1 = should stop, 0 = continue
void check_stop_tokens_hip_dispatch(const Tensor& tokens, Tensor& output,
                                     const int64_t* stop_ids, int64_t num_stop_ids,
                                     int64_t min_generated, int64_t generated_count);
void check_stop_tokens_cuda_dispatch(const Tensor& tokens, Tensor& output,
                                      const int64_t* stop_ids, int64_t num_stop_ids,
                                      int64_t min_generated, int64_t generated_count);
void check_stop_tokens_cpu_dispatch(const Tensor& tokens, Tensor& output,
                                     const int64_t* stop_ids, int64_t num_stop_ids,
                                     int64_t min_generated, int64_t generated_count);

Tensor check_stop_tokens(const Tensor& tokens, const std::vector<int64_t>& stop_token_ids,
                          int64_t min_generated = 0, int64_t generated_count = 0);

// =============================================================================
// Batch Element Masking (for early termination)
// =============================================================================

/// Check if all elements in a mask are non-zero (all done)
/// This avoids transferring entire mask to CPU
bool all_true_hip_dispatch(const Tensor& mask);
bool all_true_cuda_dispatch(const Tensor& mask);
bool all_true_cpu_dispatch(const Tensor& mask);

bool all_true(const Tensor& mask);

} // namespace vesper::ops
