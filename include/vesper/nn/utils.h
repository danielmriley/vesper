#pragma once
#include <vesper/core/tensor.h>
#include <vesper/nn/module.h>
#include <vesper/nn/embedding.h>
#include <vesper/nn/linear.h>
#include <vector>

namespace vesper::nn::utils {

/**
 * Clips gradient norm of an iterable of parameters.
 * 
 * The norm is computed over all gradients together, as if they were
 * concatenated into a single vector. Gradients are modified in-place.
 * 
 * This function operates entirely on GPU when parameters are on GPU,
 * avoiding CPU synchronization overhead.
 * 
 * @param parameters Vector of tensors whose gradients will be clipped
 * @param max_norm Maximum norm of the gradients
 * @param norm_type Type of the norm (default: 2.0 for L2 norm)
 * @return Total norm of the gradients (before clipping)
 */
float clip_grad_norm_(std::vector<Tensor>& parameters, float max_norm, float norm_type = 2.0f);

/**
 * Applies GPT-2 style scaled initialization to a tensor.
 * 
 * For transformer models, the residual connections benefit from scaling
 * the output projection weights by 1/sqrt(2*n_layers). This prevents
 * the variance from growing with depth.
 * 
 * @param tensor The tensor to initialize
 * @param std Base standard deviation (default: 0.02)
 * @param n_layers Number of transformer layers (for residual scaling)
 * @param is_residual_proj If true, scales by 1/sqrt(2*n_layers)
 */
void gpt2_init_(Tensor& tensor, float std = 0.02f, int n_layers = 1, bool is_residual_proj = false);

/**
 * Applies GPT-2/GPT-3 style initialization to all parameters in a module.
 * 
 * This follows the initialization scheme from "Language Models are Few-Shot Learners":
 * - Embedding weights: Normal(0, 0.02)
 * - All other weights: Normal(0, 0.02)
 * - Output projections in attention/MLP: Normal(0, 0.02 / sqrt(2*n_layers))
 * - Biases: Zero
 * - LayerNorm: weight=1, bias=0
 * 
 * @param module The module to initialize
 * @param n_layers Number of transformer layers (for residual scaling)
 * @param base_std Base standard deviation (default: 0.02)
 */
void init_transformer_weights(Module& module, int n_layers, float base_std = 0.02f);

/**
 * Ties the weights of an embedding layer to a linear output layer (lm_head).
 * 
 * In language models, it's common to share weights between the input embedding
 * and output projection. This reduces parameters and often improves generalization.
 * 
 * After calling this, the linear layer's weight will reference the same storage
 * as the embedding. Changes to one will affect the other.
 * 
 * Example:
 * ```cpp
 * nn::Embedding wte(vocab_size, embed_dim);
 * nn::Linear lm_head(embed_dim, vocab_size, false);  // no bias
 * nn::utils::tie_weights(wte, lm_head);
 * // Now lm_head.weight shares storage with wte.weight
 * ```
 * 
 * @param embedding The embedding layer
 * @param linear The linear output layer (lm_head)
 */
void tie_weights(nn::Embedding& embedding, nn::Linear& linear);

} // namespace vesper::nn::utils
