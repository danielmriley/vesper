#pragma once

/**
 * @file functional.h
 * @brief Functional API for neural network operations
 * 
 * All functions operate on Tensors and support autograd (backward pass).
 */

#include <vesper/core/tensor.h>

namespace vesper::nn::functional {

// ============================================================================
// Activation Functions
// ============================================================================

/// Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
/// @param input Any shape tensor
/// @return Same shape, values in [0, 1]
Tensor sigmoid(const Tensor& input);
void sigmoid_hip_dispatch(const Tensor& input, Tensor& output);
void sigmoid_cuda_dispatch(const Tensor& input, Tensor& output);

/// ReLU activation: max(0, x)
/// @param input Any shape tensor
/// @return Same shape, negative values clamped to 0
Tensor relu(const Tensor& input);
void relu_hip_dispatch(const Tensor& input, Tensor& output);
void relu_cuda_dispatch(const Tensor& input, Tensor& output);

/// GELU activation (default implementation uses tanh approximation)
/// @param input Any shape tensor
Tensor gelu(const Tensor& input);
/// GELU with tanh approximation (faster)
Tensor gelu_tanh(const Tensor& input);
/// GELU with erf (more accurate)
Tensor gelu_erf(const Tensor& input);

/// SiLU / Swish activation: x * sigmoid(x)
/// @param input Any shape tensor
Tensor silu(const Tensor& input);
/// In-place SiLU
void silu_(Tensor& input);

// ============================================================================
// Regularization
// ============================================================================

/// Dropout: randomly zeros elements during training
/// @param input Any shape tensor
/// @param p Probability of zeroing (default 0.5)
/// @param training If false, returns input unchanged
Tensor dropout(const Tensor& input, double p = 0.5, bool training = true);

// ============================================================================
// Softmax & Normalization
// ============================================================================

/// Softmax: exp(x) / sum(exp(x)) along dimension
/// @param input Any shape tensor
/// @param dim Dimension to apply softmax (can be negative)
/// @return Same shape, values sum to 1 along dim
/// @note Output may not be contiguous. Call .contiguous() if needed.
Tensor softmax(const Tensor& input, int64_t dim);

/// Log-softmax: log(softmax(x)) - more numerically stable than log(softmax())
/// @param input Any shape tensor
/// @param dim Dimension to apply log_softmax
Tensor log_softmax(const Tensor& input, int64_t dim);

/// Layer normalization
/// @param input [*, normalized_shape]
/// @param normalized_shape Usually {hidden_dim}
/// @param weight Optional scale parameter (same shape as normalized_shape)
/// @param bias Optional bias parameter (same shape as normalized_shape)
/// @param eps Small constant for numerical stability
Tensor layer_norm(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                  const Tensor& weight = {}, const Tensor& bias = {}, float eps = 1e-5);

/// RMS normalization (used in LLaMA, etc.)
/// @param input [*, normalized_shape]
/// @param normalized_shape Usually {hidden_dim}
/// @param weight Scale parameter
/// @param eps Small constant for numerical stability
Tensor rms_norm(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                const Tensor& weight, float eps = 1e-5);

// ============================================================================
// Loss Functions
// ============================================================================

/// Mean Squared Error loss
/// @param y_pred Predictions, any shape
/// @param y_true Targets, same shape as y_pred
/// @return Scalar tensor with mean((y_pred - y_true)^2)
Tensor mse_loss(const Tensor& y_pred, const Tensor& y_true);

/// Cross-entropy loss for classification
/// 
/// @param logits Raw scores BEFORE softmax, shape [N, C]
///               where N = batch size, C = number of classes
/// @param targets Integer class labels, shape [N], values in [0, C-1]
///                Must be Int32 or Int64 dtype
/// @param ignore_index Targets equal to this value are excluded from both the
///                loss and the gradient (e.g. padding tokens). The mean is taken
///                over the non-ignored rows only. Defaults to -1.
/// @return Scalar tensor with mean cross-entropy loss
///
/// @note This function applies log_softmax internally.
///       Do NOT apply softmax to logits before calling this!
///
/// Example:
///   Tensor logits = model.forward(x);        // [32, 10] for 10-class problem
///   Tensor targets = ...;                     // [32] with values 0-9
///   Tensor loss = cross_entropy_loss(logits, targets);
///   loss.backward();
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets, int64_t ignore_index = -1);

// ============================================================================
// Attention
// ============================================================================

/// Scaled dot-product attention
/// 
/// @param query  [B, H, T, D] or [B, T, D]
/// @param key    [B, H, S, D] or [B, S, D]
/// @param value  [B, H, S, D] or [B, S, D]
/// @param is_causal If true, applies causal mask (for autoregressive models)
/// @param dropout_p Dropout probability (applied to attention weights)
/// @return Same shape as query
/// 
/// Computes: softmax(Q @ K^T / sqrt(D)) @ V
/// With causal mask, positions can only attend to earlier positions.
Tensor scaled_dot_product_attention(const Tensor& query, 
                                    const Tensor& key, 
                                    const Tensor& value, 
                                    bool is_causal = false, 
                                    double dropout_p = 0.0);

// ============================================================================
// Rotary Position Embeddings (RoPE)
// ============================================================================

/// Compute RoPE frequency tensor (legacy API)
/// @param seq_len Sequence length
/// @param head_dim Dimension of each attention head
/// @param start_pos Starting position (for KV cache)
/// @param theta Base frequency (default 10000)
/// @param device Target device
Tensor compute_rope_frequencies(int seq_len, int head_dim, int start_pos = 0, 
                                float theta = 10000.0f, Device device = Device::CPU);

/// Apply rotary embeddings to tensor (legacy API)
/// @param x Input tensor [B, H, T, D]
/// @param freqs Frequency tensor from compute_rope_frequencies
Tensor apply_rotary_emb(const Tensor& x, const Tensor& freqs);

// Internal dispatch functions
void apply_rope_hip_dispatch(float* x, const float* freqs_cos, const float* freqs_sin,
                             int batch, int heads, int seq_len, int head_dim);
void apply_rope_cuda_dispatch(float* x, const float* freqs_cos, const float* freqs_sin,
                              int batch, int heads, int seq_len, int head_dim);

} // namespace vesper::nn::functional
