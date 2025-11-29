#pragma once

#include <vesper/core/tensor.h>
#include <tuple>

namespace vesper::ops {

/**
 * @brief Memory-efficient attention using Flash Attention algorithm
 * 
 * Standard attention computes the full N×N attention matrix, requiring O(N²) memory.
 * Flash Attention tiles the computation and uses online softmax to achieve O(N) memory.
 * 
 * This is critical for long sequences - at N=4096 with 32 heads, standard attention
 * needs ~2GB just for attention matrices, while Flash Attention uses ~10MB.
 * 
 * Algorithm:
 * 1. Tile Q, K, V into blocks
 * 2. For each Q block, iterate over K,V blocks
 * 3. Use online softmax to incrementally compute attention
 * 4. Never materialize the full N×N attention matrix
 * 
 * @param q Query tensor [Batch, Heads, SeqLen, HeadDim]
 * @param k Key tensor [Batch, Heads, SeqLen, HeadDim]
 * @param v Value tensor [Batch, Heads, SeqLen, HeadDim]
 * @param scale Attention scale factor (typically 1/sqrt(head_dim))
 * @param is_causal Apply causal (lower triangular) mask
 * @param dropout_p Dropout probability (0.0 = no dropout, typically 0.1 for training)
 * @param training Whether in training mode (dropout only applied when true)
 * @return Output tensor [Batch, Heads, SeqLen, HeadDim]
 * 
 * Memory: O(Batch * Heads * SeqLen * HeadDim) instead of O(Batch * Heads * SeqLen²)
 */
Tensor flash_attention(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    float scale,
    bool is_causal = true,
    float dropout_p = 0.0f,
    bool training = false);

/**
 * @brief Backward pass for Flash Attention
 * 
 * Computes gradients for Q, K, V using the same tiled algorithm to maintain
 * O(N) memory complexity during backpropagation.
 * 
 * @param grad_output Gradient of loss w.r.t. output [B, H, N, D]
 * @param q Query tensor from forward [B, H, N, D]
 * @param k Key tensor from forward [B, H, N, D]
 * @param v Value tensor from forward [B, H, N, D]
 * @param output Output from forward pass [B, H, N, D]
 * @param lse Log-sum-exp saved from forward [B, H, N]
 * @param scale Attention scale factor
 * @param is_causal Whether causal mask was applied
 * @param dropout_p Dropout probability used in forward
 * @param dropout_mask Optional dropout mask saved from forward [B, H, N, N] or empty
 * @return Tuple of (grad_q, grad_k, grad_v)
 */
std::tuple<Tensor, Tensor, Tensor> flash_attention_backward(
    const Tensor& grad_output,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& output,
    const Tensor& lse,
    float scale,
    bool is_causal = true,
    float dropout_p = 0.0f,
    const Tensor& dropout_mask = Tensor());

// Backend dispatch functions (with dropout support)
void flash_attention_hip_dispatch(
    const Tensor& q, const Tensor& k, const Tensor& v,
    Tensor& output, Tensor& lse,
    float scale, bool is_causal,
    float dropout_p, bool training, Tensor* dropout_mask_out);

void flash_attention_backward_hip_dispatch(
    const Tensor& grad_output,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& output, const Tensor& lse,
    Tensor& grad_q, Tensor& grad_k, Tensor& grad_v,
    float scale, bool is_causal,
    float dropout_p, const Tensor* dropout_mask);

void flash_attention_cpu_dispatch(
    const Tensor& q, const Tensor& k, const Tensor& v,
    Tensor& output, Tensor& lse,
    float scale, bool is_causal,
    float dropout_p, bool training, Tensor* dropout_mask_out);

void flash_attention_backward_cpu_dispatch(
    const Tensor& grad_output,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& output, const Tensor& lse,
    Tensor& grad_q, Tensor& grad_k, Tensor& grad_v,
    float scale, bool is_causal,
    float dropout_p, const Tensor* dropout_mask);

} // namespace vesper::ops
