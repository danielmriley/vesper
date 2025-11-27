#pragma once
#include <vesper/core/tensor.h>

namespace vesper::ops {

/**
 * @brief Repeat KV heads to match query heads for Grouped Query Attention (GQA)
 * 
 * This operation expands the key/value tensors from [B, KV_Heads, S, D] to 
 * [B, Q_Heads, S, D] by repeating each KV head n_rep times.
 * 
 * Used in GQA where multiple query heads share the same key-value head.
 * For example, Llama 2 70B has 64 query heads and 8 KV heads (n_rep=8).
 * 
 * @param x Input tensor of shape [Batch, KV_Heads, SeqLen, HeadDim]
 * @param n_rep Number of times to repeat each KV head (Q_Heads / KV_Heads)
 * @return Tensor of shape [Batch, KV_Heads * n_rep, SeqLen, HeadDim]
 * 
 * If n_rep == 1, returns the input unchanged (MHA case).
 */
Tensor repeat_kv(const Tensor& x, int64_t n_rep);

/**
 * @brief Backward pass for repeat_kv
 * 
 * The backward of repeat_kv is a sum-reduction over the repeated dimension.
 * 
 * @param grad_output Gradient w.r.t. output [Batch, Q_Heads, SeqLen, HeadDim]
 * @param n_rep Number of repetitions used in forward
 * @return Gradient w.r.t. input [Batch, KV_Heads, SeqLen, HeadDim]
 */
Tensor repeat_kv_backward(const Tensor& grad_output, int64_t n_rep);

// HIP dispatch functions
void repeat_kv_hip_dispatch(const Tensor& input, Tensor& output, int64_t n_rep);
void repeat_kv_backward_hip_dispatch(const Tensor& grad_output, Tensor& grad_input, int64_t n_rep);

// CPU dispatch functions  
void repeat_kv_cpu_dispatch(const Tensor& input, Tensor& output, int64_t n_rep);
void repeat_kv_backward_cpu_dispatch(const Tensor& grad_output, Tensor& grad_input, int64_t n_rep);

} // namespace vesper::ops
