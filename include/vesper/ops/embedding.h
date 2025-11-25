#pragma once
#include <vesper/core/tensor.h>

namespace vesper::ops {

// Lookup embedding vectors.
// indices: Tensor of indices (Int32 or Int64)
// weight: [num_embeddings, embedding_dim]
// padding_idx: If not -1, the embedding at this index is not updated and stays 0.
// scale_grad_by_freq: If true, gradients are scaled by the inverse of frequency of the words in the mini-batch. (Not required for V1)
// sparse: If true, gradient w.r.t. weight will be a sparse tensor. (Not required for V1 - we return dense grad)
// max_norm: If > 0, each embedding vector with norm > max_norm is renormalized to max_norm.
// norm_type: The p of the p-norm to compute for the max_norm option. Default 2.

Tensor embedding(const Tensor& input, const Tensor& weight, int64_t padding_idx = -1, bool scale_grad_by_freq = false, bool sparse = false, float max_norm = -1.0f, float norm_type = 2.0f);

// Backend Dispatchers
void embedding_cpu_dispatch(const Tensor& input, const Tensor& weight, int64_t padding_idx, float max_norm, float norm_type, Tensor& out);
void embedding_cuda_dispatch(const Tensor& input, const Tensor& weight, int64_t padding_idx, float max_norm, float norm_type, Tensor& out);
void embedding_hip_dispatch(const Tensor& input, const Tensor& weight, int64_t padding_idx, float max_norm, float norm_type, Tensor& out);

void embedding_backward_cpu_dispatch(const Tensor& grad_output, const Tensor& input, int64_t num_embeddings, int64_t padding_idx, bool scale_grad_by_freq, Tensor& grad_weight);
void embedding_backward_cuda_dispatch(const Tensor& grad_output, const Tensor& input, int64_t num_embeddings, int64_t padding_idx, bool scale_grad_by_freq, Tensor& grad_weight);
void embedding_backward_hip_dispatch(const Tensor& grad_output, const Tensor& input, int64_t num_embeddings, int64_t padding_idx, bool scale_grad_by_freq, Tensor& grad_weight);

} // namespace vesper::ops
