#pragma once
#include <vesper/nn/module.h>
#include <vesper/core/tensor.h>

namespace vesper::nn {

class Embedding : public Module {
public:
    // num_embeddings: size of the dictionary of embeddings
    // embedding_dim: size of each embedding vector
    // padding_idx: If specified, the entries at padding_idx do not contribute to the gradient;
    //              therefore, the embedding vector at padding_idx is not updated during training,
    //              i.e. it remains as a fixed "pad".
    // max_norm: If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.
    // norm_type: The p of the p-norm to compute for the max_norm option. Default 2.
    // scale_grad_by_freq: If given, this will scale gradients by the inverse of frequency in the mini-batch. Default false.
    // sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor. Default false.
    Embedding(int64_t num_embeddings, 
              int64_t embedding_dim, 
              int64_t padding_idx = -1, 
              float max_norm = -1.0f, 
              float norm_type = 2.0f,
              bool scale_grad_by_freq = false,
              bool sparse = false,
              Device device = Device::CPU);

    Tensor forward(const Tensor& input) override;

    void from_pretrained(const Tensor& embeddings, bool freeze = false);
    void reset_padding_idx();

    Tensor weight;
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    int64_t padding_idx_;
    float max_norm_;
    float norm_type_;
    bool scale_grad_by_freq_;
    bool sparse_;

private:
    void apply_max_norm(const Tensor& indices);
};

} // namespace vesper::nn
