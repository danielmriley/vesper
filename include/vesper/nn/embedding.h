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
    Embedding(int64_t num_embeddings, int64_t embedding_dim, int64_t padding_idx = -1, float max_norm = -1.0f, Device device = Device::CPU);

    Tensor forward(const Tensor& input) override;

    Tensor weight;
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    int64_t padding_idx_;
    float max_norm_;
};

} // namespace vesper::nn
