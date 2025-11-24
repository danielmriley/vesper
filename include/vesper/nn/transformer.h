#pragma once
#include <vesper/nn/module.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/normalization.h>
#include <vesper/core/tensor.h>

namespace vesper::nn {

class MLP : public Module {
public:
    MLP(int embed_dim, float dropout=0.0);
    Tensor forward(const Tensor& x) override;

    Linear c_fc;
    Linear c_proj;
    float dropout_;
};

class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(int embed_dim, int num_heads, float dropout=0.0);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, bool causal);

    Linear c_attn;
    Linear c_proj;
    int n_head;
    int n_embd;
    float dropout_;
};

class TransformerBlock : public Module {
public:
    TransformerBlock(int embed_dim, int num_heads, float dropout=0.0);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, bool causal);

    LayerNorm ln1;
    MultiHeadAttention attn;
    LayerNorm ln2;
    MLP mlp;
};

} // namespace vesper::nn
