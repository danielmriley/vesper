#pragma once
#include <vesper/nn/module.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/normalization.h>
#include <vesper/core/tensor.h>
#include <utility>

namespace vesper::nn {

class MLP : public Module {
public:
    MLP(int embed_dim, float dropout=0.0);
    Tensor forward(const Tensor& x) override;

    Linear c_fc;
    Linear c_proj;
    float dropout_;
};

class KVCache {
public:
    KVCache(int batch_size, int num_heads, int max_seq_len, int head_dim, Device device);

    // Updates cache with new k/v and returns views of the full active context
    // new_k, new_v: [Batch, Heads, SeqLen, HeadDim]
    // start_pos: The position in the sequence to write to
    std::pair<Tensor, Tensor> update(const Tensor& new_k, const Tensor& new_v, int start_pos);

    int get_max_seq_len() const { return max_seq_len_; }

private:
    Tensor k_cache_;
    Tensor v_cache_;
    int max_seq_len_;
};

class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(int embed_dim, int num_heads, float dropout=0.0);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, bool causal);
    Tensor forward(const Tensor& x, KVCache* cache, int start_pos = 0);

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
    Tensor forward(const Tensor& x, KVCache* cache, int start_pos = 0);

    LayerNorm ln1;
    MultiHeadAttention attn;
    LayerNorm ln2;
    MLP mlp;
};

} // namespace vesper::nn
