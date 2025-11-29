#pragma once
#include <vesper/nn/module.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/normalization.h>
#include <vesper/nn/swiglu.h>
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

    void reset();
    int current_seq_len() const { return current_len_; }
    int get_max_seq_len() const { return max_seq_len_; }

private:
    Tensor k_cache_;
    Tensor v_cache_;
    int max_seq_len_;
    int current_len_ = 0;
};

class MultiHeadAttention : public Module {
public:
    // use_rope: If true, applies Rotary Position Embeddings (RoPE) to Q and K.
    //           If false, no positional encoding is applied (user can add their own).
    MultiHeadAttention(int embed_dim, int num_heads, float dropout=0.0, bool use_rope=true);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, bool causal);
    Tensor forward(const Tensor& x, KVCache* cache, int start_pos = 0);

    Linear c_attn;
    Linear c_proj;
    int n_head;
    int n_embd;
    float dropout_;
    bool use_rope_;
};

class TransformerBlock : public Module {
public:
    // use_rope: If true, applies Rotary Position Embeddings in attention.
    //           If false, no positional encoding (user adds their own to input).
    TransformerBlock(int embed_dim, int num_heads, float dropout=0.0, bool use_rope=true);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, bool causal);
    Tensor forward(const Tensor& x, KVCache* cache, int start_pos = 0);

    LayerNorm ln1;
    MultiHeadAttention attn;
    LayerNorm ln2;
    MLP mlp;
};

/**
 * @brief Llama-style Transformer Block
 * 
 * Modern transformer architecture used by Llama, Mistral, and similar models.
 * Key differences from GPT-2 style TransformerBlock:
 *   - RMSNorm instead of LayerNorm (faster, no mean computation)
 *   - SwiGLU MLP instead of GELU MLP (better performance)
 *   - Pre-norm architecture (same as TransformerBlock)
 *   - RoPE positional embeddings (default on)
 * 
 * Usage:
 *   // Simple API
 *   LlamaBlock block(384, 6);  // embed_dim=384, num_heads=6
 *   
 *   // Full config API
 *   LlamaBlock::Config cfg;
 *   cfg.embed_dim = 384;
 *   cfg.num_heads = 6;
 *   cfg.ffn_hidden_dim = 1024;  // 0 = auto-compute
 *   cfg.rms_eps = 1e-6f;
 *   LlamaBlock block(cfg);
 */
class LlamaBlock : public Module {
public:
    struct Config {
        int embed_dim = 256;
        int num_heads = 4;
        int ffn_hidden_dim = 0;  // 0 = auto-compute for SwiGLU parameter parity
        float dropout = 0.0f;
        float rms_eps = 1e-5f;
        bool use_rope = true;
    };

    // Simple constructor - auto-computes FFN hidden dim
    LlamaBlock(int embed_dim, int num_heads, float dropout = 0.0f, bool use_rope = true);
    
    // Full config constructor
    explicit LlamaBlock(const Config& config);
    
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, bool causal);
    Tensor forward(const Tensor& x, KVCache* cache, int start_pos = 0);
    
    // Accessors
    const Config& config() const { return config_; }

    // Public members for direct access (like TransformerBlock)
    RMSNorm attn_norm;
    MultiHeadAttention attn;
    RMSNorm ffn_norm;
    SwiGLUMLP ffn;

private:
    Config config_;
};

} // namespace vesper::nn
