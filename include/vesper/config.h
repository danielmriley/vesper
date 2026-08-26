#pragma once

#include <string>

namespace vesper {

enum class LayerKind {
    Attention,
    DeltaNet,
};

struct ModelConfig {
    std::string arch = "vesper_tiny";
    int vocab_size = 256;
    int hidden_size = 64;
    int n_layers = 2;
    int n_heads = 4;
    int n_kv_heads = 2;
    int head_dim = 16;
    int intermediate_size = 128;
    float rms_eps = 1e-6f;
    float rope_theta = 10000.0f;
    bool qk_norm = true;
    bool tie_word_embeddings = true;
    int max_seq_len = 256;
    bool attn_gate = false;
    int rope_dim = 0;
    int full_attention_interval = 0;
    int gdn_conv_kernel = 4;
    int gdn_qk_heads = 0;
    int gdn_v_heads = 0;
    int gdn_head_dim = 0;
    int nextn_predict_layers = 0;

    int q_dim() const { return n_heads * head_dim; }
    int kv_dim() const { return n_kv_heads * head_dim; }
    int gqa_group() const { return n_heads / n_kv_heads; }
    int q_proj_rows() const { return attn_gate ? 2 * q_dim() : q_dim(); }
    int rotary_dim() const { return rope_dim > 0 ? rope_dim : head_dim; }
    bool is_hybrid() const { return full_attention_interval > 0; }
    int gdn_key_dim() const { return gdn_qk_heads * gdn_head_dim; }
    int gdn_value_dim() const { return gdn_v_heads * gdn_head_dim; }
    int gdn_qkv_dim() const { return 2 * gdn_key_dim() + gdn_value_dim(); }
    int gdn_conv_dim() const { return gdn_qkv_dim(); }
    int gdn_rec_elems() const { return gdn_v_heads * gdn_head_dim * gdn_head_dim; }
    int gdn_conv_state_elems() const { return gdn_conv_dim() * (gdn_conv_kernel - 1); }
    LayerKind layer_kind(int layer) const;

    void validate() const;
    std::string describe() const;

    static ModelConfig tiny_demo();
    static ModelConfig tiny_hybrid();
    static ModelConfig tiny_q4km();
    static ModelConfig qwen3_06b();
    static ModelConfig qwen3_8b();
    static ModelConfig qwen38_27b();
};

}  // namespace vesper
