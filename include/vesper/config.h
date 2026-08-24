#pragma once

#include <string>

namespace vesper {

struct ModelConfig {
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

    int q_dim() const { return n_heads * head_dim; }
    int kv_dim() const { return n_kv_heads * head_dim; }
    int gqa_group() const { return n_heads / n_kv_heads; }

    void validate() const;
    std::string describe() const;

    static ModelConfig tiny_demo();
    static ModelConfig qwen3_06b();
    static ModelConfig qwen3_8b();
};

}  // namespace vesper
