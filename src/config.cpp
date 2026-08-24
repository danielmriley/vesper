#include "vesper/config.h"

#include "vesper/types.h"

#include <sstream>

namespace vesper {

void ModelConfig::validate() const {
    check(vocab_size > 0, "vocab_size must be positive");
    check(hidden_size > 0, "hidden_size must be positive");
    check(n_layers > 0, "n_layers must be positive");
    check(n_heads > 0, "n_heads must be positive");
    check(n_kv_heads > 0, "n_kv_heads must be positive");
    check(head_dim > 0 && (head_dim % 2) == 0, "head_dim must be a positive even number");
    check(n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads");
    check(intermediate_size > 0, "intermediate_size must be positive");
    check(max_seq_len > 0, "max_seq_len must be positive");
    check(rms_eps > 0.0f, "rms_eps must be positive");
    check(rope_theta > 0.0f, "rope_theta must be positive");
}

std::string ModelConfig::describe() const {
    std::ostringstream out;
    out << "layers=" << n_layers
        << " hidden=" << hidden_size
        << " heads=" << n_heads << "/" << n_kv_heads
        << " head_dim=" << head_dim
        << " vocab=" << vocab_size
        << " seq=" << max_seq_len;
    return out.str();
}

ModelConfig ModelConfig::tiny_demo() {
    ModelConfig cfg;
    cfg.vocab_size = 256;
    cfg.hidden_size = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.intermediate_size = 128;
    cfg.max_seq_len = 256;
    cfg.validate();
    return cfg;
}

ModelConfig ModelConfig::qwen3_06b() {
    ModelConfig cfg;
    cfg.vocab_size = 151936;
    cfg.hidden_size = 1024;
    cfg.n_layers = 28;
    cfg.n_heads = 16;
    cfg.n_kv_heads = 8;
    cfg.head_dim = 128;
    cfg.intermediate_size = 3072;
    cfg.max_seq_len = 4096;
    cfg.rope_theta = 1000000.0f;
    cfg.qk_norm = true;
    cfg.tie_word_embeddings = true;
    cfg.validate();
    return cfg;
}

ModelConfig ModelConfig::qwen3_8b() {
    ModelConfig cfg;
    cfg.vocab_size = 151936;
    cfg.hidden_size = 4096;
    cfg.n_layers = 36;
    cfg.n_heads = 32;
    cfg.n_kv_heads = 8;
    cfg.head_dim = 128;
    cfg.intermediate_size = 12288;
    cfg.max_seq_len = 4096;
    cfg.rope_theta = 1000000.0f;
    cfg.qk_norm = true;
    cfg.tie_word_embeddings = false;
    cfg.validate();
    return cfg;
}

}  // namespace vesper
