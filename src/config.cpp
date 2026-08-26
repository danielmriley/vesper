#include "vesper/config.h"

#include "vesper/types.h"

#include <sstream>

namespace vesper {

LayerKind ModelConfig::layer_kind(int layer) const {
    check(layer >= 0 && layer < n_layers, "layer index out of range");
    if (full_attention_interval <= 0) {
        return LayerKind::Attention;
    }
    return ((layer + 1) % full_attention_interval == 0) ? LayerKind::Attention
                                                        : LayerKind::DeltaNet;
}

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
    const int rotary = rotary_dim();
    check(rotary > 0 && (rotary % 2) == 0 && rotary <= head_dim, "rope_dim must be even and <= head_dim");
    if (full_attention_interval > 0) {
        check(gdn_conv_kernel >= 2, "gdn conv kernel must be >= 2");
        check(gdn_qk_heads > 0 && gdn_v_heads > 0, "gdn head counts must be positive");
        check(gdn_v_heads % gdn_qk_heads == 0, "gdn v heads must be a multiple of qk heads");
        check(gdn_head_dim > 0, "gdn head dim must be positive");
    }
}

std::string ModelConfig::describe() const {
    std::ostringstream out;
    out << "arch=" << arch
        << " layers=" << n_layers
        << " hidden=" << hidden_size
        << " heads=" << n_heads << "/" << n_kv_heads
        << " head_dim=" << head_dim
        << " vocab=" << vocab_size
        << " seq=" << max_seq_len;
    if (is_hybrid()) {
        out << " hybrid=" << full_attention_interval
            << " gdn=" << gdn_v_heads << "x" << gdn_head_dim;
    }
    return out.str();
}

ModelConfig ModelConfig::tiny_demo() {
    ModelConfig cfg;
    cfg.arch = "vesper_tiny";
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

ModelConfig ModelConfig::tiny_hybrid() {
    ModelConfig cfg;
    cfg.arch = "vesper_hybrid";
    cfg.vocab_size = 256;
    cfg.hidden_size = 64;
    cfg.n_layers = 4;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.intermediate_size = 128;
    cfg.max_seq_len = 64;
    cfg.attn_gate = true;
    cfg.full_attention_interval = 4;
    cfg.gdn_conv_kernel = 4;
    cfg.gdn_qk_heads = 2;
    cfg.gdn_v_heads = 4;
    cfg.gdn_head_dim = 16;
    cfg.validate();
    return cfg;
}

ModelConfig ModelConfig::qwen3_06b() {
    ModelConfig cfg;
    cfg.arch = "qwen3";
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
    cfg.arch = "qwen3";
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

ModelConfig ModelConfig::qwen38_27b() {
    ModelConfig cfg;
    cfg.arch = "qwen35";
    cfg.vocab_size = 248320;
    cfg.hidden_size = 5120;
    cfg.n_layers = 64;
    cfg.n_heads = 24;
    cfg.n_kv_heads = 4;
    cfg.head_dim = 256;
    cfg.intermediate_size = 17408;
    cfg.max_seq_len = 4096;
    cfg.rms_eps = 1e-6f;
    cfg.rope_theta = 10000000.0f;
    cfg.qk_norm = true;
    cfg.tie_word_embeddings = false;
    cfg.attn_gate = true;
    cfg.rope_dim = 64;
    cfg.full_attention_interval = 4;
    cfg.gdn_conv_kernel = 4;
    cfg.gdn_qk_heads = 16;
    cfg.gdn_v_heads = 48;
    cfg.gdn_head_dim = 128;
    cfg.validate();
    return cfg;
}

}  // namespace vesper
