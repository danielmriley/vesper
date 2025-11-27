/**
 * @file config.cpp
 * @brief Implementation of TransformerConfig
 * 
 * Chapter 33.6: Building a Complete GPT/Llama Model
 */

#include <vesper/models/config.h>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace vesper::models {

int64_t TransformerConfig::hidden_dim() const {
    if (ffn_hidden_dim > 0) {
        return ffn_hidden_dim;
    }
    
    // Llama-style: (8/3) * dim, rounded to multiple_of
    int64_t h = (dim * 8) / 3;
    if (ffn_multiple_of > 1) {
        h = ((h + ffn_multiple_of - 1) / ffn_multiple_of) * ffn_multiple_of;
    }
    return h;
}

std::string TransformerConfig::describe() const {
    std::ostringstream ss;
    ss << "TransformerConfig {\n";
    ss << "  vocab_size: " << vocab_size << "\n";
    ss << "  dim: " << dim << "\n";
    ss << "  n_layers: " << n_layers << "\n";
    ss << "  n_heads: " << n_heads << "\n";
    ss << "  n_kv_heads: " << kv_heads();
    if (uses_gqa()) ss << " (GQA)";
    ss << "\n";
    ss << "  head_dim: " << head_dim() << "\n";
    ss << "  max_seq_len: " << max_seq_len << "\n";
    ss << "  ffn_hidden_dim: " << hidden_dim() << "\n";
    ss << "  norm: " << (use_rms_norm ? "RMSNorm" : "LayerNorm") << "\n";
    ss << "  position: " << (uses_rope() ? "RoPE" : "learned") << "\n";
    if (uses_rope()) {
        ss << "  rope_base: " << rope_base << "\n";
    }
    ss << "  bias: " << (use_bias ? "yes" : "no") << "\n";
    ss << "  tie_embeddings: " << (tie_word_embeddings ? "yes" : "no") << "\n";
    ss << "}";
    return ss.str();
}

// =============================================================================
// GPT-2 Configurations
// =============================================================================

TransformerConfig TransformerConfig::gpt2_small() {
    TransformerConfig cfg;
    cfg.vocab_size = 50257;
    cfg.dim = 768;
    cfg.n_layers = 12;
    cfg.n_heads = 12;
    cfg.n_kv_heads = 12;  // MHA
    cfg.max_seq_len = 1024;
    cfg.ffn_hidden_dim = 3072;  // 4x
    cfg.norm_eps = 1e-5f;
    cfg.use_rms_norm = false;   // LayerNorm
    cfg.rope_base = 0;          // Learned positions
    cfg.dropout = 0.1f;
    cfg.attention_dropout = 0.1f;
    cfg.tie_word_embeddings = true;
    cfg.use_bias = true;
    return cfg;
}

TransformerConfig TransformerConfig::gpt2_medium() {
    TransformerConfig cfg;
    cfg.vocab_size = 50257;
    cfg.dim = 1024;
    cfg.n_layers = 24;
    cfg.n_heads = 16;
    cfg.n_kv_heads = 16;
    cfg.max_seq_len = 1024;
    cfg.ffn_hidden_dim = 4096;
    cfg.norm_eps = 1e-5f;
    cfg.use_rms_norm = false;
    cfg.rope_base = 0;
    cfg.dropout = 0.1f;
    cfg.attention_dropout = 0.1f;
    cfg.tie_word_embeddings = true;
    cfg.use_bias = true;
    return cfg;
}

TransformerConfig TransformerConfig::gpt2_large() {
    TransformerConfig cfg;
    cfg.vocab_size = 50257;
    cfg.dim = 1280;
    cfg.n_layers = 36;
    cfg.n_heads = 20;
    cfg.n_kv_heads = 20;
    cfg.max_seq_len = 1024;
    cfg.ffn_hidden_dim = 5120;
    cfg.norm_eps = 1e-5f;
    cfg.use_rms_norm = false;
    cfg.rope_base = 0;
    cfg.dropout = 0.1f;
    cfg.attention_dropout = 0.1f;
    cfg.tie_word_embeddings = true;
    cfg.use_bias = true;
    return cfg;
}

TransformerConfig TransformerConfig::gpt2_xl() {
    TransformerConfig cfg;
    cfg.vocab_size = 50257;
    cfg.dim = 1600;
    cfg.n_layers = 48;
    cfg.n_heads = 25;
    cfg.n_kv_heads = 25;
    cfg.max_seq_len = 1024;
    cfg.ffn_hidden_dim = 6400;
    cfg.norm_eps = 1e-5f;
    cfg.use_rms_norm = false;
    cfg.rope_base = 0;
    cfg.dropout = 0.1f;
    cfg.attention_dropout = 0.1f;
    cfg.tie_word_embeddings = true;
    cfg.use_bias = true;
    return cfg;
}

// =============================================================================
// Llama 2 Configurations
// =============================================================================

TransformerConfig TransformerConfig::llama2_7b() {
    TransformerConfig cfg;
    cfg.vocab_size = 32000;
    cfg.dim = 4096;
    cfg.n_layers = 32;
    cfg.n_heads = 32;
    cfg.n_kv_heads = 32;  // MHA for 7B
    cfg.max_seq_len = 4096;
    cfg.ffn_hidden_dim = 11008;
    cfg.norm_eps = 1e-5f;
    cfg.use_rms_norm = true;
    cfg.rope_base = 10000.0f;
    cfg.rope_scaling = 1.0f;
    cfg.dropout = 0.0f;
    cfg.attention_dropout = 0.0f;
    cfg.tie_word_embeddings = false;
    cfg.use_bias = false;
    return cfg;
}

TransformerConfig TransformerConfig::llama2_13b() {
    TransformerConfig cfg;
    cfg.vocab_size = 32000;
    cfg.dim = 5120;
    cfg.n_layers = 40;
    cfg.n_heads = 40;
    cfg.n_kv_heads = 40;  // MHA for 13B
    cfg.max_seq_len = 4096;
    cfg.ffn_hidden_dim = 13824;
    cfg.norm_eps = 1e-5f;
    cfg.use_rms_norm = true;
    cfg.rope_base = 10000.0f;
    cfg.rope_scaling = 1.0f;
    cfg.dropout = 0.0f;
    cfg.attention_dropout = 0.0f;
    cfg.tie_word_embeddings = false;
    cfg.use_bias = false;
    return cfg;
}

TransformerConfig TransformerConfig::llama2_70b() {
    TransformerConfig cfg;
    cfg.vocab_size = 32000;
    cfg.dim = 8192;
    cfg.n_layers = 80;
    cfg.n_heads = 64;
    cfg.n_kv_heads = 8;  // GQA!
    cfg.max_seq_len = 4096;
    cfg.ffn_hidden_dim = 28672;
    cfg.norm_eps = 1e-5f;
    cfg.use_rms_norm = true;
    cfg.rope_base = 10000.0f;
    cfg.rope_scaling = 1.0f;
    cfg.dropout = 0.0f;
    cfg.attention_dropout = 0.0f;
    cfg.tie_word_embeddings = false;
    cfg.use_bias = false;
    return cfg;
}

// =============================================================================
// Llama 3 Configurations
// =============================================================================

TransformerConfig TransformerConfig::llama3_8b() {
    TransformerConfig cfg;
    cfg.vocab_size = 128256;
    cfg.dim = 4096;
    cfg.n_layers = 32;
    cfg.n_heads = 32;
    cfg.n_kv_heads = 8;  // GQA
    cfg.max_seq_len = 8192;
    cfg.ffn_hidden_dim = 14336;
    cfg.norm_eps = 1e-5f;
    cfg.use_rms_norm = true;
    cfg.rope_base = 500000.0f;  // Higher base for Llama 3
    cfg.rope_scaling = 1.0f;
    cfg.dropout = 0.0f;
    cfg.attention_dropout = 0.0f;
    cfg.tie_word_embeddings = false;
    cfg.use_bias = false;
    return cfg;
}

TransformerConfig TransformerConfig::llama3_70b() {
    TransformerConfig cfg;
    cfg.vocab_size = 128256;
    cfg.dim = 8192;
    cfg.n_layers = 80;
    cfg.n_heads = 64;
    cfg.n_kv_heads = 8;  // GQA
    cfg.max_seq_len = 8192;
    cfg.ffn_hidden_dim = 28672;
    cfg.norm_eps = 1e-5f;
    cfg.use_rms_norm = true;
    cfg.rope_base = 500000.0f;
    cfg.rope_scaling = 1.0f;
    cfg.dropout = 0.0f;
    cfg.attention_dropout = 0.0f;
    cfg.tie_word_embeddings = false;
    cfg.use_bias = false;
    return cfg;
}

// =============================================================================
// Mistral Configuration
// =============================================================================

TransformerConfig TransformerConfig::mistral_7b() {
    TransformerConfig cfg;
    cfg.vocab_size = 32000;
    cfg.dim = 4096;
    cfg.n_layers = 32;
    cfg.n_heads = 32;
    cfg.n_kv_heads = 8;  // GQA
    cfg.max_seq_len = 8192;  // Note: Mistral uses sliding window, not implemented here
    cfg.ffn_hidden_dim = 14336;
    cfg.norm_eps = 1e-5f;
    cfg.use_rms_norm = true;
    cfg.rope_base = 10000.0f;
    cfg.rope_scaling = 1.0f;
    cfg.dropout = 0.0f;
    cfg.attention_dropout = 0.0f;
    cfg.tie_word_embeddings = false;
    cfg.use_bias = false;
    return cfg;
}

// =============================================================================
// Factory from Name
// =============================================================================

static std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

TransformerConfig TransformerConfig::from_name(const std::string& name) {
    std::string lower = to_lower(name);
    
    // GPT-2 variants
    if (lower == "gpt2" || lower == "gpt2-small" || lower == "gpt2_small") {
        return gpt2_small();
    }
    if (lower == "gpt2-medium" || lower == "gpt2_medium") {
        return gpt2_medium();
    }
    if (lower == "gpt2-large" || lower == "gpt2_large") {
        return gpt2_large();
    }
    if (lower == "gpt2-xl" || lower == "gpt2_xl") {
        return gpt2_xl();
    }
    
    // Llama 2 variants
    if (lower == "llama2-7b" || lower == "llama2_7b" || lower == "llama-2-7b") {
        return llama2_7b();
    }
    if (lower == "llama2-13b" || lower == "llama2_13b" || lower == "llama-2-13b") {
        return llama2_13b();
    }
    if (lower == "llama2-70b" || lower == "llama2_70b" || lower == "llama-2-70b") {
        return llama2_70b();
    }
    
    // Llama 3 variants
    if (lower == "llama3-8b" || lower == "llama3_8b" || lower == "llama-3-8b") {
        return llama3_8b();
    }
    if (lower == "llama3-70b" || lower == "llama3_70b" || lower == "llama-3-70b") {
        return llama3_70b();
    }
    
    // Mistral
    if (lower == "mistral-7b" || lower == "mistral_7b" || lower == "mistral7b") {
        return mistral_7b();
    }
    
    throw std::runtime_error("Unknown model configuration: " + name);
}

} // namespace vesper::models
