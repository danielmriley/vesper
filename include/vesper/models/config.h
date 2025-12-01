/**
 * @file config.h
 * @brief Transformer model configurations for GPT-2, Llama, Mistral, etc.
 * 
 * Chapter 33.6: Building a Complete GPT/Llama Model
 */

#pragma once

#include <string>
#include <cstdint>

namespace vesper::models {

/**
 * @brief Configuration for Transformer language models
 * 
 * Supports both GPT-2 style (learned position embeddings, post-norm, GELU MLP)
 * and Llama style (RoPE, pre-norm, SwiGLU, GQA) architectures.
 */
struct TransformerConfig {
    // =========================================================================
    // Model Architecture
    // =========================================================================
    
    int64_t vocab_size = 32000;      ///< Vocabulary size
    int64_t dim = 4096;              ///< Hidden dimension (d_model / embed_dim)
    int64_t n_layers = 32;           ///< Number of transformer blocks
    int64_t n_heads = 32;            ///< Number of attention (query) heads
    int64_t n_kv_heads = -1;         ///< KV heads for GQA (-1 = same as n_heads)
    int64_t max_seq_len = 4096;      ///< Maximum sequence length
    
    // =========================================================================
    // Feed-Forward Network
    // =========================================================================
    
    int64_t ffn_hidden_dim = 0;      ///< FFN hidden dimension (0 = auto-compute)
    int64_t ffn_multiple_of = 256;   ///< Round FFN hidden dim to this multiple
    
    // =========================================================================
    // Normalization
    // =========================================================================
    
    float norm_eps = 1e-5f;          ///< Epsilon for normalization layers
    bool use_rms_norm = true;        ///< true = RMSNorm (Llama), false = LayerNorm (GPT)
    
    // =========================================================================
    // Position Encoding (RoPE)
    // =========================================================================
    
    float rope_base = 10000.0f;      ///< RoPE base frequency (0 = learned positions)
    float rope_scaling = 1.0f;       ///< RoPE scaling factor for extended context
    
    // =========================================================================
    // Regularization
    // =========================================================================
    
    float dropout = 0.0f;            ///< Dropout probability
    float attention_dropout = 0.0f;  ///< Attention-specific dropout
    
    // =========================================================================
    // Other Options
    // =========================================================================
    
    bool tie_word_embeddings = false; ///< Share input/output embedding weights
    bool use_bias = false;           ///< Use bias in linear layers
    
    // =========================================================================
    // Training Optimizations
    // =========================================================================
    
    int gradient_checkpointing = 0;  ///< Checkpoint every N layers (0 = disabled)
    
    // =========================================================================
    // Derived Values
    // =========================================================================
    
    /// Get head dimension (dim / n_heads)
    int64_t head_dim() const { return dim / n_heads; }
    
    /// Get actual number of KV heads (handles -1 default)
    int64_t kv_heads() const { return n_kv_heads > 0 ? n_kv_heads : n_heads; }
    
    /// Check if this config uses GQA
    bool uses_gqa() const { return kv_heads() < n_heads; }
    
    /// Check if this config uses RoPE (vs learned positions)
    bool uses_rope() const { return rope_base > 0; }
    
    /// Compute FFN hidden dimension (Llama style: 8/3 * dim, rounded)
    int64_t hidden_dim() const;
    
    /// Calculate KV cache memory ratio compared to MHA
    float kv_cache_ratio() const {
        return static_cast<float>(kv_heads()) / n_heads;
    }
    
    // =========================================================================
    // Factory Methods - GPT-2 Configurations
    // =========================================================================
    
    /// GPT-2 Small: 124M parameters
    static TransformerConfig gpt2_small();
    
    /// GPT-2 Medium: 355M parameters
    static TransformerConfig gpt2_medium();
    
    /// GPT-2 Large: 774M parameters
    static TransformerConfig gpt2_large();
    
    /// GPT-2 XL: 1.5B parameters
    static TransformerConfig gpt2_xl();
    
    // =========================================================================
    // Factory Methods - Llama 2 Configurations
    // =========================================================================
    
    /// Llama 2 7B: 6.7B parameters, MHA
    static TransformerConfig llama2_7b();
    
    /// Llama 2 13B: 13B parameters, MHA
    static TransformerConfig llama2_13b();
    
    /// Llama 2 70B: 70B parameters, GQA (64Q/8KV)
    static TransformerConfig llama2_70b();
    
    // =========================================================================
    // Factory Methods - Llama 3 Configurations
    // =========================================================================
    
    /// Llama 3 8B: 8B parameters, GQA (32Q/8KV)
    static TransformerConfig llama3_8b();
    
    /// Llama 3 70B: 70B parameters, GQA (64Q/8KV)
    static TransformerConfig llama3_70b();
    
    // =========================================================================
    // Factory Methods - Other Models
    // =========================================================================
    
    /// Mistral 7B: 7B parameters, GQA (32Q/8KV)
    static TransformerConfig mistral_7b();
    
    // =========================================================================
    // Utility
    // =========================================================================
    
    /// Create config from model name string (e.g., "llama2-7b", "gpt2-small")
    static TransformerConfig from_name(const std::string& name);
    
    /// Get a human-readable description of this config
    std::string describe() const;
};

} // namespace vesper::models
