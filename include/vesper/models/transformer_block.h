/**
 * @file transformer_block.h
 * @brief Modular transformer block supporting GPT-2 and Llama architectures
 * 
 * Chapter 33.6: Building a Complete GPT/Llama Model
 */

#pragma once

#include <vesper/models/config.h>
#include <vesper/nn/module.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/normalization.h>
#include <vesper/nn/transformer.h>
#include <vesper/nn/gqa_attention.h>
#include <vesper/nn/swiglu.h>
#include <vesper/core/tensor.h>
#include <memory>

namespace vesper::models {

/**
 * @brief GPT-2 style MLP with GELU activation
 * 
 * Architecture: Linear -> GELU -> Linear
 * Used in GPT-2, BERT, and similar models.
 */
class GPT2MLP : public nn::Module {
public:
    /**
     * @brief Construct GPT-2 MLP
     * @param d_model Input/output dimension
     * @param hidden_dim Hidden dimension (typically 4 * d_model)
     * @param bias Whether to use bias
     */
    GPT2MLP(int64_t d_model, int64_t hidden_dim, bool bias = true);
    
    Tensor forward(const Tensor& x) override;
    
private:
    nn::Linear c_fc_;    // [d_model, hidden_dim]
    nn::Linear c_proj_;  // [hidden_dim, d_model]
    int64_t d_model_;
    int64_t hidden_dim_;
};

/**
 * @brief Modular Transformer Block
 * 
 * Supports both GPT-2 style (post-norm) and Llama style (pre-norm) architectures.
 * Automatically selects components based on config:
 * - Attention: MHA (GPT-2/Llama 7B) or GQA (Llama 70B, Mistral)
 * - FFN: GELU MLP (GPT-2) or SwiGLU (Llama)
 * - Norm: LayerNorm (GPT-2) or RMSNorm (Llama)
 */
class ModelTransformerBlock : public nn::Module {
public:
    /**
     * @brief Construct a Transformer Block
     * @param config Model configuration
     * @param layer_idx Layer index (for debugging/naming)
     */
    ModelTransformerBlock(const TransformerConfig& config, int64_t layer_idx = 0);
    
    /**
     * @brief Forward pass without caching
     * @param x Input tensor [Batch, SeqLen, Dim]
     * @return Output tensor [Batch, SeqLen, Dim]
     */
    Tensor forward(const Tensor& x) override;
    
    /**
     * @brief Forward pass with causal masking control
     * @param x Input tensor
     * @param causal Whether to apply causal masking
     */
    Tensor forward(const Tensor& x, bool causal);
    
    /**
     * @brief Forward pass with KV cache
     * @param x Input tensor
     * @param cache KV cache for this layer
     * @param start_pos Position in sequence
     */
    Tensor forward(const Tensor& x, nn::KVCache* cache, int64_t start_pos);
    
    /// Get layer index
    int64_t layer_idx() const { return layer_idx_; }
    
private:
    TransformerConfig config_;
    int64_t layer_idx_;
    
    // Normalization (RMSNorm or LayerNorm)
    std::unique_ptr<nn::Module> attn_norm_;
    std::unique_ptr<nn::Module> ffn_norm_;
    
    // Attention (GQA or MHA)
    std::unique_ptr<nn::Module> attn_;
    bool use_gqa_;  // Track which attention type for casting
    
    // FFN (SwiGLU or GPT2 MLP)
    std::unique_ptr<nn::Module> ffn_;
};

} // namespace vesper::models
