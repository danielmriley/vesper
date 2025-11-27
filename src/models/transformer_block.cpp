/**
 * @file transformer_block.cpp
 * @brief Implementation of modular transformer block
 * 
 * Chapter 33.6: Building a Complete GPT/Llama Model
 */

#include <vesper/models/transformer_block.h>
#include <vesper/nn/functional.h>
#include <vesper/ops/elementwise.h>
#include <stdexcept>

namespace vesper::models {

// =============================================================================
// GPT2MLP Implementation
// =============================================================================

GPT2MLP::GPT2MLP(int64_t d_model, int64_t hidden_dim, bool bias)
    : c_fc_(d_model, hidden_dim, bias),
      c_proj_(hidden_dim, d_model, bias),
      d_model_(d_model),
      hidden_dim_(hidden_dim)
{
    register_module<GPT2MLP>("c_fc", &GPT2MLP::c_fc_);
    register_module<GPT2MLP>("c_proj", &GPT2MLP::c_proj_);
}

Tensor GPT2MLP::forward(const Tensor& x) {
    // x -> Linear -> GELU -> Linear
    Tensor h = c_fc_.forward(x);
    h = nn::functional::gelu(h);
    return c_proj_.forward(h);
}

// =============================================================================
// ModelTransformerBlock Implementation
// =============================================================================

ModelTransformerBlock::ModelTransformerBlock(const TransformerConfig& config, int64_t layer_idx)
    : config_(config), layer_idx_(layer_idx)
{
    // =========================================================================
    // Create Normalization Layers
    // =========================================================================
    
    if (config.use_rms_norm) {
        // Llama-style RMSNorm
        auto attn_rms = std::make_unique<nn::RMSNorm>(
            std::vector<int64_t>{config.dim}, config.norm_eps);
        auto ffn_rms = std::make_unique<nn::RMSNorm>(
            std::vector<int64_t>{config.dim}, config.norm_eps);
        
        register_module("attention_norm", attn_rms.get());
        register_module("ffn_norm", ffn_rms.get());
        
        attn_norm_ = std::move(attn_rms);
        ffn_norm_ = std::move(ffn_rms);
    } else {
        // GPT-2-style LayerNorm
        auto ln1 = std::make_unique<nn::LayerNorm>(
            std::vector<int64_t>{config.dim}, config.norm_eps);
        auto ln2 = std::make_unique<nn::LayerNorm>(
            std::vector<int64_t>{config.dim}, config.norm_eps);
        
        register_module("ln_1", ln1.get());
        register_module("ln_2", ln2.get());
        
        attn_norm_ = std::move(ln1);
        ffn_norm_ = std::move(ln2);
    }
    
    // =========================================================================
    // Create Attention Layer
    // =========================================================================
    
    if (config.uses_gqa()) {
        // Grouped Query Attention (Llama 70B, Mistral, Llama 3)
        auto gqa = std::make_unique<nn::GroupedQueryAttention>(
            config.dim,
            config.n_heads,
            config.kv_heads(),
            config.max_seq_len,
            config.rope_base,
            config.attention_dropout,
            config.use_bias);
        
        register_module("attention", gqa.get());
        attn_ = std::move(gqa);
        use_gqa_ = true;
    } else {
        // Standard Multi-Head Attention
        // For Llama-style (with RoPE), we still use GQA but with n_kv_heads == n_heads
        if (config.uses_rope()) {
            // Use GQA module even for MHA (when n_kv_heads == n_heads)
            auto gqa = std::make_unique<nn::GroupedQueryAttention>(
                config.dim,
                config.n_heads,
                config.n_heads,  // Same as query heads = MHA
                config.max_seq_len,
                config.rope_base,
                config.attention_dropout,
                config.use_bias);
            
            register_module("attention", gqa.get());
            attn_ = std::move(gqa);
            use_gqa_ = true;
        } else {
            // GPT-2 style MHA (learned positions, handled externally)
            auto mha = std::make_unique<nn::MultiHeadAttention>(
                static_cast<int>(config.dim),
                static_cast<int>(config.n_heads),
                config.attention_dropout);
            
            register_module("attn", mha.get());
            attn_ = std::move(mha);
            use_gqa_ = false;
        }
    }
    
    // =========================================================================
    // Create FFN Layer
    // =========================================================================
    
    if (config.use_rms_norm) {
        // Llama-style: SwiGLU
        auto swiglu = std::make_unique<nn::SwiGLUMLP>(
            config.dim, config.hidden_dim(), config.use_bias);
        
        register_module("feed_forward", swiglu.get());
        ffn_ = std::move(swiglu);
    } else {
        // GPT-2-style: GELU MLP
        auto mlp = std::make_unique<GPT2MLP>(
            config.dim, config.hidden_dim(), config.use_bias);
        
        register_module("mlp", mlp.get());
        ffn_ = std::move(mlp);
    }
}

Tensor ModelTransformerBlock::forward(const Tensor& x) {
    return forward(x, true);  // Default causal for LM
}

Tensor ModelTransformerBlock::forward(const Tensor& x, bool causal) {
    // Pre-Norm architecture (Llama style):
    //   h = x + attn(norm(x))
    //   out = h + ffn(norm(h))
    
    // Attention with residual
    Tensor attn_input = attn_norm_->forward(x);
    Tensor attn_out;
    
    if (use_gqa_) {
        attn_out = static_cast<nn::GroupedQueryAttention*>(attn_.get())->forward(attn_input, causal);
    } else {
        attn_out = static_cast<nn::MultiHeadAttention*>(attn_.get())->forward(attn_input, causal);
    }
    
    Tensor h = ops::add(x, attn_out);
    
    // Apply dropout if training
    if (config_.dropout > 0.0f && training_) {
        h = nn::functional::dropout(h, config_.dropout, true);
    }
    
    // FFN with residual
    Tensor ffn_input = ffn_norm_->forward(h);
    Tensor ffn_out = ffn_->forward(ffn_input);
    Tensor out = ops::add(h, ffn_out);
    
    if (config_.dropout > 0.0f && training_) {
        out = nn::functional::dropout(out, config_.dropout, true);
    }
    
    return out;
}

Tensor ModelTransformerBlock::forward(const Tensor& x, nn::KVCache* cache, int64_t start_pos) {
    // Same as above but with KV cache
    
    Tensor attn_input = attn_norm_->forward(x);
    Tensor attn_out;
    
    if (use_gqa_) {
        attn_out = static_cast<nn::GroupedQueryAttention*>(attn_.get())->forward(
            attn_input, cache, start_pos);
    } else {
        attn_out = static_cast<nn::MultiHeadAttention*>(attn_.get())->forward(
            attn_input, cache, static_cast<int>(start_pos));
    }
    
    Tensor h = ops::add(x, attn_out);
    
    // No dropout during inference (cache implies inference)
    
    Tensor ffn_input = ffn_norm_->forward(h);
    Tensor ffn_out = ffn_->forward(ffn_input);
    
    return ops::add(h, ffn_out);
}

} // namespace vesper::models
