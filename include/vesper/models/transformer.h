/**
 * @file transformer.h
 * @brief Complete Transformer language model (GPT-2, Llama, Mistral)
 * 
 * Chapter 33.6: Building a Complete GPT/Llama Model
 */

#pragma once

#include <vesper/models/config.h>
#include <vesper/models/transformer_block.h>
#include <vesper/nn/module.h>
#include <vesper/nn/embedding.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/normalization.h>
#include <vesper/nn/transformer.h>
#include <vesper/core/tensor.h>
#include <memory>
#include <vector>
#include <string>

namespace vesper::models {

/**
 * @brief Complete Transformer Language Model
 * 
 * Supports both GPT-2 and Llama-style architectures with:
 * - Token embeddings (required)
 * - Position embeddings (GPT-2) or RoPE (Llama)
 * - N transformer blocks
 * - Final normalization
 * - Output projection (LM head)
 * 
 * Features:
 * - Training forward pass (full sequence)
 * - Inference with KV cache (autoregressive generation)
 * - Optional weight tying (input/output embeddings)
 */
class TransformerLM : public nn::Module {
public:
    /**
     * @brief Construct a Transformer LM from config
     * @param config Model configuration
     */
    explicit TransformerLM(const TransformerConfig& config);
    
    // =========================================================================
    // Forward Passes
    // =========================================================================
    
    /**
     * @brief Training forward pass (full sequence)
     * @param tokens Input token IDs [Batch, SeqLen]
     * @return Logits [Batch, SeqLen, VocabSize]
     */
    Tensor forward(const Tensor& tokens) override;
    
    /**
     * @brief Inference forward pass with KV cache
     * @param tokens New token IDs [Batch, NewTokens]
     * @param start_pos Starting position in sequence
     * @return Logits [Batch, NewTokens, VocabSize] (or just last for efficiency)
     */
    Tensor forward_with_cache(const Tensor& tokens, int64_t start_pos);
    
    /**
     * @brief Generate tokens autoregressively
     * @param prompt Input prompt token IDs [Batch, PromptLen]
     * @param max_new_tokens Maximum number of new tokens to generate
     * @param temperature Sampling temperature (1.0 = neutral)
     * @param top_k If > 0, only sample from top k tokens
     * @return Generated token IDs [Batch, PromptLen + NumGenerated]
     */
    Tensor generate(const Tensor& prompt, 
                   int64_t max_new_tokens,
                   float temperature = 1.0f,
                   int64_t top_k = 0);
    
    /**
     * @brief Compute cross-entropy loss for language modeling
     * @param tokens Input token IDs [Batch, SeqLen]
     * @param targets Target token IDs [Batch, SeqLen] (usually tokens shifted by 1)
     * @return Scalar loss tensor
     */
    Tensor compute_loss(const Tensor& tokens, const Tensor& targets);
    
    // =========================================================================
    // KV Cache Management
    // =========================================================================
    
    /**
     * @brief Initialize KV caches for all layers
     * @param batch_size Batch size
     * @param device Device to allocate caches on
     */
    void init_cache(int64_t batch_size, Device device);
    
    /**
     * @brief Clear all KV caches
     */
    void clear_cache();

    /**
     * @brief Reorder the beam rows of every layer's KV cache
     *
     * Used by beam search: after each step the surviving beams may have forked
     * from different parents, so each layer's cached past context must be
     * permuted to follow the beam it continues.
     *
     * @param src_rows src_rows[i] = source beam index that destination row i takes
     */
    void reorder_cache(const std::vector<int64_t>& src_rows);

    /**
     * @brief Check if caches are initialized
     */
    bool has_cache() const { return !kv_caches_.empty(); }
    
    // =========================================================================
    // Model Information
    // =========================================================================
    
    /// Get model configuration
    const TransformerConfig& config() const { return config_; }
    
    /// Count total parameters
    int64_t num_parameters() const;
    
    /// Get model size in bytes (FP32)
    int64_t model_size_bytes() const { return num_parameters() * 4; }
    
    /// Get human-readable model description
    std::string describe() const;
    
    // =========================================================================
    // Weight Access (for loading/saving)
    // =========================================================================
    
    /// Get token embedding weight
    Tensor& tok_embedding_weight() { return tok_emb_.weight; }
    
    /// Get output projection weight (may be tied to embedding)
    Tensor& output_weight();
    
private:
    TransformerConfig config_;
    
    // Embeddings
    nn::Embedding tok_emb_;
    std::unique_ptr<nn::Embedding> pos_emb_;  // GPT-2 only, nullptr for Llama
    
    // Transformer blocks
    std::vector<std::unique_ptr<ModelTransformerBlock>> blocks_;
    
    // Final normalization
    std::unique_ptr<nn::Module> final_norm_;
    
    // Output projection (lm_head)
    std::unique_ptr<nn::Linear> lm_head_;  // nullptr if tie_word_embeddings
    
    // KV caches (one per layer)
    std::vector<std::unique_ptr<nn::KVCache>> kv_caches_;
    
    // Cached values
    float dropout_;
};

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * @brief Create a model from configuration
 * @param config Model configuration
 * @return Unique pointer to model
 */
std::unique_ptr<TransformerLM> create_model(const TransformerConfig& config);

/**
 * @brief Create a model by name
 * @param name Model name (e.g., "gpt2-small", "llama2-7b")
 * @return Unique pointer to model
 */
std::unique_ptr<TransformerLM> create_model(const std::string& name);

/// Convenience factory for GPT-2 variants
std::unique_ptr<TransformerLM> create_gpt2(const std::string& size = "small");

/// Convenience factory for Llama 2 variants  
std::unique_ptr<TransformerLM> create_llama2(const std::string& size = "7b");

/// Convenience factory for Llama 3 variants
std::unique_ptr<TransformerLM> create_llama3(const std::string& size = "8b");

} // namespace vesper::models
