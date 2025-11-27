#pragma once

#include <vesper/generation/sampling.h>
#include <vesper/models/transformer.h>
#include <vesper/core/tensor.h>
#include <functional>
#include <memory>

namespace vesper::generation {

/**
 * @brief High-level text generator for transformer models
 * 
 * The Generator class provides a clean interface for autoregressive text generation
 * using a transformer language model. It supports:
 * - Standard generation with various sampling strategies
 * - Streaming generation with token callbacks
 * - Multiple stop conditions (EOS tokens, max length)
 * - Batch generation
 */
class Generator {
public:
    /**
     * @brief Construct a generator with a transformer model
     * @param model Pointer to the transformer model (must outlive Generator)
     */
    explicit Generator(models::TransformerLM* model);
    
    /**
     * @brief Generate tokens autoregressively
     * @param prompt_ids Input prompt token IDs [Batch, PromptLen]
     * @param params Sampling parameters
     * @return Generated token IDs [Batch, PromptLen + GeneratedLen]
     */
    Tensor generate(const Tensor& prompt_ids, const SamplingParams& params);
    
    /// Callback type for streaming generation
    /// @param batch_idx Which batch element produced the token
    /// @param token_id The generated token ID
    /// @return Return false to stop generation for this batch element
    using TokenCallback = std::function<bool(int64_t batch_idx, int64_t token_id)>;
    
    /**
     * @brief Streaming generation with token callback
     * 
     * Calls the callback for each generated token, allowing real-time
     * processing (e.g., printing tokens as they're generated).
     * 
     * @param prompt_ids Input prompt token IDs [Batch, PromptLen]
     * @param params Sampling parameters  
     * @param on_token Callback invoked for each generated token
     * @return Final generated token IDs [Batch, TotalLen]
     */
    Tensor generate_streaming(const Tensor& prompt_ids,
                              const SamplingParams& params,
                              TokenCallback on_token);
    
    /**
     * @brief Set the random seed for reproducible generation
     * @param seed Random seed (-1 for random)
     */
    void set_seed(int64_t seed) { seed_ = seed; }
    
    /**
     * @brief Get the underlying model
     */
    models::TransformerLM* model() { return model_; }
    const models::TransformerLM* model() const { return model_; }

private:
    models::TransformerLM* model_;
    int64_t seed_ = -1;
    
    /// Check if generation should stop for a given token
    bool should_stop(int64_t token_id, const SamplingParams& params, 
                     int64_t generated_count) const;
};

} // namespace vesper::generation
