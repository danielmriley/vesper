#pragma once

#include <vesper/models/transformer.h>
#include <vesper/core/tensor.h>
#include <vector>
#include <cstdint>

namespace vesper::generation {

/**
 * @brief Parameters for beam search decoding
 */
struct BeamSearchParams {
    int64_t num_beams = 4;          ///< Number of beams to maintain
    int64_t max_new_tokens = 256;   ///< Maximum tokens to generate
    float length_penalty = 1.0f;    ///< Length penalty (>1 favors longer sequences)
    bool early_stopping = false;    ///< Stop when num_beams sentences finish
    std::vector<int64_t> eos_token_ids; ///< End-of-sequence token IDs
    
    /// Minimum score difference from top beam to consider (pruning)
    float beam_score_threshold = -std::numeric_limits<float>::infinity();
    
    /// Whether to return all beams or just the best
    bool return_all_beams = false;
};

/**
 * @brief Beam search decoder for transformer models
 * 
 * Beam search maintains multiple candidate sequences (beams) and expands
 * them in parallel, keeping the top-scoring sequences at each step.
 * 
 * Advantages:
 * - More deterministic than sampling
 * - Often produces higher-quality output for translation tasks
 * 
 * Disadvantages:
 * - Less diverse output
 * - More computationally expensive
 * - May produce repetitive or generic text
 */
class BeamSearcher {
public:
    /**
     * @brief Construct a beam searcher with a transformer model
     * @param model Pointer to the transformer model (must outlive BeamSearcher)
     */
    explicit BeamSearcher(models::TransformerLM* model);
    
    /**
     * @brief Perform beam search to generate sequences
     * @param prompt_ids Input prompt token IDs [Batch, PromptLen]
     * @param params Beam search parameters
     * @return Best sequence for each batch item [Batch, TotalLen]
     * 
     * Note: If params.return_all_beams is true, returns [Batch * NumBeams, TotalLen]
     */
    Tensor search(const Tensor& prompt_ids, const BeamSearchParams& params);
    
    /**
     * @brief Get scores for all completed sequences
     * Only valid after search() is called
     * @return Vector of (sequence, score) pairs per batch element
     */
    const std::vector<std::vector<std::pair<std::vector<int64_t>, float>>>& 
    get_all_sequences() const { return finished_sequences_; }
    
    /**
     * @brief Get the underlying model
     */
    models::TransformerLM* model() { return model_; }

private:
    models::TransformerLM* model_;
    
    // Storage for finished sequences and their scores
    std::vector<std::vector<std::pair<std::vector<int64_t>, float>>> finished_sequences_;
    
    /// Check if a token is an EOS token
    bool is_eos(int64_t token_id, const BeamSearchParams& params) const;
    
    /// Apply length penalty to a score
    float apply_length_penalty(float score, int64_t length, float penalty) const;
};

} // namespace vesper::generation
