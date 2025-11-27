#pragma once
#include <vesper/core/tensor.h>
#include <vesper/core/device.h>
#include <memory>

namespace vesper::nn {

/**
 * @brief RoPEFrequencies - Precomputed Rotary Positional Embedding frequencies
 * 
 * This class precomputes and caches the cos/sin values for RoPE, which are
 * position-dependent but model-independent. The frequencies follow:
 *   theta_i = base^(-2i/dim) for i in [0, dim/2)
 *   angle(m, i) = m * theta_i
 * 
 * Storage format: [max_seq_len, head_dim/2, 2] where last dim is [cos, sin]
 */
class RoPEFrequencies {
public:
    /**
     * @brief Construct precomputed RoPE frequencies
     * @param max_seq_len Maximum sequence length to precompute
     * @param head_dim Dimension per attention head (must be even)
     * @param base Base frequency (10000 for Llama2, 500000 for Llama3)
     * @param device Device to store frequencies on
     */
    RoPEFrequencies(int64_t max_seq_len, int64_t head_dim, 
                    float base = 10000.0f, Device device = Device::CPU);
    
    /**
     * @brief Get frequencies for a range of positions
     * @param start_pos Starting position index
     * @param seq_len Number of positions to retrieve
     * @return Tensor of shape [seq_len, head_dim/2, 2] with [cos, sin] pairs
     */
    Tensor get(int64_t start_pos, int64_t seq_len) const;
    
    /**
     * @brief Get the raw frequency angles (before cos/sin)
     * @param start_pos Starting position index
     * @param seq_len Number of positions
     * @return Tensor of shape [seq_len, head_dim/2] with raw angles
     */
    Tensor get_angles(int64_t start_pos, int64_t seq_len) const;
    
    /**
     * @brief Update base frequency for NTK-aware scaling
     * @param new_base New base frequency
     * 
     * This recomputes all cached frequencies. Use for dynamic context
     * extension in models like Llama 3.1+.
     */
    void update_base(float new_base);
    
    /**
     * @brief Move frequencies to a different device
     * @param device Target device
     */
    void to(Device device);
    
    // Accessors
    int64_t max_seq_len() const { return max_seq_len_; }
    int64_t head_dim() const { return head_dim_; }
    float base() const { return base_; }
    Device device() const { return device_; }
    
private:
    void compute_frequencies();
    
    Tensor freqs_;           // [max_seq_len, head_dim/2, 2] with [cos, sin]
    Tensor angles_;          // [max_seq_len, head_dim/2] raw angles
    int64_t max_seq_len_;
    int64_t head_dim_;
    float base_;
    Device device_;
};

namespace functional {

/**
 * @brief Apply RoPE rotation in-place to a single tensor
 * @param x Tensor of shape [Batch, Heads, SeqLen, HeadDim]
 * @param freqs Frequencies of shape [SeqLen, HeadDim/2, 2] with [cos, sin]
 * 
 * This is an in-place operation for efficiency during inference.
 */
void apply_rope_inplace(Tensor& x, const Tensor& freqs);

/**
 * @brief Apply RoPE rotation to query and key tensors in-place
 * @param q Query tensor [Batch, Heads, SeqLen, HeadDim]
 * @param k Key tensor [Batch, Heads, SeqLen, HeadDim]
 * @param freqs Frequencies [SeqLen, HeadDim/2, 2]
 */
void apply_rope(Tensor& q, Tensor& k, const Tensor& freqs);

/**
 * @brief Apply inverse RoPE rotation (for backward pass)
 * @param x Tensor to rotate back
 * @param freqs Same frequencies used in forward pass
 * 
 * Inverse rotation negates the sin component.
 */
void apply_rope_inverse(Tensor& x, const Tensor& freqs);

} // namespace functional

// NTK-aware scaling utilities
namespace rope_utils {

/**
 * @brief Compute NTK-aware base frequency for context extension
 * @param original_max_len Original max context length the model was trained on
 * @param current_len Current/desired context length
 * @param original_base Original base frequency (e.g., 10000)
 * @param alpha Scaling factor (default 1.0)
 * @return Adjusted base frequency
 */
float compute_ntk_base(int64_t original_max_len, int64_t current_len,
                       float original_base, float alpha = 1.0f);

} // namespace rope_utils

} // namespace vesper::nn
