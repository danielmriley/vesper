#pragma once

#include <vesper/nn/module.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/transformer.h>
#include <vesper/core/tensor.h>
#include <memory>

namespace vesper::nn {

/**
 * @brief Grouped Query Attention (GQA) Module
 * 
 * GQA is an efficient attention variant where multiple query heads share
 * the same key-value heads. This dramatically reduces KV cache memory
 * during inference while maintaining model quality.
 * 
 * Used in Llama 2 (70B), Llama 3, Mistral, and other modern LLMs.
 * 
 * Memory savings: (num_heads - num_kv_heads) / num_heads
 * For example, Llama 2 70B with 64 Q heads and 8 KV heads saves 87.5%
 * 
 * @see Chapter 33.5: Grouped Query Attention
 */
class GroupedQueryAttention : public Module {
public:
    /**
     * @brief Construct a Grouped Query Attention module
     * 
     * @param embed_dim Total embedding dimension (d_model)
     * @param num_heads Number of query heads
     * @param num_kv_heads Number of KV heads (must divide num_heads evenly)
     * @param max_seq_len Maximum sequence length for RoPE (default: 4096)
     * @param rope_base Base value for RoPE frequencies (default: 10000.0)
     * @param dropout Dropout probability (default: 0.0)
     * @param bias Whether to use bias in linear projections (default: false)
     */
    GroupedQueryAttention(
        int64_t embed_dim, 
        int64_t num_heads,
        int64_t num_kv_heads,
        int64_t max_seq_len = 4096,
        float rope_base = 10000.0f,
        float dropout = 0.0f,
        bool bias = false);
    
    /**
     * @brief Forward pass without KV cache (training or full-context inference)
     * @param x Input tensor [Batch, SeqLen, EmbedDim]
     * @return Output tensor [Batch, SeqLen, EmbedDim]
     */
    Tensor forward(const Tensor& x) override;
    
    /**
     * @brief Forward pass with causal masking flag
     * @param x Input tensor [Batch, SeqLen, EmbedDim]
     * @param causal Whether to apply causal masking
     * @return Output tensor [Batch, SeqLen, EmbedDim]
     */
    Tensor forward(const Tensor& x, bool causal);
    
    /**
     * @brief Forward pass with KV cache for efficient autoregressive generation
     * @param x Input tensor [Batch, SeqLen, EmbedDim]
     * @param cache KV cache to read from and update
     * @param start_pos Starting position in the sequence
     * @return Output tensor [Batch, SeqLen, EmbedDim]
     */
    Tensor forward(const Tensor& x, KVCache* cache, int64_t start_pos = 0);
    
    // Getters
    int64_t embed_dim() const { return embed_dim_; }
    int64_t num_heads() const { return num_heads_; }
    int64_t num_kv_heads() const { return num_kv_heads_; }
    int64_t head_dim() const { return head_dim_; }
    int64_t num_rep() const { return num_rep_; }
    
    /**
     * @brief Calculate KV cache memory ratio compared to standard MHA
     * @return Ratio of GQA cache size to MHA cache size
     */
    float kv_cache_ratio() const {
        return static_cast<float>(num_kv_heads_) / num_heads_;
    }
    
private:
    int64_t embed_dim_;     // Total embedding dimension
    int64_t num_heads_;     // Number of query heads
    int64_t num_kv_heads_;  // Number of KV heads (groups)
    int64_t head_dim_;      // Dimension per head
    int64_t num_rep_;       // Number of Q heads per KV head (num_heads / num_kv_heads)
    float dropout_;
    float rope_base_;
    int64_t max_seq_len_;
    
    Linear wq_;   // Query projection: [embed_dim, num_heads * head_dim]
    Linear wk_;   // Key projection:   [embed_dim, num_kv_heads * head_dim]
    Linear wv_;   // Value projection: [embed_dim, num_kv_heads * head_dim]
    Linear wo_;   // Output projection: [num_heads * head_dim, embed_dim]
};

/**
 * @brief GQA-specific KV Cache with reduced memory footprint
 * 
 * Unlike standard KVCache which stores num_heads worth of K/V,
 * GQAKVCache only stores num_kv_heads worth, providing significant
 * memory savings for models like Llama 2 70B.
 */
class GQAKVCache {
public:
    /**
     * @brief Construct a GQA KV Cache
     * 
     * @param batch_size Batch size
     * @param num_kv_heads Number of KV heads (NOT query heads!)
     * @param max_seq_len Maximum sequence length to cache
     * @param head_dim Dimension per head
     * @param device Device to allocate cache on
     */
    GQAKVCache(int64_t batch_size, int64_t num_kv_heads, 
               int64_t max_seq_len, int64_t head_dim, Device device);
    
    /**
     * @brief Update cache with new K/V and return full context
     * 
     * @param new_k New keys [Batch, KV_Heads, SeqLen, HeadDim]
     * @param new_v New values [Batch, KV_Heads, SeqLen, HeadDim]
     * @param start_pos Starting position to write at
     * @return Pair of (K, V) views covering [0, start_pos + SeqLen)
     */
    std::pair<Tensor, Tensor> update(const Tensor& new_k, const Tensor& new_v, int64_t start_pos);
    
    /**
     * @brief Reset the cache for a new sequence
     */
    void reset();
    
    /**
     * @brief Get current sequence length stored in cache
     */
    int64_t current_seq_len() const { return current_len_; }
    
    /**
     * @brief Get maximum sequence length capacity
     */
    int64_t max_seq_len() const { return max_seq_len_; }
    
    /**
     * @brief Calculate memory savings compared to MHA cache
     * @param num_query_heads Number of query heads in the model
     * @return Memory savings as a fraction (e.g., 0.875 for 87.5% savings)
     */
    static float memory_savings(int64_t num_query_heads, int64_t num_kv_heads) {
        return 1.0f - static_cast<float>(num_kv_heads) / num_query_heads;
    }
    
private:
    Tensor k_cache_;
    Tensor v_cache_;
    int64_t max_seq_len_;
    int64_t current_len_ = 0;
};

/**
 * @brief Llama-style model configuration with GQA support
 */
struct LlamaConfig {
    int64_t dim;           // Model dimension (d_model)
    int64_t n_layers;      // Number of transformer layers
    int64_t n_heads;       // Number of query heads
    int64_t n_kv_heads;    // Number of KV heads (for GQA)
    int64_t vocab_size;    // Vocabulary size
    int64_t max_seq_len;   // Maximum sequence length
    float rope_base;       // RoPE base frequency
    int64_t hidden_dim;    // FFN hidden dimension (optional, computed if 0)
    
    // Predefined configurations
    static LlamaConfig llama2_7b() {
        return {4096, 32, 32, 32, 32000, 4096, 10000.0f, 11008};  // MHA (n_heads == n_kv_heads)
    }
    
    static LlamaConfig llama2_13b() {
        return {5120, 40, 40, 40, 32000, 4096, 10000.0f, 13824};  // MHA
    }
    
    static LlamaConfig llama2_70b() {
        return {8192, 80, 64, 8, 32000, 4096, 10000.0f, 28672};   // GQA! 64Q/8KV
    }
    
    static LlamaConfig llama3_8b() {
        return {4096, 32, 32, 8, 128000, 8192, 500000.0f, 14336}; // GQA
    }
    
    static LlamaConfig llama3_70b() {
        return {8192, 80, 64, 8, 128000, 8192, 500000.0f, 28672}; // GQA
    }
    
    static LlamaConfig mistral_7b() {
        return {4096, 32, 32, 8, 32000, 8192, 10000.0f, 14336};   // GQA
    }
    
    // Check if this config uses GQA
    bool uses_gqa() const { return n_kv_heads < n_heads; }
    
    // Calculate KV cache memory ratio vs MHA
    float kv_cache_ratio() const {
        return static_cast<float>(n_kv_heads) / n_heads;
    }
};

} // namespace vesper::nn
