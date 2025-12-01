/**
 * @file gqa_attention.cpp
 * @brief Implementation of Grouped Query Attention
 * 
 * Chapter 33.5: Grouped Query Attention (GQA)
 * 
 * Optimized with FUSED QKV projection:
 * - Single GEMM instead of 3 separate GEMMs
 * - Better GPU utilization due to larger output dimension
 */

#include <vesper/nn/gqa_attention.h>
#include <vesper/nn/functional.h>
#include <vesper/ops/attention_ops.h>
#include <vesper/core/factories.h>
#include <vesper/core/slice.h>
#include <stdexcept>
#include <cmath>

namespace vesper::nn {

// =============================================================================
// GroupedQueryAttention Implementation with Fused QKV
// =============================================================================

GroupedQueryAttention::GroupedQueryAttention(
    int64_t embed_dim, 
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t max_seq_len,
    float rope_base,
    float dropout,
    bool bias)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(embed_dim / num_heads),
      num_rep_(num_heads / num_kv_heads),
      dropout_(dropout),
      rope_base_(rope_base),
      max_seq_len_(max_seq_len),
      // Fused QKV: output = Q (num_heads * head_dim) + K (num_kv_heads * head_dim) + V (num_kv_heads * head_dim)
      wqkv_(embed_dim, num_heads * head_dim_, num_kv_heads * head_dim_, bias),
      wo_(num_heads * head_dim_, embed_dim, bias)
{
    // Validate configuration
    if (num_heads % num_kv_heads != 0) {
        throw std::invalid_argument(
            "GroupedQueryAttention: num_heads (" + std::to_string(num_heads) + 
            ") must be divisible by num_kv_heads (" + std::to_string(num_kv_heads) + ")");
    }
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument(
            "GroupedQueryAttention: embed_dim (" + std::to_string(embed_dim) + 
            ") must be divisible by num_heads (" + std::to_string(num_heads) + ")");
    }
    
    // Register submodules (fused QKV replaces separate wq, wk, wv)
    register_module<GroupedQueryAttention>("wqkv", &GroupedQueryAttention::wqkv_);
    register_module<GroupedQueryAttention>("wo", &GroupedQueryAttention::wo_);
}

Tensor GroupedQueryAttention::forward(const Tensor& x) {
    return forward(x, true);  // Default to causal for transformer training
}

Tensor GroupedQueryAttention::forward(const Tensor& x, bool causal) {
    // x: [B, T, C] where C = embed_dim
    int64_t B = x.shape()[0];
    int64_t T = x.shape()[1];
    int64_t C = x.shape()[2];
    
    // 1. FUSED QKV projection (single GEMM instead of 3!)
    auto [q, k, v] = wqkv_.forward_split(x);
    // q: [B, T, num_heads * head_dim]
    // k: [B, T, num_kv_heads * head_dim]
    // v: [B, T, num_kv_heads * head_dim]
    
    // 2. Reshape: [B, T, H, D] and transpose to [B, H, T, D]
    q = q.view({B, T, num_heads_, head_dim_}).transpose(1, 2).contiguous();
    k = k.view({B, T, num_kv_heads_, head_dim_}).transpose(1, 2).contiguous();
    v = v.view({B, T, num_kv_heads_, head_dim_}).transpose(1, 2).contiguous();
    
    // 3. Apply RoPE (Rotary Position Embeddings)
    Tensor freqs = functional::compute_rope_frequencies(T, head_dim_, 0, rope_base_, q.device());
    q = functional::apply_rotary_emb(q, freqs);
    k = functional::apply_rotary_emb(k, freqs);
    
    // 4. Expand K, V to match Q's head count using repeat_kv
    // [B, num_kv_heads, T, head_dim] -> [B, num_heads, T, head_dim]
    k = ops::repeat_kv(k, num_rep_);
    v = ops::repeat_kv(v, num_rep_);
    
    // 5. Scaled Dot-Product Attention
    float attn_dropout = training_ ? dropout_ : 0.0f;
    Tensor out = functional::scaled_dot_product_attention(q, k, v, causal, attn_dropout);
    
    // 6. Merge heads: [B, num_heads, T, head_dim] -> [B, T, C]
    out = out.transpose(1, 2).reshape({B, T, embed_dim_});
    
    // 7. Output projection
    out = wo_.forward(out);
    
    if (dropout_ > 0.0f && training_) {
        out = functional::dropout(out, dropout_, true);
    }
    
    return out;
}

Tensor GroupedQueryAttention::forward(const Tensor& x, KVCache* cache, int64_t start_pos) {
    // x: [B, T, C]
    int64_t B = x.shape()[0];
    int64_t T = x.shape()[1];
    
    // 1. FUSED QKV projection
    auto [q, k, v] = wqkv_.forward_split(x);
    
    // 2. Reshape and transpose
    q = q.view({B, T, num_heads_, head_dim_}).transpose(1, 2).contiguous();
    k = k.view({B, T, num_kv_heads_, head_dim_}).transpose(1, 2).contiguous();
    v = v.view({B, T, num_kv_heads_, head_dim_}).transpose(1, 2).contiguous();
    
    // 3. Apply RoPE with position offset
    Tensor freqs = functional::compute_rope_frequencies(T, head_dim_, start_pos, rope_base_, q.device());
    q = functional::apply_rotary_emb(q, freqs);
    k = functional::apply_rotary_emb(k, freqs);
    
    // 4. Update KV cache
    // Note: The cache stores num_kv_heads worth of data, not num_heads
    // For GQA-specific cache (GQAKVCache), this works directly.
    // For standard KVCache configured with num_kv_heads, this also works.
    if (cache) {
        std::pair<Tensor, Tensor> kv_updated = cache->update(k, v, start_pos);
        k = kv_updated.first;
        v = kv_updated.second;
    }
    
    // 5. Expand K, V to match Q's head count
    k = ops::repeat_kv(k, num_rep_);
    v = ops::repeat_kv(v, num_rep_);
    
    // 6. Attention
    // For decoding (T=1), no causal mask needed as we only attend to past
    // For prefill (T>1), need causal mask
    bool use_causal = (T > 1);
    float attn_dropout = training_ ? dropout_ : 0.0f;
    Tensor out = functional::scaled_dot_product_attention(q, k, v, use_causal, attn_dropout);
    
    // 7. Merge heads and project
    out = out.transpose(1, 2).reshape({B, T, embed_dim_});
    out = wo_.forward(out);
    
    if (dropout_ > 0.0f && training_) {
        out = functional::dropout(out, dropout_, true);
    }
    
    return out;
}

// =============================================================================
// GQAKVCache Implementation
// =============================================================================

GQAKVCache::GQAKVCache(int64_t batch_size, int64_t num_kv_heads, 
                       int64_t max_seq_len, int64_t head_dim, Device device)
    : max_seq_len_(max_seq_len), current_len_(0)
{
    // Allocate reduced-size cache (num_kv_heads, not num_query_heads)
    std::vector<int64_t> shape = {batch_size, num_kv_heads, max_seq_len, head_dim};
    k_cache_ = vesper::zeros(shape, DType::Float32, device);
    v_cache_ = vesper::zeros(shape, DType::Float32, device);
}

std::pair<Tensor, Tensor> GQAKVCache::update(const Tensor& new_k, const Tensor& new_v, int64_t start_pos) {
    // new_k, new_v: [Batch, KV_Heads, SeqLen, HeadDim]
    int64_t seq_len = new_k.shape()[2];
    
    // Build index selectors for the slice to update
    std::vector<IndexSelector> selectors = {
        Slice(),                              // All batches
        Slice(),                              // All KV heads
        Slice(start_pos, start_pos + seq_len),  // Position range
        Slice()                               // All head dims
    };
    
    // Update K cache
    Tensor k_slot = k_cache_.index(selectors);
    k_slot.copy_from(new_k);
    
    // Update V cache
    Tensor v_slot = v_cache_.index(selectors);
    v_slot.copy_from(new_v);
    
    // Update current length
    current_len_ = std::max(current_len_, start_pos + seq_len);
    
    // Return views of active context
    std::vector<IndexSelector> view_selectors = {
        Slice(),
        Slice(),
        Slice(0, current_len_),
        Slice()
    };
    
    return {k_cache_.index(view_selectors), v_cache_.index(view_selectors)};
}

void GQAKVCache::reset() {
    current_len_ = 0;
    // Zero out for safety (optional, but matches expected reset semantics)
    k_cache_.zero_();
    v_cache_.zero_();
}

} // namespace vesper::nn
