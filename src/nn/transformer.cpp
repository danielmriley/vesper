#include <vesper/nn/transformer.h>
#include <vesper/nn/functional.h>
#include <vesper/core/slice.h>
#include <vesper/core/factories.h>
#include <vesper/core/macros.h>
#include <cmath>

namespace vesper::nn {

// --- MLP ---

MLP::MLP(int embed_dim, float dropout) 
    : c_fc(embed_dim, 4 * embed_dim),
      c_proj(4 * embed_dim, embed_dim),
      dropout_(dropout) {
    register_module("c_fc", &c_fc);
    register_module("c_proj", &c_proj);
}

Tensor MLP::forward(const Tensor& x_in) {
    Tensor x = x_in; // Copy handle to allow reassignment
    x = c_fc(x);
    x = functional::gelu(x);
    x = c_proj(x);
    if (dropout_ > 0.0f && training_) {
        x = functional::dropout(x, dropout_, true);
    }
    return x;
}

// --- KVCache ---

KVCache::KVCache(int batch_size, int num_heads, int max_seq_len, int head_dim, Device device)
    : max_seq_len_(max_seq_len) {
    std::vector<int64_t> shape = {
        (int64_t)batch_size, 
        (int64_t)num_heads, 
        (int64_t)max_seq_len, 
        (int64_t)head_dim
    };
    k_cache_ = vesper::zeros(shape, DType::Float32, device);
    v_cache_ = vesper::zeros(shape, DType::Float32, device);
    current_len_ = 0;
}

void KVCache::reset() {
    current_len_ = 0;
    // We don't strictly need to zero out the tensors, but we can if we want safety.
    // For performance, we usually just reset the length.
    // But let's zero them to be safe and match "reset" semantics.
    Tensor zero_k = vesper::zeros(k_cache_.shape(), k_cache_.dtype(), k_cache_.device());
    Tensor zero_v = vesper::zeros(v_cache_.shape(), v_cache_.dtype(), v_cache_.device());
    k_cache_.copy_from(zero_k);
    v_cache_.copy_from(zero_v);
}

std::pair<Tensor, Tensor> KVCache::update(const Tensor& new_k, const Tensor& new_v, int start_pos) {
    // new_k, new_v: [Batch, Heads, SeqLen, HeadDim]
    // We write to [:, :, start_pos : start_pos + SeqLen, :]
    
    int64_t seq_len = new_k.shape()[2];

    std::vector<IndexSelector> selectors = {
        Slice(), 
        Slice(), 
        Slice(start_pos, start_pos + seq_len), 
        Slice()
    };

    // Update K
    Tensor k_slot = k_cache_.index(selectors);
    k_slot.copy_from(new_k);

    // Update V
    Tensor v_slot = v_cache_.index(selectors);
    v_slot.copy_from(new_v);

    current_len_ = std::max(current_len_, start_pos + (int)seq_len);

    // Return views of the active context: [:, :, 0:current_len_, :]
    // Note: The chapter says return 0 to start_pos + seq_len.
    // If we are doing random access updates, current_len_ might be larger.
    // But usually we append.
    // Let's return up to current_len_.
    
    std::vector<IndexSelector> view_selectors = {
        Slice(),
        Slice(),
        Slice(0, current_len_),
        Slice()
    };

    return {k_cache_.index(view_selectors), v_cache_.index(view_selectors)};
}

void KVCache::reorder(const std::vector<int64_t>& src_rows) {
    // Cache layout: [Batch, Heads, max_seq_len, head_dim]; axis 0 is the beam dim.
    // src_rows[i] = source beam index whose past KV context destination row i takes.
    int64_t batch_size = k_cache_.shape()[0];
    VESPER_CHECK(static_cast<int64_t>(src_rows.size()) == batch_size,
                 "KVCache::reorder: src_rows size must equal cache batch dimension");

    // Snapshot the pre-reorder state so overlapping source/destination rows
    // (e.g. two beams forking from the same parent) read the original values.
    Tensor k_src = k_cache_.clone();
    Tensor v_src = v_cache_.clone();

    for (int64_t i = 0; i < batch_size; ++i) {
        int64_t src = src_rows[i];
        if (src == i) continue;  // Row already in place.
        VESPER_CHECK(src >= 0 && src < batch_size,
                     "KVCache::reorder: source beam index out of range");

        std::vector<IndexSelector> dst_sel = {Slice(i, i + 1), Slice(), Slice(), Slice()};
        std::vector<IndexSelector> src_sel = {Slice(src, src + 1), Slice(), Slice(), Slice()};

        k_cache_.index(dst_sel).copy_from(k_src.index(src_sel));
        v_cache_.index(dst_sel).copy_from(v_src.index(src_sel));
    }
}

// --- MultiHeadAttention ---

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads, float dropout, bool use_rope)
    : c_attn(embed_dim, 3 * embed_dim),
      c_proj(embed_dim, embed_dim),
      n_head(num_heads),
      n_embd(embed_dim),
      dropout_(dropout),
      use_rope_(use_rope) {
    register_module("c_attn", &c_attn);
    register_module("c_proj", &c_proj);
}

Tensor MultiHeadAttention::forward(const Tensor& x) {
    return forward(x, false);
}

Tensor MultiHeadAttention::forward(const Tensor& x_in, bool causal) {
    Tensor x = x_in;
    // x: [B, T, C]
    auto B = x.shape()[0];
    auto T = x.shape()[1];
    auto C = x.shape()[2];

    // Calculate query, key, values for all heads in batch and move head forward to be the batch dim
    // q, k, v shape: (B, n_head, T, hs)

    Tensor qkv = c_attn(x); // [B, T, 3*C]

    // Split qkv into q, k, v
    // We use index() to slice along the last dimension
    Tensor q = qkv.index({Slice(), Slice(), Slice(0, C)});
    Tensor k = qkv.index({Slice(), Slice(), Slice(C, 2 * C)});
    Tensor v = qkv.index({Slice(), Slice(), Slice(2 * C, 3 * C)});

    // Reshape and transpose to get heads
    // [B, T, C] -> [B, T, n_head, C/n_head] -> [B, n_head, T, C/n_head]
    int head_dim = C / n_head;
    
    q = q.contiguous().view({B, T, n_head, head_dim}).transpose(1, 2).contiguous();
    k = k.contiguous().view({B, T, n_head, head_dim}).transpose(1, 2).contiguous();
    v = v.contiguous().view({B, T, n_head, head_dim}).transpose(1, 2).contiguous();

    // Apply RoPE only if enabled
    if (use_rope_) {
        Tensor freqs = functional::compute_rope_frequencies(T, head_dim, 0, 10000.0f, q.device());
        q = functional::apply_rotary_emb(q, freqs);
        k = functional::apply_rotary_emb(k, freqs);
    }

    // Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    // Pass training_ directly to attention dropout
    float attn_dropout = training_ ? dropout_ : 0.0f;
    Tensor y = functional::scaled_dot_product_attention(q, k, v, causal, attn_dropout);

    // Re-assemble all head outputs side by side
    // y: [B, n_head, T, head_dim]
    // transpose: [B, T, n_head, head_dim]
    // contiguous().view: [B, T, C]
    y = y.transpose(1, 2).reshape({B, T, C});

    // Output projection
    y = c_proj(y);
    
    if (dropout_ > 0.0f && training_) {
        y = functional::dropout(y, dropout_, true);
    }

    return y;
}

Tensor MultiHeadAttention::forward(const Tensor& x_in, KVCache* cache, int start_pos) {
    Tensor x = x_in;
    auto B = x.shape()[0];
    auto T = x.shape()[1];
    auto C = x.shape()[2];

    Tensor qkv = c_attn(x); 

    Tensor q = qkv.index({Slice(), Slice(), Slice(0, C)});
    Tensor k = qkv.index({Slice(), Slice(), Slice(C, 2 * C)});
    Tensor v = qkv.index({Slice(), Slice(), Slice(2 * C, 3 * C)});

    int head_dim = C / n_head;
    
    q = q.contiguous().view({B, T, n_head, head_dim}).transpose(1, 2).contiguous();
    k = k.contiguous().view({B, T, n_head, head_dim}).transpose(1, 2).contiguous();
    v = v.contiguous().view({B, T, n_head, head_dim}).transpose(1, 2).contiguous();

    // Apply RoPE only if enabled
    if (use_rope_) {
        Tensor freqs = functional::compute_rope_frequencies(T, head_dim, start_pos, 10000.0f, q.device());
        q = functional::apply_rotary_emb(q, freqs);
        k = functional::apply_rotary_emb(k, freqs);
    }

    if (cache) {
        std::pair<Tensor, Tensor> kv_updated = cache->update(k, v, start_pos);
        k = kv_updated.first;
        v = kv_updated.second;
    }

    // For decoding (T=1) with cache, we attend to all past tokens (which are in K, V).
    // So causal=false is appropriate as we don't need to mask future tokens (they are not in K, V).
    // However, for prefill (T > 1), we MUST use causal masking so tokens don't attend to future tokens in the prompt.
    bool use_causal_mask = (T > 1);
    
    // IMPORTANT: Only apply dropout during training, NOT during inference!
    float attn_dropout = training_ ? dropout_ : 0.0f;
    Tensor y = functional::scaled_dot_product_attention(q, k, v, use_causal_mask, attn_dropout);

    y = y.transpose(1, 2).reshape({B, T, C});

    y = c_proj(y);
    
    // Only apply output dropout during training
    if (dropout_ > 0.0f && training_) {
        y = functional::dropout(y, dropout_, true);
    }

    return y;
}

// --- TransformerBlock ---

TransformerBlock::TransformerBlock(int embed_dim, int num_heads, float dropout, bool use_rope)
    : ln1({(int64_t)embed_dim}),
      attn(embed_dim, num_heads, dropout, use_rope),
      ln2({(int64_t)embed_dim}),
      mlp(embed_dim, dropout) {
    register_module("ln1", &ln1);
    register_module("attn", &attn);
    register_module("ln2", &ln2);
    register_module("mlp", &mlp);
}

Tensor TransformerBlock::forward(const Tensor& x) {
    return forward(x, true);  // Default to causal=true for autoregressive models
}

Tensor TransformerBlock::forward(const Tensor& x_in, bool causal) {
    // Propagate training mode to member variables
    // (This is needed because member variables are separate from registered modules)
    attn.train(training_);
    mlp.train(training_);
    
    Tensor x = x_in;
    // x = x + attn(ln1(x))
    Tensor residual = x;
    x = ln1(x);
    x = attn.forward(x, causal);
    x = residual + x;

    // x = x + mlp(ln2(x))
    residual = x;
    x = ln2(x);
    x = mlp(x);
    x = residual + x;

    return x;
}

Tensor TransformerBlock::forward(const Tensor& x_in, KVCache* cache, int start_pos) {
    Tensor x = x_in;
    // x = x + attn(ln1(x), cache, start_pos)
    Tensor residual = x;
    x = ln1(x);
    x = attn.forward(x, cache, start_pos);
    x = residual + x;

    // x = x + mlp(ln2(x))
    residual = x;
    x = ln2(x);
    x = mlp(x);
    x = residual + x;

    return x;
}

// --- LlamaBlock ---

LlamaBlock::LlamaBlock(int embed_dim, int num_heads, float dropout, bool use_rope)
    : LlamaBlock(Config{embed_dim, num_heads, 0, dropout, 1e-5f, use_rope}) {
}

LlamaBlock::LlamaBlock(const Config& config)
    : config_(config),
      attn_norm({(int64_t)config.embed_dim}, config.rms_eps),
      attn(config.embed_dim, config.num_heads, config.dropout, config.use_rope),
      ffn_norm({(int64_t)config.embed_dim}, config.rms_eps),
      ffn(config.embed_dim, 
          config.ffn_hidden_dim > 0 
              ? config.ffn_hidden_dim 
              : SwiGLUMLP::compute_hidden_dim(config.embed_dim, 64),
          /*bias=*/false) 
{
    // Update config with computed hidden dim if it was auto
    if (config_.ffn_hidden_dim == 0) {
        config_.ffn_hidden_dim = SwiGLUMLP::compute_hidden_dim(config.embed_dim, 64);
    }
    
    register_module("attn_norm", &attn_norm);
    register_module("attn", &attn);
    register_module("ffn_norm", &ffn_norm);
    register_module("ffn", &ffn);
}

Tensor LlamaBlock::forward(const Tensor& x) {
    return forward(x, true);  // Default to causal=true for autoregressive models
}

Tensor LlamaBlock::forward(const Tensor& x_in, bool causal) {
    // Propagate training mode to submodules
    attn.train(training_);
    ffn.train(training_);
    
    Tensor x = x_in;
    
    // Pre-norm attention with residual
    Tensor residual = x;
    x = attn_norm.forward(x);
    x = attn.forward(x, causal);
    x = residual + x;
    
    // Pre-norm FFN with residual
    residual = x;
    x = ffn_norm.forward(x);
    x = ffn.forward(x);
    x = residual + x;
    
    return x;
}

Tensor LlamaBlock::forward(const Tensor& x_in, KVCache* cache, int start_pos) {
    Tensor x = x_in;
    
    // Pre-norm attention with residual + KV cache
    Tensor residual = x;
    x = attn_norm.forward(x);
    x = attn.forward(x, cache, start_pos);
    x = residual + x;
    
    // Pre-norm FFN with residual
    residual = x;
    x = ffn_norm.forward(x);
    x = ffn.forward(x);
    x = residual + x;
    
    return x;
}

} // namespace vesper::nn
