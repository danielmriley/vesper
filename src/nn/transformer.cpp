#include <vesper/nn/transformer.h>
#include <vesper/nn/functional.h>
#include <vesper/core/slice.h>
#include <vesper/core/factories.h>
#include <cmath>

namespace vesper::nn {

// --- MLP ---

MLP::MLP(int embed_dim, float dropout) 
    : c_fc(embed_dim, 4 * embed_dim),
      c_proj(4 * embed_dim, embed_dim),
      dropout_(dropout) {
    register_module("c_fc", std::make_shared<Linear>(c_fc));
    register_module("c_proj", std::make_shared<Linear>(c_proj));
}

Tensor MLP::forward(const Tensor& x_in) {
    Tensor x = x_in; // Copy handle to allow reassignment
    x = c_fc(x);
    x = functional::gelu(x);
    x = c_proj(x);
    if (dropout_ > 0.0) {
        x = functional::dropout(x, dropout_, true); // Assuming training=true
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

    // Return views of the active context: [:, :, 0:start_pos+seq_len, :]
    std::vector<IndexSelector> view_selectors = {
        Slice(),
        Slice(),
        Slice(0, start_pos + seq_len),
        Slice()
    };

    return {k_cache_.index(view_selectors), v_cache_.index(view_selectors)};
}

// --- MultiHeadAttention ---

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads, float dropout)
    : c_attn(embed_dim, 3 * embed_dim),
      c_proj(embed_dim, embed_dim),
      n_head(num_heads),
      n_embd(embed_dim),
      dropout_(dropout) {
    register_module("c_attn", std::make_shared<Linear>(c_attn));
    register_module("c_proj", std::make_shared<Linear>(c_proj));
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

    // Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    Tensor y = functional::scaled_dot_product_attention(q, k, v, causal, dropout_);

    // Re-assemble all head outputs side by side
    // y: [B, n_head, T, head_dim]
    // transpose: [B, T, n_head, head_dim]
    // contiguous().view: [B, T, C]
    y = y.transpose(1, 2).reshape({B, T, C});

    // Output projection
    y = c_proj(y);
    
    if (dropout_ > 0.0) {
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

    // TODO: Apply RoPE here
    // apply_rope(q, k, start_pos);

    if (cache) {
        std::pair<Tensor, Tensor> kv_updated = cache->update(k, v, start_pos);
        k = kv_updated.first;
        v = kv_updated.second;
    }

    // For decoding (T=1) with cache, we attend to all past tokens (which are in K, V).
    // So causal=false is appropriate as we don't need to mask future tokens (they are not in K, V).
    // However, for prefill (T > 1), we MUST use causal masking so tokens don't attend to future tokens in the prompt.
    bool use_causal_mask = (T > 1);
    Tensor y = functional::scaled_dot_product_attention(q, k, v, use_causal_mask, dropout_);

    y = y.transpose(1, 2).reshape({B, T, C});

    y = c_proj(y);
    
    if (dropout_ > 0.0) {
        y = functional::dropout(y, dropout_, true);
    }

    return y;
}

// --- TransformerBlock ---

TransformerBlock::TransformerBlock(int embed_dim, int num_heads, float dropout)
    : ln1({(int64_t)embed_dim}),
      attn(embed_dim, num_heads, dropout),
      ln2({(int64_t)embed_dim}),
      mlp(embed_dim, dropout) {
    register_module("ln1", std::make_shared<LayerNorm>(ln1));
    register_module("attn", std::make_shared<MultiHeadAttention>(attn));
    register_module("ln2", std::make_shared<LayerNorm>(ln2));
    register_module("mlp", std::make_shared<MLP>(mlp));
}

Tensor TransformerBlock::forward(const Tensor& x) {
    return forward(x, false);
}

Tensor TransformerBlock::forward(const Tensor& x_in, bool causal) {
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

} // namespace vesper::nn
