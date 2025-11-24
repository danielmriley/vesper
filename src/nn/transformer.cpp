#include <vesper/nn/transformer.h>
#include <vesper/nn/functional.h>
#include <vesper/core/slice.h>
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
    
    q = q.contiguous().view({B, T, n_head, head_dim}).transpose(1, 2);
    k = k.contiguous().view({B, T, n_head, head_dim}).transpose(1, 2);
    v = v.contiguous().view({B, T, n_head, head_dim}).transpose(1, 2);

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

} // namespace vesper::nn
