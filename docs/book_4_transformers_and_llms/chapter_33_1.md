
# Chapter 33.1: Building a Transformer Block

## 1. Introduction

We now combine LayerNorm, Multi-Head Attention, and Feed-Forward Networks to build a standard Transformer Block (specifically, a decoder block for GPT-style models).

## 2. Architecture

A standard Transformer Decoder Block consists of:

1. **Input** $x$
2. **Self-Attention Sub-layer**:
   $$ x = x + \text{Dropout}(\text{MHA}(\text{LayerNorm}(x))) $$
   *(Note: We use Pre-Norm architecture, which is standard for modern LLMs)*
3. **Feed-Forward Sub-layer**:
   $$ x = x + \text{Dropout}(\text{FFN}(\text{LayerNorm}(x))) $$

### Multi-Head Attention (MHA) Module

- Projects input $x$ to $Q, K, V$ using `Linear` layers.
- Splits heads (reshape + permute).
- Calls `scaled_dot_product_attention`.
- Merges heads (permute + reshape).
- Projects output using a final `Linear` layer.
- **Weight Tying**: In some architectures, the input embedding weights and the final output projection weights (unembedding) are shared. This is not part of the block itself but a model-level consideration.

### Feed-Forward Network (FFN) Module

- `Linear(dim, 4 * dim)`
- `GELU`
- `Linear(4 * dim, dim)`
- **Dropout**: Typically applied after the second linear layer.

## 3. Implementation Plan

### `nn::MultiHeadAttention`

```cpp
class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(int embed_dim, int num_heads, float dropout=0.0);
    Tensor forward(Tensor x, bool causal=false);
    
    Linear c_attn; // Combined Q,K,V projection for efficiency
    Linear c_proj; // Output projection
    int n_head;
    float dropout_;
};
```

### `nn::TransformerBlock`

```cpp
class TransformerBlock : public Module {
public:
    TransformerBlock(int embed_dim, int num_heads, float dropout=0.0);
    Tensor forward(Tensor x);
    
    LayerNorm ln1, ln2;
    MultiHeadAttention attn;
    MLP mlp; // FFN
    float dropout_;
};
```

## 4. Usage Example

```cpp
auto block = nn::TransformerBlock(768, 12); // GPT-2 Small config
Tensor x = randn({1, 1024, 768});
Tensor out = block(x);
```

## 5. Testing Strategy

1. **Parameter Count**: Verify the number of parameters matches theoretical values.
2. **Overfitting**: Try to overfit a single batch of data with one block. Loss should go to zero.
