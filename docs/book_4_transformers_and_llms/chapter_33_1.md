# Chapter 33.1: Building a Transformer Block

## 1. Introduction

This chapter combines the components from previous chapters—LayerNorm, Multi-Head Attention,
and Feed-Forward Networks—into a complete Transformer block. We focus on the decoder-only
architecture used in GPT, Llama, and other autoregressive language models.

**Key Design Choices**:
- **Pre-Norm Architecture**: LayerNorm before attention/FFN (better gradient flow)
- **Combined QKV Projection**: Single linear layer for efficiency
- **GPT-2 Style FFN**: 4x hidden expansion with GELU activation

---

## 2. Architecture Overview

### Pre-Norm vs Post-Norm

| Aspect | Pre-Norm (Modern) | Post-Norm (Original) |
|--------|-------------------|----------------------|
| Formula | `x + Attn(LN(x))` | `LN(x + Attn(x))` |
| Gradient Flow | Better for deep nets | Can struggle at 100+ layers |
| Used In | GPT-2, Llama, Mistral | BERT, Original Transformer |

We implement Pre-Norm:

```
Input x ∈ ℝ^(B×S×E)

Sub-layer 1 (Self-Attention):
    h₁ = x + Dropout(MHA(LayerNorm(x)))

Sub-layer 2 (Feed-Forward):
    h₂ = h₁ + Dropout(FFN(LayerNorm(h₁)))

Output: h₂ ∈ ℝ^(B×S×E)
```

### Data Flow Diagram

```
Input x
    │
    ├──────────────────────┐
    │                      │ (Residual)
    ▼                      │
┌─────────┐                │
│LayerNorm│                │
└────┬────┘                │
     ▼                     │
┌─────────────────────┐    │
│Multi-Head Attention │    │
└─────────┬───────────┘    │
          ▼                │
     ┌────────┐            │
     │Dropout │            │
     └────┬───┘            │
          ▼                │
        (+)◄───────────────┘
          │
          ├──────────────────────┐
          │                      │ (Residual)
          ▼                      │
    ┌─────────┐                  │
    │LayerNorm│                  │
    └────┬────┘                  │
         ▼                       │
    ┌─────────┐                  │
    │   FFN   │                  │
    └────┬────┘                  │
         ▼                       │
    ┌────────┐                   │
    │Dropout │                   │
    └────┬───┘                   │
         ▼                       │
       (+)◄──────────────────────┘
         │
         ▼
      Output
```

---

## 3. Multi-Head Attention Module

### Design

The MHA module combines projection, attention, and output into a single class:

```cpp
class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(int embed_dim, int num_heads, float dropout = 0.0f);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, bool causal);

private:
    Linear c_attn;   // Combined Q,K,V: (E) → (3E)
    Linear c_proj;   // Output: (E) → (E)
    int n_head;
    int n_embd;
    float dropout_;
};
```

### Forward Pass Implementation

```cpp
Tensor MultiHeadAttention::forward(const Tensor& x, bool causal) {
    auto [B, T, C] = extract_dims(x);  // Batch, SeqLen, EmbedDim
    int head_dim = C / n_head;

    // 1. Project to Q, K, V (fused for efficiency)
    Tensor qkv = c_attn(x);  // [B, T, 3*C]

    // 2. Split into Q, K, V
    Tensor q = qkv.index({Slice(), Slice(), Slice(0, C)});
    Tensor k = qkv.index({Slice(), Slice(), Slice(C, 2*C)});
    Tensor v = qkv.index({Slice(), Slice(), Slice(2*C, 3*C)});

    // 3. Reshape for multi-head: [B, T, C] → [B, H, T, D]
    q = q.view({B, T, n_head, head_dim}).transpose(1, 2);
    k = k.view({B, T, n_head, head_dim}).transpose(1, 2);
    v = v.view({B, T, n_head, head_dim}).transpose(1, 2);

    // 4. Scaled dot-product attention
    Tensor y = functional::scaled_dot_product_attention(q, k, v, causal, dropout_);

    // 5. Merge heads: [B, H, T, D] → [B, T, C]
    y = y.transpose(1, 2).reshape({B, T, C});

    // 6. Output projection
    return c_proj(y);
}
```

### Why Fused QKV Projection?

| Approach | Memory Reads | Kernel Launches |
|----------|--------------|-----------------|
| Separate W_Q, W_K, W_V | 3 | 3 |
| Fused c_attn | 1 | 1 |

Single fused projection is ~2x faster for the projection step.

---

## 4. Feed-Forward Network (MLP)

### Architecture

Standard transformer FFN with 4x expansion:

```
Input x ∈ ℝ^E
    │
    ▼
Linear(E → 4E)
    │
    ▼
  GELU
    │
    ▼
Linear(4E → E)
    │
    ▼
 Dropout
    │
    ▼
Output ∈ ℝ^E
```

### Implementation

```cpp
class MLP : public Module {
public:
    MLP(int embed_dim, float dropout = 0.0f);
    Tensor forward(const Tensor& x) override;

private:
    Linear c_fc;     // (E) → (4E)
    Linear c_proj;   // (4E) → (E)
    float dropout_;
};

Tensor MLP::forward(const Tensor& x) {
    Tensor h = c_fc(x);
    h = functional::gelu(h);
    h = c_proj(h);
    if (dropout_ > 0.0f) {
        h = functional::dropout(h, dropout_, is_training());
    }
    return h;
}
```

### Variants

| Model | FFN Ratio | Activation | Notes |
|-------|-----------|------------|-------|
| GPT-2 | 4x | GELU | Standard |
| Llama | 2.67x (8/3) | SiLU | With gate |
| Mistral | 2.67x | SiLU | SwiGLU variant |

---

## 5. Complete Transformer Block

### Implementation

```cpp
class TransformerBlock : public Module {
public:
    TransformerBlock(int embed_dim, int num_heads, float dropout = 0.0f);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, bool causal);

private:
    LayerNorm ln1, ln2;
    MultiHeadAttention attn;
    MLP mlp;
};

Tensor TransformerBlock::forward(const Tensor& x, bool causal) {
    // Self-attention with residual
    Tensor h = x + attn.forward(ln1(x), causal);
    
    // FFN with residual
    h = h + mlp(ln2(h));
    
    return h;
}
```

### Parameter Count Formula

For a block with embed_dim $E$ and num_heads $H$:

| Component | Parameters |
|-----------|------------|
| c_attn (QKV) | $E \times 3E + 3E$ |
| c_proj (Attn out) | $E \times E + E$ |
| c_fc (FFN up) | $E \times 4E + 4E$ |
| c_proj (FFN down) | $4E \times E + E$ |
| ln1 | $2E$ |
| ln2 | $2E$ |
| **Total** | $12E^2 + 13E$ |

**Example (GPT-2 Small, E=768)**:
$12 \times 768^2 + 13 \times 768 = 7,077,888 + 9,984 = 7,087,872$ params per block

---

## 6. Usage Examples

### Basic Usage

```cpp
auto block = nn::TransformerBlock(768, 12);  // GPT-2 Small config

Tensor x = randn({1, 1024, 768});   // [B, S, E]
Tensor out = block.forward(x, /*causal=*/true);
// out.shape() == {1, 1024, 768}
```

### Stacking Blocks

```cpp
std::vector<TransformerBlock> blocks;
for (int i = 0; i < 12; ++i) {
    blocks.emplace_back(768, 12, 0.1f);
}

Tensor h = x;
for (auto& block : blocks) {
    h = block.forward(h, true);
}
```

### With Embeddings (Full Model Sketch)

```cpp
class GPT : public Module {
    Embedding wte;  // Token embeddings
    Embedding wpe;  // Position embeddings
    std::vector<TransformerBlock> blocks;
    LayerNorm ln_f;
    Linear lm_head;

    Tensor forward(const Tensor& input_ids) {
        auto [B, T] = extract_dims(input_ids);
        
        Tensor tok_emb = wte(input_ids);
        Tensor pos_emb = wpe(arange(T));
        Tensor h = tok_emb + pos_emb;
        
        for (auto& block : blocks) {
            h = block.forward(h, /*causal=*/true);
        }
        
        h = ln_f(h);
        return lm_head(h);  // Logits: [B, T, vocab_size]
    }
};
```

---

## 7. Comprehensive Testing Strategy

### 7.1 Shape Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_output_shape` | Input (B,S,E) | Output (B,S,E) |
| `test_variable_seq_len` | S=1, S=512, S=2048 | Shape preserved |
| `test_batch_independence` | B=1 vs B=32 | Same per-sample output |

### 7.2 Parameter Count Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_param_count_small` | E=64, H=4 | Match formula |
| `test_param_count_gpt2` | E=768, H=12 | 7,097,856 |
| `test_param_count_gpt2_medium` | E=1024, H=16 | 12,632,064 |

### 7.3 Forward Pass Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_forward_no_nan` | Random input | No NaN/Inf in output |
| `test_forward_determinism` | Same input twice | Identical output (no dropout) |
| `test_causal_vs_bidirectional` | Compare outputs | Different when S>1 |

### 7.4 Gradient Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_gradient_exists` | Backward pass | All params have gradients |
| `test_gradient_finite` | Check values | No NaN/Inf gradients |
| `test_gradient_magnitude` | Analyze range | Reasonable scale |

### 7.5 Overfitting Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_overfit_single_batch` | Train on 1 sample | Loss → 0 |
| `test_overfit_convergence` | 100 steps | Final loss < 0.01 |

### 7.6 Residual Connection Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_residual_identity` | Zero-init attn/FFN output | h_out ≈ h_in |
| `test_residual_gradient_flow` | Deep stack gradient | Non-vanishing |

### 7.7 Consistency Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_cpu_vs_hip` | Same input, both devices | `max_diff < 1e-4` |
| `test_float16_forward` | Half precision | Reasonable output |

### 7.8 Module Registration Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_named_parameters` | All params named | ln1.*, attn.*, etc. |
| `test_to_device` | Move to HIP | All params on HIP |
| `test_train_eval_mode` | Toggle mode | Dropout behavior changes |

---

## 8. Common Model Configurations

| Model | E | H | Layers | FFN | Params/Block |
|-------|---|---|--------|-----|--------------|
| GPT-2 Small | 768 | 12 | 12 | 4x | 7.1M |
| GPT-2 Medium | 1024 | 16 | 24 | 4x | 12.6M |
| GPT-2 Large | 1280 | 20 | 36 | 4x | 19.7M |
| Llama-7B | 4096 | 32 | 32 | 2.67x | ~200M |

---

## 9. Performance Considerations

| Optimization | Impact | Implementation |
|--------------|--------|----------------|
| Fused QKV | ~2x proj speed | Single Linear for Q,K,V |
| Pre-Norm | Better gradients | LN before attention |
| Memory layout | 10-20% | Contiguous tensors |
| Dropout fusion | Minor | Fuse with attention |

---

## 10. References

1. Vaswani et al. "Attention Is All You Need" (2017)
2. Radford et al. "Language Models are Unsupervised Multitask Learners" (2019) — GPT-2
3. Xiong et al. "On Layer Normalization in the Transformer Architecture" (2020)
4. Touvron et al. "LLaMA: Open and Efficient Foundation Language Models" (2023)
