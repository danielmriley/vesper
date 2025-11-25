# Chapter 32: The Attention Mechanism

## 1. Introduction

Attention is the fundamental innovation of the Transformer architecture. It enables dynamic
weighting of input positions based on learned relevance, replacing recurrence with parallelism.
This chapter covers the complete implementation of scaled dot-product attention and multi-head
attention, including causal masking for autoregressive models.

---

## 2. Scaled Dot-Product Attention

### Mathematical Formulation

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

**Components**:
- $Q$ (Query): What am I looking for? Shape: `(B, H, S, D)`
- $K$ (Key): What do I contain? Shape: `(B, H, S, D)`
- $V$ (Value): What do I return? Shape: `(B, H, S, D)`
- $d_k$: Head dimension (scaling factor to prevent gradient vanishing)

### Why Scale by $\sqrt{d_k}$?

Without scaling, dot products grow with dimension:
- For random vectors $q, k \in \mathbb{R}^{d_k}$ with unit variance components
- $\mathbb{E}[q \cdot k] = 0$, but $\text{Var}[q \cdot k] = d_k$

Large dot products push softmax into saturation regions where gradients vanish.
Scaling by $\sqrt{d_k}$ normalizes variance back to 1.

---

## 3. Implementation Steps

### Step-by-Step Breakdown

```
Input: Q, K, V ∈ ℝ^(B×H×S×D)

1. Compute scores:     scores = Q @ K^T           → (B, H, S, S)
2. Scale:              scores = scores / √D
3. Mask (optional):    scores[i,j] = -∞  where j > i  (causal)
4. Softmax:            probs = softmax(scores, dim=-1)
5. Dropout (optional): probs = dropout(probs, p)
6. Weighted sum:       output = probs @ V         → (B, H, S, D)
```

### API

```cpp
namespace vesper::nn::functional {
    Tensor scaled_dot_product_attention(
        const Tensor& Q,
        const Tensor& K, 
        const Tensor& V,
        bool is_causal = false,
        float dropout_p = 0.0f,
        const Tensor* attn_mask = nullptr  // Optional custom mask
    );
}
```

---

## 4. Causal Masking

### Purpose

In autoregressive models (GPT, Llama), token $t$ can only attend to tokens $\leq t$.
This prevents "cheating" by looking at future tokens during training.

### Mask Construction

```cpp
Tensor causal_mask(int seq_len, Device device) {
    // Upper triangular matrix of -inf, zeros on/below diagonal
    // mask[i][j] = 0 if j <= i, else -inf
    Tensor mask = full({seq_len, seq_len}, -INFINITY, DType::Float32, device);
    return tril(mask, 0);  // Keep lower triangle + diagonal
}
```

### Application

```cpp
// In attention computation
if (is_causal) {
    Tensor mask = causal_mask(S, scores.device());
    scores = scores + mask;  // Broadcasting: (B,H,S,S) + (S,S)
}
```

After adding mask, softmax converts $-\infty$ to probability 0.

---

## 5. Multi-Head Attention

### Concept

Instead of a single attention function, use $H$ parallel "heads" with smaller dimensions:

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_H) W^O $$

Where each head computes:
$$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

### Reshaping Logic

```
Input:  x ∈ ℝ^(B×S×E)     where E = H × D

1. Project:  Q, K, V = x @ W_Q, x @ W_K, x @ W_V    → (B, S, E)
2. Reshape:  (B, S, H*D) → (B, S, H, D)
3. Permute:  (B, S, H, D) → (B, H, S, D)            # Batch GEMM friendly
4. Attention: scaled_dot_product_attention(Q, K, V)
5. Permute:  (B, H, S, D) → (B, S, H, D)
6. Reshape:  (B, S, H, D) → (B, S, E)
7. Project:  output = reshaped @ W_O                → (B, S, E)
```

### Class Definition

```cpp
class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(int embed_dim, int num_heads, float dropout = 0.0f);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, bool causal);

private:
    Linear c_attn;   // Combined Q, K, V projection: (E) → (3*E)
    Linear c_proj;   // Output projection: (E) → (E)
    int n_head;
    int n_embd;
    float dropout_;
};
```

---

## 6. Backward Pass

### Attention Gradient Flow

Given $P = \text{softmax}(S)$ and $O = PV$:

$$ \frac{\partial L}{\partial V} = P^T \frac{\partial L}{\partial O} $$

$$ \frac{\partial L}{\partial P} = \frac{\partial L}{\partial O} V^T $$

$$ \frac{\partial L}{\partial S} = P \odot \left( \frac{\partial L}{\partial P} - \sum_j \frac{\partial L}{\partial P_j} P_j \right) $$

$$ \frac{\partial L}{\partial Q} = \frac{\partial L}{\partial S} K / \sqrt{d_k} $$

$$ \frac{\partial L}{\partial K} = \frac{\partial L}{\partial S}^T Q / \sqrt{d_k} $$

### Numerical Stability

The softmax backward requires the Jacobian-vector product, which can be computed
efficiently as shown above without materializing the full Jacobian.

---

## 7. Flash Attention (Overview)

Standard attention has $O(S^2)$ memory for storing the attention matrix.
Flash Attention fuses operations and uses tiling to reduce memory to $O(S)$.

**Key Ideas**:
1. Never materialize the full $S \times S$ attention matrix
2. Compute softmax in tiles using online softmax algorithm
3. Recompute attention in backward pass instead of storing

*Full Flash Attention implementation covered in Chapter 35.*

---

## 8. Usage Examples

### Basic Attention

```cpp
Tensor Q = randn({2, 8, 128, 64});  // [B, H, S, D]
Tensor K = randn({2, 8, 128, 64});
Tensor V = randn({2, 8, 128, 64});

// Non-causal attention (encoder-style)
Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, false);

// Causal attention (decoder-style)  
Tensor out_causal = nn::functional::scaled_dot_product_attention(Q, K, V, true);
```

### Multi-Head Attention Module

```cpp
auto mha = nn::MultiHeadAttention(512, 8);  // embed=512, heads=8

Tensor x = randn({32, 128, 512});  // [B, S, E]
Tensor y = mha.forward(x, /*causal=*/true);
```

---

## 9. Comprehensive Testing Strategy

### 9.1 Shape Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_attention_output_shape` | Q,K,V: (B,H,S,D) | Output: (B,H,S,D) |
| `test_mha_output_shape` | Input: (B,S,E) | Output: (B,S,E) |
| `test_different_seq_lens` | Q: S=64, K/V: S=128 | Cross-attention shape |

### 9.2 Causal Masking Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_causality_invariance` | Modify future tokens | Output at t unchanged |
| `test_causal_mask_values` | Check mask structure | Upper triangle = -inf |
| `test_first_token` | Token 0 attention | Only attends to self |
| `test_last_token` | Token S-1 attention | Attends to all previous |

### 9.3 Numerical Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_attention_sum_to_one` | softmax(scores) rows | sum = 1.0 ± 1e-6 |
| `test_uniform_attention` | Q=K (identical queries) | Equal weights when no mask |
| `test_large_values_stability` | Q,K with values ~100 | No inf/nan |
| `test_gradient_finite` | Backward pass | All gradients finite |

### 9.4 Correctness Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_known_values` | Hand-computed small example | Exact match |
| `test_vs_naive_impl` | Compare to for-loop impl | `max_diff < 1e-5` |
| `test_backward_finite_diff` | Numerical gradient check | `rel_diff < 1e-3` |

### 9.5 Consistency Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_cpu_vs_hip` | Same inputs, both backends | `max_diff < 1e-4` |
| `test_determinism` | Run twice with same seed | Identical output |
| `test_dtype_float16` | Half precision attention | Reasonable tolerance |

### 9.6 Edge Cases

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_seq_len_1` | S=1 (single token) | Valid output |
| `test_batch_1` | B=1 | No batch dim issues |
| `test_head_1` | H=1 (single head) | Equivalent to SDPA |
| `test_very_long_seq` | S=4096 | Memory and correctness |

### 9.7 Dropout Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_dropout_zeros` | p=0.0 in eval mode | Output unchanged |
| `test_dropout_train_mode` | p=0.1 in train mode | Some weights zeroed |
| `test_dropout_scaling` | Verify 1/(1-p) scaling | Mean preserved |

---

## 10. Performance Considerations

| Optimization | Impact | Notes |
|--------------|--------|-------|
| Batch GEMM | Essential | Single kernel for Q@K^T |
| Memory layout | 10-20% | Ensure contiguous BHSD |
| Fused softmax | 5-10% | Avoid separate max/exp/sum |
| Flash Attention | 2-4x memory | For long sequences |

---

## 11. References

1. Vaswani et al. "Attention Is All You Need" (2017)
2. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
3. Rabe, Staats. "Self-Attention Does Not Need O(n²) Memory" (2021)
