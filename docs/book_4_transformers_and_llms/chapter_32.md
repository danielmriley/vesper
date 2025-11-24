
# Vesper Future Plans - Chapter 29: The Attention Mechanism

## 1. Introduction

Attention is the core mechanism of the Transformer. It allows the model to "attend" to different parts of the input sequence when computing the representation for a specific token.

## 2. Scaled Dot-Product Attention

The mathematical formula is:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

-   $Q$ (Query): What I'm looking for.
-   $K$ (Key): What I contain.
-   $V$ (Value): What I pass on.
-   $d_k$: Dimension of the key vectors (scaling factor).

## 3. Implementation Plan

We will implement a functional helper `vesper::nn::functional::scaled_dot_product_attention`.

### Steps
1.  **Matmul 1**: `scores = Q @ K.transpose(-2, -1)`
    -   Shapes: `(B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)`
2.  **Scale**: `scores = scores / sqrt(D)`
    -   **Why Scale?**: Without scaling, dot products grow large with dimension $D$, pushing softmax into regions with extremely small gradients (vanishing gradients).
3.  **Masking (Optional)**: For causal (GPT-style) attention, we must mask out future tokens. Set `scores[i, j] = -inf` where `j > i`.
4.  **Softmax**: `probs = softmax(scores, dim=-1)`
5.  **Dropout (Optional)**: Apply dropout to `probs`. This is a standard regularization technique in Transformers.
    -   `probs = dropout(probs, p=dropout_p)`
6.  **Matmul 2**: `output = probs @ V`
    -   Shapes: `(B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)`

### Multi-Head Reshaping Logic
The input to MHA is typically `(Batch, SeqLen, EmbedDim)`.
To split into heads, we reshape and permute:
1.  Reshape: `(B, S, H * D_head) -> (B, S, H, D_head)`
2.  Permute: `(B, S, H, D_head) -> (B, H, S, D_head)`
This aligns the data for the Batch GEMM `(B*H, S, D_head)`.

### Causal Masking
We need a helper to create a triangular mask.

```cpp
Tensor causal_mask(int seq_len) {
    // Returns a matrix with 0s on/below diagonal and -inf above
}
```

## 4. Usage Example

```cpp
Tensor Q = ...; // [B, H, S, D]
Tensor K = ...;
Tensor V = ...;

Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, true); // is_causal=true
```

## 5. Testing Strategy

1.  **Shape Check**: Verify output shapes match `(B, H, S, D)`.
2.  **Causality**: In a causal setting, ensure that changing the input at position $t+1$ does NOT affect the output at position $t$.
