# Vesper Future Plans - Chapter 38.3: Fused Normalization and Residuals

## 1. Goal

Implement a fused kernel for **Add + LayerNorm** (or Add + RMSNorm). This pattern appears in every single Transformer block ("Pre-Norm" or "Post-Norm" architecture).

## 2. The Pattern

A Transformer block typically looks like:
```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```
Or the more common "Pre-Norm":
```python
x = x + attention(norm1(x))
```
Ideally, the "Residual Add" (`x + ...`) and the subsequent "Normalization" (`norm(...)`) should be one operation.

1.  **Read input `x` and residual `res`.**
2.  **Compute `x_new = x + res`.**
3.  **Compute Mean/Variance of `x_new` (Reduction).**
4.  **Normalize `x_new` and scale/shift.**
5.  **Write output.**

## 3. The Challenge

This is a "reduction-followed-by-elementwise" fusion.
*   It requires two passes over the data (one to compute stats, one to normalize), or a single pass if using specific algorithms (Welford) but that's harder to vector-load.
*   Typically, we load data into registers/shared memory, compute statistics across the feature dimension, and then normalize *before* evicting from the cache.

## 4. Implementation Plan

1.  **Fused Kernel**: Create a kernel `add_layernorm_kernel`.
2.  **Warp Reduction**: Use warp-level primitives (`__shfl_down_sync`) to compute the mean and variance of the row extremely fast without shared memory synchronization if the dimension is small enough (<= 1024).
3.  **Integration**: Update `TransformerBlock` to use this fused operator.

## 5. Why It's Next

Normalization layers are memory-bandwidth bound. They do very little compute per byte loaded. Fusing the residual addition (which is also memory bound) effectively hides the cost of one of them. This is a classic "memory bandwidth optimization" case study.
