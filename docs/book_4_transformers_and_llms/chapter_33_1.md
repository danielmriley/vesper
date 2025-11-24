```markdown

# Chapter 33.1: Efficient Inference (KV Cache)

## 1. Introduction

When generating text token-by-token (autoregressive inference), the model attends to all previous tokens. Re-computing the Key and Value vectors for the entire history at every step is wasteful ($O(N^2)$).

**KV Caching** allows us to compute only the new token's Q, K, V, and append the new K, V to a cache. This reduces complexity to $O(N)$.

## 2. Mechanism

### Without Cache (Step $t$)
-   Input: Sequence $[x_0, ..., x_t]$
-   Compute $K, V$ for all $t+1$ tokens.
-   Compute Attention.

### With Cache (Step $t$)
-   Input: Only new token $[x_t]$
-   Cache: Contains $K_{0:t-1}, V_{0:t-1}$
-   Compute $K_t, V_t$ for just $x_t$.
-   Concatenate: $K_{all} = [K_{cache}, K_t]$.
-   Update Cache.
-   Compute Attention using $Q_t$ and $K_{all}, V_{all}$.

## 3. Detailed Implementation Design

### The `KVCache` Class
Instead of a simple struct, we implement a class to manage the pre-allocated memory and update logic.

```cpp
class KVCache {
public:
    // Allocates the full buffer on device
    KVCache(int batch_size, int num_heads, int max_seq_len, int head_dim, Device device);

    // Updates cache with new k/v and returns views of the full active context
    // new_k, new_v: [Batch, Heads, 1, HeadDim]
    // start_pos: The position in the sequence to write to
    std::pair<Tensor, Tensor> update(const Tensor& new_k, const Tensor& new_v, int start_pos);

    int get_max_seq_len() const { return max_seq_len_; }

private:
    Tensor k_cache_; // Shape: [Batch, Heads, MaxSeqLen, HeadDim]
    Tensor v_cache_; // Shape: [Batch, Heads, MaxSeqLen, HeadDim]
    int max_seq_len_;
};
```

### Memory Management Strategy
1.  **Pre-allocation**: We allocate `k_cache_` and `v_cache_` with `MaxSeqLen` (e.g., 2048 or 4096) during initialization. This avoids expensive `malloc` calls during the generation loop.
2.  **In-Place Update**: The `update` method uses tensor slicing to write `new_k` and `new_v` directly into the pre-allocated memory at `[:, :, start_pos, :]`.
3.  **View Return**: The method returns a **view** (not a copy) of the cache from index `0` to `start_pos + 1`. This view is passed to the attention mechanism.

### Integration with `MultiHeadAttention`

The `forward` method needs to handle both the "prefill" phase (processing the prompt) and the "decode" phase (token-by-token generation).

```cpp
// In MultiHeadAttention
Tensor forward(Tensor x, KVCache* cache = nullptr, int start_pos = 0) {
    // 1. Project Q, K, V
    auto [q, k, v] = project_qkv(x); // q,k,v are [Batch, Heads, SeqLen, HeadDim]

    // 2. Apply Rotary Embeddings (RoPE)
    // Critical: Use 'start_pos' to generate correct frequencies for the new tokens.
    apply_rope(q, k, start_pos);

    // 3. Cache Management
    if (cache) {
        // Write new k, v to cache and get back the full context view
        // k becomes [Batch, Heads, TotalContext, HeadDim]
        std::tie(k, v) = cache->update(k, v, start_pos);
    }

    // 4. Scaled Dot-Product Attention
    // If caching, Q is [B, H, 1, D] and K is [B, H, Context, D]
    // Output is [B, H, 1, D]
    return scaled_dot_product_attention(q, k, v);
}
```

### Positional Embeddings (RoPE) Interaction
It is vital to apply RoPE **before** caching.
-   The cache stores the *rotated* keys.
-   When a new token comes in, we rotate it based on its absolute position (`start_pos`).
-   We then attend against the previously rotated keys in the cache.
-   This works because RoPE is a relative encoding applied via absolute positions.

## 4. Usage Example



```cpp
// Initial prompt
Tensor input = tokenize("The quick brown");
auto [logits, cache] = model(input); 

// Generation loop
for (int i = 0; i < 20; ++i) {
    Tensor next_token = sample(logits);
    
    // Pass only the new token + cache
    auto result = model(next_token, cache);
    
    logits = result.first;
    cache = result.second; // Update cache
}
```

## 5. Testing Strategy

1.  **Equivalence**: Verify that the output of the cached forward pass (step-by-step) is *identical* (within float tolerance) to the output of a full forward pass of the concatenated sequence.


```