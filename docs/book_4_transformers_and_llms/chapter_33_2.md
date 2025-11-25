# Chapter 33.2: Efficient Inference with KV Cache

## 1. Introduction

Autoregressive text generation computes one token at a time, where each new token attends
to all previous tokens. Naively, this means re-computing Keys and Values for the entire
sequence at every step—an $O(N^2)$ operation that becomes prohibitively slow for long sequences.

**KV Caching** eliminates this redundancy by storing previously computed K and V tensors,
reducing per-step complexity to $O(N)$.

### Performance Impact

| Sequence Length | Without Cache | With Cache | Speedup |
|-----------------|---------------|------------|---------|
| 128 tokens | 128 matmuls | 1 matmul | 128x |
| 1024 tokens | 1024 matmuls | 1 matmul | 1024x |
| 4096 tokens | 4096 matmuls | 1 matmul | 4096x |

---

## 2. The Problem: Redundant Computation

### Autoregressive Generation Without Cache

At step $t$, generating token $x_t$:

```
Input:     [x₀, x₁, x₂, ..., x_{t-1}, x_t]
           ↓
Compute:   Q, K, V for ALL t+1 tokens
           ↓
Attention: Q @ K^T (full S×S matrix)
           ↓
Output:    logits for position t only (rest discarded!)
```

**Waste**: We re-compute $K_0, K_1, ..., K_{t-1}$ and $V_0, V_1, ..., V_{t-1}$ at every step,
even though they never change.

### The Insight

Keys and Values for past tokens are **immutable**—they depend only on the input token
and its position, not on future tokens. We can compute them once and reuse.

---

## 3. KV Cache Mechanism

### With Cache (Step $t$)

```
Previous Cache:  K_{cache} = [K₀, K₁, ..., K_{t-1}]  shape: (B, H, t, D)
                 V_{cache} = [V₀, V₁, ..., V_{t-1}]  shape: (B, H, t, D)

New Token:       x_t
                 ↓
Compute:         Q_t, K_t, V_t for just x_t        shape: (B, H, 1, D)
                 ↓
Concatenate:     K_{all} = concat(K_{cache}, K_t)  shape: (B, H, t+1, D)
                 V_{all} = concat(V_{cache}, V_t)  shape: (B, H, t+1, D)
                 ↓
Update Cache:    K_{cache} ← K_{all}
                 V_{cache} ← V_{all}
                 ↓
Attention:       Q_t @ K_{all}^T                   shape: (B, H, 1, t+1)
                 ↓
Output:          (B, H, 1, D) → logits
```

### Two Phases of Inference

| Phase | Input | Q shape | K/V shape | Cache Action |
|-------|-------|---------|-----------|--------------|
| **Prefill** | Full prompt (S tokens) | (B,H,S,D) | (B,H,S,D) | Initialize cache |
| **Decode** | Single token | (B,H,1,D) | (B,H,1,D) | Append to cache |

---

## 4. Implementation Design

### The `KVCache` Class

```cpp
class KVCache {
public:
    /// Pre-allocate cache buffers for maximum sequence length
    KVCache(int batch_size, int num_heads, int max_seq_len, int head_dim, 
            Device device, DType dtype = DType::Float32);

    /// Update cache with new K/V and return views of full context
    /// @param new_k, new_v: Shape (B, H, new_tokens, D)
    /// @param start_pos: Position in sequence to write to
    /// @return Pair of (K_context, V_context) views from 0 to start_pos + new_tokens
    std::pair<Tensor, Tensor> update(const Tensor& new_k, const Tensor& new_v, 
                                      int start_pos);

    /// Reset cache for new sequence
    void reset();

    /// Get current sequence length in cache
    int current_seq_len() const { return current_len_; }
    int max_seq_len() const { return max_seq_len_; }

private:
    Tensor k_cache_;  // Shape: (B, H, MaxSeqLen, D)
    Tensor v_cache_;  // Shape: (B, H, MaxSeqLen, D)
    int max_seq_len_;
    int current_len_ = 0;
};
```

### Memory Management Strategy

**Pre-allocation** is critical for performance:

```cpp
KVCache::KVCache(int B, int H, int max_seq, int D, Device device, DType dtype) 
    : max_seq_len_(max_seq), current_len_(0) {
    // Allocate full buffers upfront—no malloc during generation
    k_cache_ = zeros({B, H, max_seq, D}, dtype, device);
    v_cache_ = zeros({B, H, max_seq, D}, dtype, device);
}
```

**In-place Update** avoids copies:

```cpp
std::pair<Tensor, Tensor> KVCache::update(const Tensor& new_k, const Tensor& new_v, 
                                           int start_pos) {
    int new_tokens = new_k.shape()[2];
    
    // Write new K/V directly into pre-allocated memory
    // k_cache_[:, :, start_pos:start_pos+new_tokens, :] = new_k
    k_cache_.index_put_({Slice(), Slice(), Slice(start_pos, start_pos + new_tokens), Slice()}, 
                        new_k);
    v_cache_.index_put_({Slice(), Slice(), Slice(start_pos, start_pos + new_tokens), Slice()}, 
                        new_v);
    
    current_len_ = start_pos + new_tokens;
    
    // Return VIEW of active portion (no copy)
    Tensor k_context = k_cache_.index({Slice(), Slice(), Slice(0, current_len_), Slice()});
    Tensor v_context = v_cache_.index({Slice(), Slice(), Slice(0, current_len_), Slice()});
    
    return {k_context, v_context};
}
```

### Memory Layout

```
k_cache_ buffer:
┌─────────────────────────────────────────────────────┐
│ K₀ │ K₁ │ K₂ │ ... │ K_{t-1} │ (unused)            │
└─────────────────────────────────────────────────────┘
                                ↑
                            current_len_
```

---

## 5. Integration with Multi-Head Attention

### Modified Forward Signature

```cpp
class MultiHeadAttention : public Module {
public:
    /// Standard forward (no caching)
    Tensor forward(const Tensor& x) override;
    
    /// Forward with optional KV cache
    /// @param cache: If non-null, use/update cache
    /// @param start_pos: Position in sequence (for RoPE and cache indexing)
    Tensor forward(const Tensor& x, KVCache* cache, int start_pos = 0);
};
```

### Implementation

```cpp
Tensor MultiHeadAttention::forward(const Tensor& x, KVCache* cache, int start_pos) {
    auto [B, T, C] = extract_dims(x);
    int head_dim = C / n_head;

    // 1. Project Q, K, V
    Tensor qkv = c_attn(x);
    auto [q, k, v] = split_qkv(qkv, C);

    // 2. Reshape to (B, H, T, D)
    q = q.view({B, T, n_head, head_dim}).transpose(1, 2);
    k = k.view({B, T, n_head, head_dim}).transpose(1, 2);
    v = v.view({B, T, n_head, head_dim}).transpose(1, 2);

    // 3. Apply Rotary Embeddings (RoPE) with correct positions
    //    CRITICAL: Use start_pos for absolute position encoding
    apply_rope(q, k, start_pos);

    // 4. Cache management
    if (cache != nullptr) {
        // Append new K/V to cache, get full context
        std::tie(k, v) = cache->update(k, v, start_pos);
        // Now k, v have shape (B, H, start_pos + T, D)
    }

    // 5. Attention (causal masking handled by position)
    //    Q: (B, H, T, D), K: (B, H, Context, D)
    Tensor y = functional::scaled_dot_product_attention(q, k, v, /*causal=*/true);

    // 6. Merge heads and project
    y = y.transpose(1, 2).reshape({B, T, C});
    return c_proj(y);
}
```

### RoPE Interaction

**Critical**: Apply rotary embeddings **before** caching.

```cpp
void apply_rope(Tensor& q, Tensor& k, int start_pos) {
    int seq_len = q.shape()[2];
    
    // Generate frequencies for positions [start_pos, start_pos + seq_len)
    Tensor freqs = compute_rope_frequencies(start_pos, seq_len);
    
    // Apply rotation to Q and K
    q = apply_rotary_emb(q, freqs);
    k = apply_rotary_emb(k, freqs);
}
```

Why this works:
- Cache stores **rotated** keys at their absolute positions
- New query is rotated at its absolute position
- Dot product correctly computes relative position encoding

---

## 6. Full Inference Loop

### Prefill + Decode Pattern

```cpp
Tensor generate(Model& model, const Tensor& prompt_ids, int max_new_tokens) {
    int B = prompt_ids.shape()[0];
    int prompt_len = prompt_ids.shape()[1];
    
    // Initialize caches for each layer
    std::vector<KVCache> caches;
    for (int i = 0; i < model.num_layers(); ++i) {
        caches.emplace_back(B, model.num_heads(), model.max_seq_len(), 
                           model.head_dim(), model.device());
    }
    
    // Phase 1: Prefill - process entire prompt
    Tensor logits = model.forward(prompt_ids, caches, /*start_pos=*/0);
    Tensor next_token = sample(logits.index({Slice(), -1, Slice()}));
    
    std::vector<Tensor> generated = {next_token};
    
    // Phase 2: Decode - one token at a time
    for (int i = 0; i < max_new_tokens - 1; ++i) {
        int pos = prompt_len + i;
        
        // Forward with just the new token
        logits = model.forward(next_token.unsqueeze(1), caches, pos);
        next_token = sample(logits.squeeze(1));
        
        generated.push_back(next_token);
        
        if (is_eos(next_token)) break;
    }
    
    return stack(generated, /*dim=*/1);
}
```

---

## 7. Memory Analysis

### Cache Size Calculation

Per layer, per sequence:
$$ \text{Cache Size} = 2 \times B \times H \times S_{max} \times D \times \text{sizeof(dtype)} $$

**Example (Llama-7B, FP16)**:
- $B = 1$, $H = 32$, $S_{max} = 4096$, $D = 128$, layers = 32
- Per layer: $2 \times 1 \times 32 \times 4096 \times 128 \times 2 = 64$ MB
- Total: $32 \times 64$ MB = **2 GB** for KV cache alone

### Memory Optimization Techniques

| Technique | Memory Reduction | Trade-off |
|-----------|------------------|-----------|
| FP16 cache | 2x | Minor precision loss |
| INT8 KV quantization | 4x | Requires calibration |
| Sliding window | $W/S_{max}$x | Limited context |
| Paged attention | Dynamic | Implementation complexity |

---

## 8. Comprehensive Testing Strategy

### 8.1 Correctness Tests (Cache Equivalence)

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_cached_vs_full_forward` | Compare step-by-step cached output to full sequence forward | `max_diff < 1e-5` |
| `test_prefill_equivalence` | Prefill phase matches full forward | Exact match |
| `test_incremental_decode` | Each decode step matches sliced full forward | `max_diff < 1e-5` |

### 8.2 Cache State Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_cache_shape_after_prefill` | S tokens → cache has S entries | `current_len == S` |
| `test_cache_shape_after_decode` | Prefill + N decodes | `current_len == S + N` |
| `test_cache_reset` | Reset clears state | `current_len == 0` |
| `test_cache_overflow` | Exceed max_seq_len | Proper error/handling |

### 8.3 Position Encoding Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_rope_positions_prefill` | RoPE applied with pos 0 to S-1 | Correct frequencies |
| `test_rope_positions_decode` | RoPE applied with pos S, S+1, ... | Matches full forward |
| `test_position_invariance` | Same token at same position → same K | Exact match |

### 8.4 Memory Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_no_allocation_during_decode` | Monitor memory | No new allocations |
| `test_view_not_copy` | Cache returns views | Shared storage |
| `test_cache_memory_size` | Check actual allocation | Matches formula |

### 8.5 Multi-Layer Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_per_layer_cache` | Each layer has independent cache | Correct K/V per layer |
| `test_cache_propagation` | Caches update correctly through layers | Sequential correctness |

### 8.6 Batch Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_batch_independence` | B=4, each sequence independent | Per-sequence correctness |
| `test_variable_length_batch` | Different sequence lengths | Padding/mask handling |

### 8.7 Edge Cases

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_single_token_prompt` | S=1 prefill | Valid cache state |
| `test_max_length_generation` | Generate to max_seq_len | No crash, correct output |
| `test_empty_cache_decode` | Decode without prefill | Error or valid handling |

---

## 9. Performance Benchmarks

### Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| Prefill throughput | Tokens/second for prompt | Memory bandwidth limited |
| Decode latency | ms/token for generation | Single-digit ms |
| Memory efficiency | Actual vs theoretical cache size | ~100% |
| Cache hit rate | Views vs copies | 100% views |

### Expected Performance

| Sequence Length | Naive Decode | Cached Decode | Speedup |
|-----------------|--------------|---------------|---------|
| 128 | 8.2 ms/tok | 0.8 ms/tok | 10x |
| 512 | 32.8 ms/tok | 0.9 ms/tok | 36x |
| 2048 | 131 ms/tok | 1.1 ms/tok | 119x |

---

## 10. Common Pitfalls

1. **RoPE Position Mismatch**: Applying RoPE with wrong `start_pos` breaks generation
2. **Cache View Invalidation**: Modifying cache while view is in use
3. **Batch Size Changes**: Cache allocated for B=1, inference with B=4
4. **Memory Leaks**: Not resetting cache between sequences
5. **Causal Mask in Decode**: Must handle 1×Context attention correctly

---

## 11. References

1. Pope et al. "Efficiently Scaling Transformer Inference" (2022)
2. Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)
3. Shazeer. "Fast Transformer Decoding: One Write-Head is All You Need" (2019) — MQA/GQA
