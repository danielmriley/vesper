```markdown
# Chapter 33.5: Grouped Query Attention (GQA)

## 1. Introduction

Grouped Query Attention (GQA) is an architectural innovation that dramatically reduces the memory footprint of the KV cache during inference, while maintaining model quality. It was introduced by Google in the "GQA: Training Generalized Multi-Query Transformers from Multi-Head Checkpoints" paper and is now used in Llama 2 (70B), Llama 3, Mistral, Mixtral, and many other modern LLMs.

### The KV Cache Memory Problem

During autoregressive generation, the KV cache grows linearly with sequence length:

$$
\text{KV Cache Size} = 2 \times B \times H \times S \times D_{head} \times \text{precision}
$$

For Llama 2 70B with 64 heads, 8192 context, FP16:
$$
2 \times 1 \times 64 \times 8192 \times 128 \times 2 = 268 \text{ MB per layer}
$$
With 80 layers: **~21 GB just for KV cache!**

### Attention Variants

| Type | Notation | KV Heads | Memory | Models |
|------|----------|----------|--------|--------|
| Multi-Head Attention (MHA) | H Q, H K, H V | H | 100% | GPT-3, Llama 1 |
| Multi-Query Attention (MQA) | H Q, 1 K, 1 V | 1 | 1/H | PaLM, Falcon |
| Grouped Query Attention (GQA) | H Q, G K, G V | G | G/H | Llama 2 70B, Llama 3, Mistral |

GQA is a middle ground: multiple query heads share each key-value pair, reducing memory while preserving quality.

## 2. Mathematical Foundation

### 2.1 Standard Multi-Head Attention

$$
\text{Attention}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

Each head $i$ has its own $Q_i, K_i, V_i$ projections.

### 2.2 Grouped Query Attention

With $H$ query heads and $G$ KV groups (where $H \mod G = 0$):

$$
\text{Attention}_i = \text{softmax}\left(\frac{Q_i K_{g(i)}^T}{\sqrt{d_k}}\right) V_{g(i)}
$$

Where $g(i) = \lfloor i \cdot G / H \rfloor$ maps query head $i$ to its KV group.

**Example**: Llama 2 70B has H=64 query heads and G=8 KV groups. Each KV group is shared by 8 query heads.

### 2.3 KV Cache Savings

| Model | Query Heads (H) | KV Groups (G) | Ratio | Savings |
|-------|----------------|---------------|-------|---------|
| Llama 1 70B | 64 | 64 (MHA) | 1:1 | 0% |
| Llama 2 70B | 64 | 8 (GQA) | 8:1 | 87.5% |
| Mistral 7B | 32 | 8 | 4:1 | 75% |

## 3. Implementation Plan

### 3.1 GQA Attention Module

```cpp
// include/vesper/nn/gqa_attention.h

namespace vesper::nn {

class GroupedQueryAttention : public Module {
public:
    // embed_dim: Total embedding dimension
    // num_heads: Number of query heads
    // num_kv_heads: Number of KV heads (groups)
    // max_seq_len: Maximum sequence length for RoPE
    GroupedQueryAttention(
        int64_t embed_dim, 
        int64_t num_heads,
        int64_t num_kv_heads,
        int64_t max_seq_len = 4096,
        float rope_base = 10000.0f,
        float dropout = 0.0f);
    
    // Forward with optional KV cache
    Tensor forward(const Tensor& x, 
                   KVCache* cache = nullptr, 
                   int64_t start_pos = 0);
    
    // Getters
    int64_t num_heads() const { return num_heads_; }
    int64_t num_kv_heads() const { return num_kv_heads_; }
    int64_t head_dim() const { return head_dim_; }
    
private:
    int64_t num_heads_;      // Query heads
    int64_t num_kv_heads_;   // KV heads (groups)
    int64_t head_dim_;       // Dimension per head
    int64_t num_rep_;        // Heads per group (H / G)
    float dropout_;
    
    Linear wq_;   // Query projection
    Linear wk_;   // Key projection (smaller)
    Linear wv_;   // Value projection (smaller)
    Linear wo_;   // Output projection
    
    std::unique_ptr<RoPEFrequencies> rope_freqs_;
};

} // namespace vesper::nn
```

### 3.2 GQA Implementation

```cpp
// src/nn/gqa_attention.cpp

namespace vesper::nn {

GroupedQueryAttention::GroupedQueryAttention(
    int64_t embed_dim, 
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t max_seq_len,
    float rope_base,
    float dropout)
    : num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(embed_dim / num_heads),
      num_rep_(num_heads / num_kv_heads),
      dropout_(dropout)
{
    VESPER_CHECK(num_heads % num_kv_heads == 0,
        "num_heads must be divisible by num_kv_heads");
    VESPER_CHECK(embed_dim % num_heads == 0,
        "embed_dim must be divisible by num_heads");
    
    // Query: full size [embed_dim, num_heads * head_dim]
    wq_ = register_module("wq", Linear(embed_dim, num_heads_ * head_dim_, false));
    
    // Key/Value: reduced size [embed_dim, num_kv_heads * head_dim]
    wk_ = register_module("wk", Linear(embed_dim, num_kv_heads_ * head_dim_, false));
    wv_ = register_module("wv", Linear(embed_dim, num_kv_heads_ * head_dim_, false));
    
    // Output: full size
    wo_ = register_module("wo", Linear(num_heads_ * head_dim_, embed_dim, false));
    
    // RoPE
    rope_freqs_ = std::make_unique<RoPEFrequencies>(
        max_seq_len, head_dim_, rope_base);
}

Tensor GroupedQueryAttention::forward(const Tensor& x, 
                                       KVCache* cache, 
                                       int64_t start_pos) 
{
    auto [B, S, D] = x.sizes3d();  // Batch, SeqLen, EmbedDim
    
    // 1. Project to Q, K, V
    Tensor q = wq_(x);  // [B, S, num_heads * head_dim]
    Tensor k = wk_(x);  // [B, S, num_kv_heads * head_dim]
    Tensor v = wv_(x);  // [B, S, num_kv_heads * head_dim]
    
    // 2. Reshape to [B, S, num_heads/num_kv_heads, head_dim]
    q = q.view({B, S, num_heads_, head_dim_});
    k = k.view({B, S, num_kv_heads_, head_dim_});
    v = v.view({B, S, num_kv_heads_, head_dim_});
    
    // 3. Transpose to [B, H, S, D]
    q = q.transpose(1, 2);  // [B, num_heads, S, head_dim]
    k = k.transpose(1, 2);  // [B, num_kv_heads, S, head_dim]
    v = v.transpose(1, 2);  // [B, num_kv_heads, S, head_dim]
    
    // 4. Apply RoPE
    Tensor freqs = rope_freqs_->get(start_pos, S);
    apply_rope(q, freqs, start_pos);  // Q gets all heads rotated
    apply_rope_kv(k, freqs, start_pos);  // K only rotates KV heads
    
    // 5. KV Cache handling
    if (cache) {
        std::tie(k, v) = cache->update(k, v, start_pos);
    }
    
    // 6. Expand K, V to match Q's head count
    // [B, num_kv_heads, S, D] -> [B, num_heads, S, D]
    k = repeat_kv(k, num_rep_);
    v = repeat_kv(v, num_rep_);
    
    // 7. Scaled Dot-Product Attention
    Tensor out = scaled_dot_product_attention(q, k, v, /*is_causal=*/true, dropout_);
    
    // 8. Merge heads and project
    out = out.transpose(1, 2).contiguous().view({B, S, -1});
    return wo_(out);
}

} // namespace vesper::nn
```

### 3.3 The `repeat_kv` Function

This is the key operation that expands KV heads to match query heads:

```cpp
// Repeat KV heads to match query heads
// Input: [Batch, KV_Heads, SeqLen, HeadDim]
// Output: [Batch, Query_Heads, SeqLen, HeadDim]
Tensor repeat_kv(const Tensor& x, int64_t n_rep) {
    if (n_rep == 1) {
        return x;  // No repetition needed (MHA case)
    }
    
    auto [B, H, S, D] = x.sizes4d();
    
    // Method 1: Using unsqueeze and expand (view-based, no copy)
    // [B, H, S, D] -> [B, H, 1, S, D] -> [B, H, n_rep, S, D] -> [B, H*n_rep, S, D]
    Tensor expanded = x.unsqueeze(2);           // [B, H, 1, S, D]
    expanded = expanded.expand({B, H, n_rep, S, D});  // [B, H, n_rep, S, D]
    return expanded.reshape({B, H * n_rep, S, D});
    
    // Note: expand() creates a view with stride=0 on the repeated dimension.
    // This is memory-efficient but may cause issues with some kernels.
    // For kernels that require contiguous data, call .contiguous() afterwards.
}
```

### 3.4 GPU Kernel for repeat_kv

For cases where we need a contiguous result, a custom kernel is more efficient than expand + contiguous:

```cpp
// src/ops/hip/repeat_kv.hip

__global__ void repeat_kv_kernel(
    const float* __restrict__ input,   // [B, KV_H, S, D]
    float* __restrict__ output,         // [B, Q_H, S, D]
    int batch, int kv_heads, int seq_len, int head_dim,
    int n_rep)
{
    // Each thread handles one element of the output
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_heads * n_rep * seq_len * head_dim;
    
    if (idx >= total) return;
    
    // Decode output index
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int q_h = (idx / (head_dim * seq_len)) % (kv_heads * n_rep);
    int b = idx / (head_dim * seq_len * kv_heads * n_rep);
    
    // Map query head to KV head
    int kv_h = q_h / n_rep;
    
    // Read from input
    int in_idx = ((b * kv_heads + kv_h) * seq_len + s) * head_dim + d;
    output[idx] = input[in_idx];
}

// Vectorized version (float4)
__global__ void repeat_kv_kernel_vec4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int batch, int kv_heads, int seq_len, int head_dim_div4,
    int n_rep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_heads * n_rep * seq_len * head_dim_div4;
    
    if (idx >= total) return;
    
    int d4 = idx % head_dim_div4;
    int s = (idx / head_dim_div4) % seq_len;
    int q_h = (idx / (head_dim_div4 * seq_len)) % (kv_heads * n_rep);
    int b = idx / (head_dim_div4 * seq_len * kv_heads * n_rep);
    
    int kv_h = q_h / n_rep;
    
    int in_idx = ((b * kv_heads + kv_h) * seq_len + s) * head_dim_div4 + d4;
    output[idx] = input[in_idx];
}
```

## 4. KV Cache for GQA

The KV cache dimensions change with GQA:

```cpp
// include/vesper/nn/kv_cache.h

class GQAKVCache {
public:
    // Note: Uses num_kv_heads, not num_heads
    GQAKVCache(int64_t batch_size, int64_t num_kv_heads, 
               int64_t max_seq_len, int64_t head_dim, Device device)
        : max_seq_len_(max_seq_len), current_len_(0)
    {
        // Allocate reduced-size cache
        k_cache_ = zeros({batch_size, num_kv_heads, max_seq_len, head_dim}, 
                         DType::Float32, device);
        v_cache_ = zeros({batch_size, num_kv_heads, max_seq_len, head_dim}, 
                         DType::Float32, device);
    }
    
    std::pair<Tensor, Tensor> update(const Tensor& new_k, const Tensor& new_v, 
                                      int64_t start_pos) {
        int64_t seq_len = new_k.shape()[2];
        
        // Write new K, V to cache at [start_pos : start_pos + seq_len]
        k_cache_.slice(2, start_pos, start_pos + seq_len).copy_(new_k);
        v_cache_.slice(2, start_pos, start_pos + seq_len).copy_(new_v);
        
        current_len_ = start_pos + seq_len;
        
        // Return view of full context
        return {
            k_cache_.slice(2, 0, current_len_),
            v_cache_.slice(2, 0, current_len_)
        };
    }
    
    // Memory savings compared to MHA
    static float memory_ratio(int64_t num_heads, int64_t num_kv_heads) {
        return static_cast<float>(num_kv_heads) / num_heads;
    }
    
private:
    Tensor k_cache_;
    Tensor v_cache_;
    int64_t max_seq_len_;
    int64_t current_len_;
};
```

## 5. Attention Computation with GQA

### 5.1 Method 1: Expand then Standard Attention

```cpp
// Simple but memory-intensive
k_expanded = repeat_kv(k, num_rep);  // Expand to match Q
v_expanded = repeat_kv(v, num_rep);
output = scaled_dot_product_attention(q, k_expanded, v_expanded);
```

### 5.2 Method 2: Grouped Attention Kernel

More memory-efficient: compute attention within groups without expansion.

```cpp
// Compute attention scores for GQA without expanding K
// Q: [B, Q_Heads, S_q, D]
// K: [B, KV_Heads, S_k, D]
// Output: [B, Q_Heads, S_q, S_k]
__global__ void gqa_scores_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ scores,
    int batch, int q_heads, int kv_heads, int s_q, int s_k, int d,
    float scale)
{
    // Each thread block computes a tile of the output
    int q_head = blockIdx.y;
    int kv_head = q_head / (q_heads / kv_heads);  // Map Q head -> KV head
    
    // Standard attention score computation
    // score[b, q_h, i, j] = sum_d(Q[b, q_h, i, d] * K[b, kv_h, j, d]) * scale
    
    extern __shared__ float shared_mem[];
    float* sQ = shared_mem;
    float* sK = shared_mem + TILE_Q * TILE_D;
    
    // ... (tiled GEMM-like computation)
}
```

## 6. Configuration Examples

### 6.1 Llama Configurations

```cpp
struct LlamaConfig {
    int64_t dim;
    int64_t n_layers;
    int64_t n_heads;
    int64_t n_kv_heads;  // GQA!
    int64_t vocab_size;
    int64_t max_seq_len;
    float rope_base;
    
    static LlamaConfig llama2_7b() {
        return {4096, 32, 32, 32, 32000, 4096, 10000.0f};  // MHA
    }
    
    static LlamaConfig llama2_70b() {
        return {8192, 80, 64, 8, 32000, 4096, 10000.0f};   // GQA!
    }
    
    static LlamaConfig llama3_8b() {
        return {4096, 32, 32, 8, 128000, 8192, 500000.0f}; // GQA
    }
    
    static LlamaConfig llama3_70b() {
        return {8192, 80, 64, 8, 128000, 8192, 500000.0f}; // GQA
    }
    
    static LlamaConfig mistral_7b() {
        return {4096, 32, 32, 8, 32000, 8192, 10000.0f};   // GQA
    }
};
```

## 7. Backward Pass

### 7.1 repeat_kv Backward

The backward of `repeat_kv` is a sum-reduction:

```cpp
// Forward: repeat_kv expands [B, KV_H, S, D] -> [B, Q_H, S, D]
// Backward: sum over the repeated dimension

Tensor repeat_kv_backward(const Tensor& grad_output, int64_t n_rep) {
    // grad_output: [B, Q_H, S, D]
    // grad_input: [B, KV_H, S, D]
    
    auto [B, Q_H, S, D] = grad_output.sizes4d();
    int64_t KV_H = Q_H / n_rep;
    
    // Reshape: [B, Q_H, S, D] -> [B, KV_H, n_rep, S, D]
    Tensor reshaped = grad_output.view({B, KV_H, n_rep, S, D});
    
    // Sum over n_rep dimension
    return reshaped.sum(/*dim=*/2);  // [B, KV_H, S, D]
}
```

### 7.2 GPU Kernel for Backward

```cpp
__global__ void repeat_kv_backward_kernel(
    const float* __restrict__ grad_out,  // [B, Q_H, S, D]
    float* __restrict__ grad_in,          // [B, KV_H, S, D]
    int batch, int kv_heads, int q_heads, int seq_len, int head_dim)
{
    int n_rep = q_heads / kv_heads;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_heads * seq_len * head_dim;
    
    if (idx >= total) return;
    
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int kv_h = (idx / (head_dim * seq_len)) % kv_heads;
    int b = idx / (head_dim * seq_len * kv_heads);
    
    // Sum gradients from all Q heads that share this KV head
    float sum = 0.0f;
    for (int r = 0; r < n_rep; ++r) {
        int q_h = kv_h * n_rep + r;
        int out_idx = ((b * q_heads + q_h) * seq_len + s) * head_dim + d;
        sum += grad_out[out_idx];
    }
    
    grad_in[idx] = sum;
}
```

## 8. Testing Strategy

### 8.1 Unit Tests

```cpp
// tests/nn/test_gqa.cpp

TEST(GQA, RepeatKV_Correctness) {
    // [B=1, KV_H=2, S=3, D=4] -> [B=1, Q_H=8, S=3, D=4] with n_rep=4
    Tensor x = arange(24).view({1, 2, 3, 4}).to(DType::Float32);
    Tensor y = repeat_kv(x, 4);
    
    EXPECT_EQ(y.shape(), std::vector<int64_t>({1, 8, 3, 4}));
    
    // Check that heads 0,1,2,3 are copies of KV head 0
    // and heads 4,5,6,7 are copies of KV head 1
    for (int q_h = 0; q_h < 8; ++q_h) {
        int kv_h = q_h / 4;
        EXPECT_TRUE(allclose(
            y.select(1, q_h),   // Query head q_h
            x.select(1, kv_h)   // KV head kv_h
        ));
    }
}

TEST(GQA, AttentionOutputShape) {
    GroupedQueryAttention gqa(
        /*embed_dim=*/512,
        /*num_heads=*/8,
        /*num_kv_heads=*/2,  // 4:1 ratio
        /*max_seq_len=*/128
    );
    
    Tensor x = randn({2, 16, 512});
    Tensor y = gqa.forward(x);
    
    EXPECT_EQ(y.shape(), std::vector<int64_t>({2, 16, 512}));
}

TEST(GQA, ParameterCount) {
    int64_t d = 512;
    int64_t h = 8;     // Query heads
    int64_t kv_h = 2;  // KV heads
    int64_t head_dim = d / h;  // 64
    
    GroupedQueryAttention gqa(d, h, kv_h, 128);
    
    // Q: d * (h * head_dim) = 512 * 512
    // K: d * (kv_h * head_dim) = 512 * 128
    // V: d * (kv_h * head_dim) = 512 * 128
    // O: (h * head_dim) * d = 512 * 512
    int64_t expected = d * (h * head_dim) +      // Q
                       d * (kv_h * head_dim) +   // K
                       d * (kv_h * head_dim) +   // V
                       (h * head_dim) * d;       // O
    
    // = 512*512 + 512*128 + 512*128 + 512*512
    // = 262144 + 65536 + 65536 + 262144 = 655360
    
    int64_t actual = 0;
    for (auto& p : gqa.parameters()) {
        actual += p.numel();
    }
    
    EXPECT_EQ(actual, expected);
}

TEST(GQA, CausalMasking) {
    GroupedQueryAttention gqa(64, 4, 2, 128);
    
    Tensor x = randn({1, 8, 64});
    Tensor y = gqa.forward(x);
    
    // Changing future tokens should not affect past outputs
    Tensor x_modified = x.clone();
    x_modified.select(1, 7).fill_(999.0f);  // Change last token
    
    Tensor y_modified = gqa.forward(x_modified);
    
    // First 7 positions should be identical
    for (int i = 0; i < 7; ++i) {
        EXPECT_TRUE(allclose(y.select(1, i), y_modified.select(1, i)));
    }
}

TEST(GQA, EquivalentToMHA_WhenEqual) {
    // When num_heads == num_kv_heads, GQA should be identical to MHA
    int64_t d = 128;
    int64_t h = 4;
    
    GroupedQueryAttention gqa(d, h, h, 64);  // num_kv_heads == num_heads
    MultiHeadAttention mha(d, h, 64);
    
    // Copy weights (assuming same structure)
    // ... weight copying ...
    
    Tensor x = randn({1, 8, d});
    Tensor y_gqa = gqa.forward(x);
    Tensor y_mha = mha.forward(x);
    
    // Should produce identical results
    EXPECT_TRUE(allclose(y_gqa, y_mha, 1e-5, 1e-5));
}
```

### 8.2 Stress Tests

```cpp
TEST(GQA, StressTest_LlamaScale) {
    // Simulate Llama 2 70B attention
    int64_t embed_dim = 8192;
    int64_t num_heads = 64;
    int64_t num_kv_heads = 8;
    int64_t batch = 4;
    int64_t seq_len = 2048;
    
    GroupedQueryAttention gqa(embed_dim, num_heads, num_kv_heads, 4096);
    gqa.to(Device::HIP);
    
    Tensor x = randn({batch, seq_len, embed_dim}, Device::HIP);
    
    // Warm-up
    Tensor y = gqa.forward(x);
    hipDeviceSynchronize();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        y = gqa.forward(x);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "GQA [4, 2048, 8192] 64Q/8KV x10: " << ms << " ms" << std::endl;
}

TEST(GQA, StressTest_KVCacheMemory) {
    // Compare memory usage: MHA vs GQA
    int64_t batch = 8;
    int64_t max_seq = 8192;
    int64_t head_dim = 128;
    
    int64_t mha_heads = 64;
    int64_t gqa_kv_heads = 8;
    
    // MHA cache
    size_t mha_cache_size = 2 * batch * mha_heads * max_seq * head_dim * sizeof(float);
    
    // GQA cache
    size_t gqa_cache_size = 2 * batch * gqa_kv_heads * max_seq * head_dim * sizeof(float);
    
    float savings = 1.0f - (float)gqa_cache_size / mha_cache_size;
    
    std::cout << "MHA KV Cache: " << mha_cache_size / (1024*1024) << " MB" << std::endl;
    std::cout << "GQA KV Cache: " << gqa_cache_size / (1024*1024) << " MB" << std::endl;
    std::cout << "Memory Savings: " << savings * 100 << "%" << std::endl;
    
    EXPECT_NEAR(savings, 0.875, 0.001);  // 87.5% savings for 64->8 heads
}

TEST(GQA, StressTest_LongContextInference) {
    GroupedQueryAttention gqa(512, 8, 2, 32768);  // 32K context
    gqa.to(Device::HIP);
    
    GQAKVCache cache(1, 2, 32768, 64, Device::HIP);
    
    // Prefill with 1024 tokens
    Tensor prompt = randn({1, 1024, 512}, Device::HIP);
    gqa.forward(prompt, &cache, 0);
    
    // Generate 30K more tokens
    for (int64_t pos = 1024; pos < 31000; pos += 100) {
        Tensor new_tokens = randn({1, 100, 512}, Device::HIP);
        Tensor out = gqa.forward(new_tokens, &cache, pos);
        
        EXPECT_FALSE(out.isnan().any().item<bool>());
    }
}

TEST(GQA, StressTest_GradientFlow) {
    GroupedQueryAttention gqa(128, 8, 2, 64);
    
    Tensor x = randn({2, 32, 128}, /*requires_grad=*/true);
    
    for (int iter = 0; iter < 100; ++iter) {
        Tensor y = gqa.forward(x);
        Tensor loss = y.sum();
        loss.backward();
        
        // Check gradients are valid
        EXPECT_FALSE(x.grad().isnan().any().item<bool>());
        
        // Check parameter gradients
        for (auto& p : gqa.parameters()) {
            EXPECT_FALSE(p.grad().isnan().any().item<bool>());
        }
        
        gqa.zero_grad();
        x.grad().zero_();
    }
}
```

### 8.3 Numerical Tests

```cpp
TEST(GQA, RepeatKV_GradCheck) {
    Tensor x = randn({1, 2, 4, 8}, /*requires_grad=*/true);
    
    auto func = [](const Tensor& input) {
        return repeat_kv(input, 4).sum();
    };
    
    EXPECT_TRUE(autograd::gradcheck(func, x, 1e-4, 1e-3));
}

TEST(GQA, FullBackward_GradCheck) {
    GroupedQueryAttention gqa(32, 4, 2, 16);
    
    Tensor x = randn({1, 4, 32}, /*requires_grad=*/true);
    
    auto func = [&](const Tensor& input) {
        return gqa.forward(input).sum();
    };
    
    EXPECT_TRUE(autograd::gradcheck(func, x, 1e-4, 1e-3));
}
```

## 9. Performance Analysis

### 9.1 Memory Bandwidth

| Operation | MHA | GQA (8:1) | Savings |
|-----------|-----|-----------|---------|
| K projection | B×S×D×H×d | B×S×D×G×d | 87.5% |
| V projection | B×S×D×H×d | B×S×D×G×d | 87.5% |
| KV cache read | B×H×S×d | B×G×S×d | 87.5% |
| repeat_kv | - | B×H×S×d | (new cost) |

### 9.2 Compute vs Memory Tradeoff

- **Training**: GQA slightly reduces memory for gradients but compute is similar.
- **Inference**: Major wins in KV cache memory and bandwidth.
- **Prefill**: Similar to MHA (need to compute all K/V anyway).
- **Decode**: Significant speedup due to reduced memory bandwidth.

## 10. Summary

Grouped Query Attention is a key architectural optimization for efficient LLM inference:

1. **Core Idea**: Share K/V heads among multiple Q heads.
2. **Implementation**: Use `repeat_kv` to expand K/V or write a grouped attention kernel.
3. **KV Cache**: Store only `num_kv_heads` worth of data, not `num_heads`.
4. **Memory Savings**: Up to 87.5% for Llama 2 70B configuration.

With GQA implemented, Vesper can efficiently run large models that would otherwise be memory-bound during inference.

```
