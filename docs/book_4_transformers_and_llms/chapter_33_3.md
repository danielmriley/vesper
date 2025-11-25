```markdown
# Chapter 33.3: Rotary Positional Embeddings (RoPE)

## 1. Introduction

Rotary Positional Embeddings (RoPE) have become the de facto standard for positional encoding in modern LLMs (Llama, Mistral, Qwen, Gemma, etc.). Unlike absolute positional embeddings (learned vectors added to token embeddings) or sinusoidal embeddings (fixed patterns), RoPE encodes position by **rotating** the query and key vectors in the complex plane.

### Why RoPE?

1. **Relative Position Encoding**: The dot product `q · k` naturally encodes the *relative* distance between tokens, not their absolute positions.
2. **Extrapolation**: RoPE allows some degree of length extrapolation beyond the training context.
3. **Efficiency**: Applied during attention computation, not as a separate embedding layer.
4. **No Additional Parameters**: RoPE is parameter-free—just a mathematical transformation.

## 2. Mathematical Foundation

### The Core Idea

RoPE treats pairs of dimensions in the embedding as 2D vectors and rotates them by an angle proportional to the position.

For a vector $\mathbf{x} = [x_0, x_1, x_2, x_3, \ldots, x_{d-1}]$ at position $m$:

1. **Pair dimensions**: $(x_0, x_1), (x_2, x_3), \ldots, (x_{d-2}, x_{d-1})$
2. **Each pair is rotated by angle** $\theta_i \cdot m$ where $\theta_i = 10000^{-2i/d}$

### Rotation Matrix (2D)

For a single pair $(x_{2i}, x_{2i+1})$:

$$
\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} = 
\begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix}
\begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}
$$

Expanded:
$$
x'_{2i} = x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i)
$$
$$
x'_{2i+1} = x_{2i} \sin(m\theta_i) + x_{2i+1} \cos(m\theta_i)
$$

### The Frequency Schedule

The base frequencies follow a geometric progression:

$$
\theta_i = \text{base}^{-2i/d} \quad \text{where base} = 10000 \text{ (or 500000 for Llama 3)}
$$

For `d=64` (typical head dimension):
- $\theta_0 = 10000^{0/64} = 1.0$
- $\theta_1 = 10000^{-2/64} \approx 0.631$
- $\theta_{31} = 10000^{-62/64} \approx 0.0001$

Lower dimensions rotate faster (high frequency), higher dimensions rotate slower (low frequency).

### Why This Works for Relative Position

The key insight: when computing attention scores $q_m \cdot k_n$, the rotation angles combine:

$$
\text{RoPE}(q, m) \cdot \text{RoPE}(k, n) = f(q, k, m-n)
$$

The dot product depends only on the *relative* position $(m - n)$, not the absolute positions.

## 3. Implementation Plan

### 3.1 Precomputing Frequencies

Since the frequencies $\theta_i$ are constant, we precompute them once.

```cpp
// include/vesper/nn/rope.h

namespace vesper::nn {

class RoPEFrequencies {
public:
    // Precompute cos/sin tables for positions [0, max_seq_len)
    // Shape: [max_seq_len, head_dim/2, 2] where last dim is [cos, sin]
    RoPEFrequencies(int64_t max_seq_len, int64_t head_dim, 
                    float base = 10000.0f, Device device = Device::CPU);
    
    // Get frequencies for a range of positions
    // Returns: [seq_len, head_dim/2, 2]
    Tensor get(int64_t start_pos, int64_t seq_len) const;
    
    // For dynamic NTK-aware scaling (Llama 3.1+)
    void update_base(float new_base);
    
private:
    Tensor freqs_;       // [max_seq_len, head_dim/2, 2]
    int64_t head_dim_;
    float base_;
    Device device_;
};

} // namespace vesper::nn
```

### 3.2 CPU Implementation

```cpp
// src/nn/rope.cpp

RoPEFrequencies::RoPEFrequencies(int64_t max_seq_len, int64_t head_dim, 
                                  float base, Device device)
    : head_dim_(head_dim), base_(base), device_(device) 
{
    VESPER_CHECK(head_dim % 2 == 0, "head_dim must be even for RoPE");
    
    int64_t half_dim = head_dim / 2;
    
    // Compute inverse frequencies: theta_i = base^(-2i/d)
    std::vector<float> inv_freq(half_dim);
    for (int64_t i = 0; i < half_dim; ++i) {
        inv_freq[i] = 1.0f / std::pow(base, static_cast<float>(2 * i) / head_dim);
    }
    
    // Precompute cos/sin for all positions
    // Shape: [max_seq_len, half_dim, 2]
    freqs_ = zeros({max_seq_len, half_dim, 2}, DType::Float32, device);
    float* data = freqs_.data_ptr<float>();
    
    for (int64_t pos = 0; pos < max_seq_len; ++pos) {
        for (int64_t i = 0; i < half_dim; ++i) {
            float angle = pos * inv_freq[i];
            int64_t idx = (pos * half_dim + i) * 2;
            data[idx]     = std::cos(angle);  // cos
            data[idx + 1] = std::sin(angle);  // sin
        }
    }
}

Tensor RoPEFrequencies::get(int64_t start_pos, int64_t seq_len) const {
    // Return a slice [start_pos : start_pos + seq_len]
    return freqs_.slice(0, start_pos, start_pos + seq_len);
}
```

### 3.3 The RoPE Application Function

```cpp
// include/vesper/nn/functional/rope.h

namespace vesper::nn::functional {

// Apply RoPE to query and key tensors
// q, k: [Batch, Heads, SeqLen, HeadDim]
// freqs: [SeqLen, HeadDim/2, 2] containing [cos, sin]
// start_pos: Position offset (for KV cache inference)
void apply_rope(Tensor& q, Tensor& k, const Tensor& freqs, int64_t start_pos = 0);

// In-place version for a single tensor
void apply_rope_inplace(Tensor& x, const Tensor& freqs, int64_t start_pos = 0);

} // namespace vesper::nn::functional
```

### 3.4 GPU Kernel

```cpp
// src/ops/hip/rope.hip

#include <hip/hip_runtime.h>

// Kernel: Apply RoPE rotation to Q or K tensor
// Input x: [Batch, Heads, SeqLen, HeadDim]
// freqs: [SeqLen, HeadDim/2, 2] with [cos, sin] pairs
__global__ void rope_kernel(
    float* __restrict__ x,
    const float* __restrict__ freqs,
    int batch_size, int num_heads, int seq_len, int head_dim,
    int start_pos)
{
    // Grid: (batch * heads, seq_len, head_dim / 2)
    int batch_head = blockIdx.x;
    int pos = blockIdx.y * blockDim.y + threadIdx.y;
    int pair_idx = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (pos >= seq_len || pair_idx >= head_dim / 2) return;
    
    int b = batch_head / num_heads;
    int h = batch_head % num_heads;
    
    // Absolute position for frequency lookup
    int abs_pos = start_pos + pos;
    
    // Load cos/sin for this position and dimension pair
    int freq_idx = pos * (head_dim / 2) * 2 + pair_idx * 2;
    float cos_val = freqs[freq_idx];
    float sin_val = freqs[freq_idx + 1];
    
    // Load the pair of values to rotate
    int x_base = ((b * num_heads + h) * seq_len + pos) * head_dim;
    int idx0 = x_base + pair_idx * 2;
    int idx1 = x_base + pair_idx * 2 + 1;
    
    float x0 = x[idx0];
    float x1 = x[idx1];
    
    // Apply rotation
    x[idx0] = x0 * cos_val - x1 * sin_val;
    x[idx1] = x0 * sin_val + x1 * cos_val;
}

void apply_rope_hip(Tensor& x, const Tensor& freqs, int64_t start_pos) {
    auto shape = x.shape();
    int batch = shape[0];
    int heads = shape[1];
    int seq_len = shape[2];
    int head_dim = shape[3];
    
    dim3 threads(1, 16, 16);  // pos, pair_idx
    dim3 blocks(
        batch * heads,
        (seq_len + 15) / 16,
        (head_dim / 2 + 15) / 16
    );
    
    hipStream_t stream = static_cast<hipStream_t>(
        Stream::current(Device::HIP).raw_handle());
    
    hipLaunchKernelGGL(rope_kernel, blocks, threads, 0, stream,
        x.data_ptr<float>(),
        freqs.data_ptr<float>(),
        batch, heads, seq_len, head_dim,
        static_cast<int>(start_pos)
    );
}
```

### 3.5 Vectorized Kernel (float4)

For better memory throughput, we can process 2 pairs (4 floats) at once:

```cpp
__global__ void rope_vectorized_kernel(
    float* __restrict__ x,
    const float* __restrict__ freqs,
    int batch_size, int num_heads, int seq_len, int head_dim,
    int start_pos)
{
    // Each thread handles 2 dimension pairs (4 floats)
    int batch_head = blockIdx.x;
    int pos = blockIdx.y * blockDim.y + threadIdx.y;
    int vec_idx = threadIdx.x;  // 0 to (head_dim/4 - 1)
    
    if (pos >= seq_len) return;
    
    int half_dim = head_dim / 2;
    int pair_idx = vec_idx * 2;  // Process pairs [pair_idx, pair_idx+1]
    
    if (pair_idx + 1 >= half_dim) return;
    
    int b = batch_head / num_heads;
    int h = batch_head % num_heads;
    
    // Load 2 cos/sin pairs
    int freq_base = pos * half_dim * 2 + pair_idx * 2;
    float cos0 = freqs[freq_base];
    float sin0 = freqs[freq_base + 1];
    float cos1 = freqs[freq_base + 2];
    float sin1 = freqs[freq_base + 3];
    
    // Load 4 values as float4
    int x_base = ((b * num_heads + h) * seq_len + pos) * head_dim;
    float4* x_vec = reinterpret_cast<float4*>(&x[x_base + pair_idx * 2]);
    float4 vals = *x_vec;
    
    // Apply rotations to both pairs
    float4 result;
    result.x = vals.x * cos0 - vals.y * sin0;  // pair 0
    result.y = vals.x * sin0 + vals.y * cos0;
    result.z = vals.z * cos1 - vals.w * sin1;  // pair 1
    result.w = vals.z * sin1 + vals.w * cos1;
    
    *x_vec = result;
}
```

## 4. Integration with Multi-Head Attention

### 4.1 Updated MHA Module

```cpp
// src/nn/multi_head_attention.cpp

class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(int embed_dim, int num_heads, int max_seq_len = 4096,
                       float rope_base = 10000.0f, float dropout = 0.0f)
        : n_heads_(num_heads), 
          head_dim_(embed_dim / num_heads),
          dropout_(dropout)
    {
        VESPER_CHECK(embed_dim % num_heads == 0, 
            "embed_dim must be divisible by num_heads");
        
        // QKV projection (combined for efficiency)
        c_attn_ = register_module("c_attn", Linear(embed_dim, 3 * embed_dim, false));
        
        // Output projection
        c_proj_ = register_module("c_proj", Linear(embed_dim, embed_dim, false));
        
        // RoPE frequencies (precomputed)
        rope_freqs_ = std::make_unique<RoPEFrequencies>(
            max_seq_len, head_dim_, rope_base, Device::HIP);
    }
    
    Tensor forward(Tensor x, KVCache* cache = nullptr, int start_pos = 0) {
        auto [B, S, D] = std::make_tuple(x.shape()[0], x.shape()[1], x.shape()[2]);
        
        // 1. Project to Q, K, V
        Tensor qkv = c_attn_(x);  // [B, S, 3*D]
        auto chunks = qkv.chunk(3, /*dim=*/-1);
        Tensor q = chunks[0], k = chunks[1], v = chunks[2];
        
        // 2. Reshape to [B, H, S, D_head]
        q = q.view({B, S, n_heads_, head_dim_}).transpose(1, 2);
        k = k.view({B, S, n_heads_, head_dim_}).transpose(1, 2);
        v = v.view({B, S, n_heads_, head_dim_}).transpose(1, 2);
        
        // 3. Apply RoPE to Q and K
        Tensor freqs = rope_freqs_->get(start_pos, S);
        nn::functional::apply_rope(q, k, freqs, start_pos);
        
        // 4. KV Cache (if provided)
        if (cache) {
            std::tie(k, v) = cache->update(k, v, start_pos);
        }
        
        // 5. Scaled Dot-Product Attention
        Tensor out = nn::functional::scaled_dot_product_attention(
            q, k, v, /*is_causal=*/true, dropout_);
        
        // 6. Merge heads and project
        out = out.transpose(1, 2).contiguous().view({B, S, D});
        return c_proj_(out);
    }
    
private:
    int n_heads_;
    int head_dim_;
    float dropout_;
    Linear c_attn_, c_proj_;
    std::unique_ptr<RoPEFrequencies> rope_freqs_;
};
```

## 5. Advanced: NTK-Aware Scaling (Llama 3.1+)

For models that need to extrapolate beyond training context, we can dynamically adjust the RoPE base:

```cpp
// Dynamic NTK-aware scaling
float compute_ntk_base(int64_t original_max_len, int64_t current_len, 
                        float original_base, float alpha = 1.0f) {
    if (current_len <= original_max_len) {
        return original_base;
    }
    
    // Scale factor based on how much we're extrapolating
    float scale = static_cast<float>(current_len) / original_max_len;
    
    // NTK-aware base adjustment
    // This spreads the rotations more evenly across the extended context
    float new_base = original_base * std::pow(scale, alpha);
    return new_base;
}
```

## 6. Backward Pass (Autograd)

RoPE is differentiable. The backward pass applies the *inverse* rotation (negate the angle):

```cpp
class RoPEBackward : public autograd::Function {
public:
    static Tensor forward(Tensor& q, Tensor& k, const Tensor& freqs) {
        // Store freqs for backward
        // Apply RoPE in-place
        nn::functional::apply_rope(q, k, freqs);
        return q;  // Or return both somehow
    }
    
    static std::vector<Tensor> backward(const Tensor& grad_q, const Tensor& grad_k,
                                         const Tensor& freqs) {
        // Inverse rotation: same cos, negate sin
        Tensor neg_freqs = freqs.clone();
        // Negate sin values (index 1 of each [cos, sin] pair)
        // ... implementation details ...
        
        Tensor grad_q_in = grad_q.clone();
        Tensor grad_k_in = grad_k.clone();
        apply_rope_backward(grad_q_in, grad_k_in, neg_freqs);
        
        return {grad_q_in, grad_k_in};
    }
};
```

## 7. Testing Strategy

### 7.1 Unit Tests

```cpp
// tests/nn/test_rope.cpp

TEST(RoPE, FrequencyComputation) {
    RoPEFrequencies freqs(128, 64, 10000.0f, Device::CPU);
    Tensor f = freqs.get(0, 1);  // Position 0
    
    // At position 0, all angles are 0, so cos=1, sin=0
    float* data = f.data_ptr<float>();
    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(data[i * 2], 1.0f, 1e-6);      // cos(0) = 1
        EXPECT_NEAR(data[i * 2 + 1], 0.0f, 1e-6);  // sin(0) = 0
    }
}

TEST(RoPE, RotationPreservesNorm) {
    // RoPE is a rotation, so it preserves vector norms
    Tensor x = randn({1, 8, 16, 64});  // [B, H, S, D]
    Tensor x_copy = x.clone();
    
    RoPEFrequencies rope(128, 64);
    Tensor freqs = rope.get(0, 16);
    
    apply_rope_inplace(x, freqs, 0);
    
    // Check norms are preserved
    Tensor orig_norm = x_copy.norm(/*dim=*/-1);
    Tensor new_norm = x.norm(/*dim=*/-1);
    
    EXPECT_TRUE(allclose(orig_norm, new_norm, 1e-5, 1e-5));
}

TEST(RoPE, RelativePositionProperty) {
    // q_m · k_n should depend only on (m - n)
    Tensor q = randn({1, 1, 1, 64});
    Tensor k = randn({1, 1, 1, 64});
    
    RoPEFrequencies rope(128, 64);
    
    // Scenario 1: q at pos 5, k at pos 3 (relative = 2)
    Tensor q1 = q.clone(), k1 = k.clone();
    apply_rope_inplace(q1, rope.get(5, 1), 5);
    apply_rope_inplace(k1, rope.get(3, 1), 3);
    float dot1 = (q1 * k1).sum().item<float>();
    
    // Scenario 2: q at pos 10, k at pos 8 (relative = 2)
    Tensor q2 = q.clone(), k2 = k.clone();
    apply_rope_inplace(q2, rope.get(10, 1), 10);
    apply_rope_inplace(k2, rope.get(8, 1), 8);
    float dot2 = (q2 * k2).sum().item<float>();
    
    EXPECT_NEAR(dot1, dot2, 1e-5);
}

TEST(RoPE, InverseRotation) {
    // Applying RoPE then inverse should return original
    Tensor x = randn({1, 8, 16, 64});
    Tensor original = x.clone();
    
    RoPEFrequencies rope(128, 64);
    Tensor freqs = rope.get(0, 16);
    
    // Forward
    apply_rope_inplace(x, freqs, 0);
    
    // Create inverse freqs (negate sin)
    Tensor inv_freqs = freqs.clone();
    // Negate every sin component
    // ...
    
    // Backward
    apply_rope_inplace(x, inv_freqs, 0);
    
    EXPECT_TRUE(allclose(x, original, 1e-5, 1e-5));
}
```

### 7.2 Stress Tests

```cpp
TEST(RoPE, StressTest_LongContext) {
    // Test with very long sequences (up to 128K)
    std::vector<int64_t> seq_lengths = {1024, 4096, 16384, 65536, 131072};
    
    for (int64_t seq_len : seq_lengths) {
        RoPEFrequencies rope(seq_len, 128);  // Llama head_dim
        
        // Should not throw or produce NaN
        Tensor freqs = rope.get(seq_len - 1, 1);  // Last position
        EXPECT_FALSE(freqs.isnan().any().item<bool>());
        EXPECT_FALSE(freqs.isinf().any().item<bool>());
    }
}

TEST(RoPE, StressTest_BatchedProcessing) {
    // Large batch on GPU
    Tensor x = randn({64, 32, 2048, 128}, Device::HIP);  // [B=64, H=32, S=2048, D=128]
    
    RoPEFrequencies rope(4096, 128, 10000.0f, Device::HIP);
    Tensor freqs = rope.get(0, 2048);
    
    auto start = std::chrono::high_resolution_clock::now();
    apply_rope_inplace(x, freqs, 0);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "RoPE on [64, 32, 2048, 128]: " << ms << " ms" << std::endl;
    
    // Verify no NaN
    EXPECT_FALSE(x.isnan().any().item<bool>());
}

TEST(RoPE, StressTest_IncrementalDecoding) {
    // Simulate token-by-token generation with KV cache
    int64_t max_len = 2048;
    RoPEFrequencies rope(max_len, 64, 10000.0f, Device::HIP);
    
    // Initial prompt (128 tokens)
    Tensor q = randn({1, 8, 128, 64}, Device::HIP);
    Tensor freqs = rope.get(0, 128);
    apply_rope_inplace(q, freqs, 0);
    
    // Decode 1000 more tokens one at a time
    for (int pos = 128; pos < 1128; ++pos) {
        Tensor new_q = randn({1, 8, 1, 64}, Device::HIP);
        Tensor new_freqs = rope.get(pos, 1);
        apply_rope_inplace(new_q, new_freqs, pos);
        
        // Verify
        EXPECT_FALSE(new_q.isnan().any().item<bool>());
    }
}
```

### 7.3 Gradient Tests

```cpp
TEST(RoPE, GradientCheck) {
    Tensor x = randn({1, 2, 4, 8}, /*requires_grad=*/true);
    
    RoPEFrequencies rope(16, 8);
    Tensor freqs = rope.get(0, 4);
    
    // Numerical gradient check
    auto func = [&](Tensor input) {
        Tensor out = input.clone();
        apply_rope_inplace(out, freqs, 0);
        return out.sum();
    };
    
    EXPECT_TRUE(autograd::gradcheck(func, x, /*eps=*/1e-4, /*rtol=*/1e-3));
}
```

## 8. Performance Considerations

| Optimization | Impact | Complexity |
|-------------|--------|------------|
| Precompute cos/sin | High | Low |
| float4 vectorization | Medium | Medium |
| Fuse with QKV projection | High | High |
| Cache frequencies on device | Medium | Low |

### Benchmark Results (Expected)

| Batch | Heads | SeqLen | HeadDim | Time (μs) |
|-------|-------|--------|---------|-----------|
| 1 | 32 | 2048 | 128 | ~50 |
| 16 | 32 | 2048 | 128 | ~200 |
| 1 | 32 | 32768 | 128 | ~500 |

## 9. Common Pitfalls

1. **Forgetting to apply RoPE before caching**: The KV cache stores *rotated* keys. New tokens must be rotated with their absolute position.

2. **Wrong dimension pairing**: Some implementations pair (0, d/2), (1, d/2+1), etc. (interleaved). Llama uses adjacent pairs (0,1), (2,3). Check the model's original implementation.

3. **Float precision at long contexts**: At position 100K+, some frequencies may lose precision. Consider using `double` for frequency computation.

4. **Base frequency mismatch**: Llama 2 uses 10000, Llama 3 uses 500000. Using the wrong base produces garbage outputs.

## 10. Summary

RoPE is elegant in theory and efficient in practice. The key implementation points are:

1. **Precompute** cos/sin tables for the maximum context length.
2. **Apply in-place** to Q and K before attention.
3. **Use start_pos** for incremental decoding with KV cache.
4. **Vectorize** the kernel for GPU efficiency.

With RoPE implemented, Vesper can now match the positional encoding behavior of Llama, Mistral, and other modern LLMs.

```
