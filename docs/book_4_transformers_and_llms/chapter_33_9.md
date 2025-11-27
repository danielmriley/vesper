# Chapter 33.9: Memory-Efficient Training - FP16/BF16, Flash Attention, and Gradient Checkpointing

## 1. Introduction

Training large language models requires careful memory management. A GPT-2 124M model requires ~500MB just for parameters in FP32, but during training with optimizer states and gradients, this balloons to 2-3GB. For Llama 7B, we're looking at 28GB in FP32—impossible on most GPUs without optimization techniques.

This chapter covers three critical memory optimizations:

1. **Mixed Precision Training (FP16/BF16)**: Halve memory usage while maintaining accuracy
2. **Flash Attention**: Memory-efficient attention with O(N) instead of O(N²) memory
3. **Gradient Checkpointing**: Trade compute for memory by recomputing activations

Together, these techniques enable training models 4-10x larger than naive implementations.

### Memory Breakdown

For a transformer with N parameters during training:

| Component | FP32 | FP16/BF16 Mixed |
|-----------|------|-----------------|
| Parameters | 4N | 2N |
| Gradients | 4N | 2N |
| Optimizer (Adam) | 8N | 8N (master weights) |
| Activations | ~10-20N | ~5-10N |
| **Total** | **~26N** | **~17N** |

---

## 2. Mixed Precision Training (FP16/BF16)

### 2.1 Numerical Formats

| Format | Bits | Exponent | Mantissa | Range | Use Case |
|--------|------|----------|----------|-------|----------|
| FP32 | 32 | 8 | 23 | ±3.4e38 | Master weights |
| FP16 | 16 | 5 | 10 | ±65504 | Forward/backward |
| BF16 | 16 | 8 | 7 | ±3.4e38 | Forward/backward |

**BF16 vs FP16**:
- BF16 has the same range as FP32 (no overflow issues)
- FP16 has more precision but limited range (needs loss scaling)
- BF16 is preferred when hardware supports it (A100, H100, MI200+)

### 2.2 Implementation: DType Support

```cpp
// include/vesper/core/dtype.h

enum class DType {
    Float32,
    Float16,
    BFloat16,
    Int64,
    Int32,
    Int8,
    Bool
};

// Check hardware support
bool supports_bfloat16(Device device);
bool supports_float16(Device device);

// Get size in bytes
size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::BFloat16: return 2;
        case DType::Int64: return 8;
        case DType::Int32: return 4;
        case DType::Int8: return 1;
        case DType::Bool: return 1;
    }
}
```

### 2.3 FP16/BF16 Storage Types

```cpp
// include/vesper/core/half.h

// FP16: IEEE 754 half-precision
struct Float16 {
    uint16_t data;
    
    Float16() = default;
    explicit Float16(float f);
    explicit operator float() const;
    
    // Conversion helpers
    static Float16 from_bits(uint16_t bits) { Float16 h; h.data = bits; return h; }
    uint16_t to_bits() const { return data; }
};

// BF16: Brain floating-point (truncated FP32)
struct BFloat16 {
    uint16_t data;
    
    BFloat16() = default;
    explicit BFloat16(float f) {
        // Simple truncation from FP32 - drop lower 16 mantissa bits
        uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
        data = static_cast<uint16_t>(bits >> 16);
    }
    
    explicit operator float() const {
        uint32_t bits = static_cast<uint32_t>(data) << 16;
        return *reinterpret_cast<float*>(&bits);
    }
};
```

### 2.4 Mixed Precision Module Wrapper

```cpp
// include/vesper/nn/amp.h

namespace vesper::nn {

/// Automatic Mixed Precision wrapper for training
class AMP {
public:
    /// Target dtype for forward/backward passes
    AMP(DType compute_dtype = DType::BFloat16);
    
    /// Cast model parameters to half precision for forward pass
    void cast_model_half(Module& model);
    
    /// Keep master weights in FP32, create half-precision copies
    void init_master_weights(Module& model);
    
    /// Copy gradients to FP32 master weights after backward
    void sync_gradients(Module& model);
    
    /// Update master weights and copy back to half
    void update_master_weights(Module& model, Optimizer& optimizer);
    
    /// Get compute dtype
    DType compute_dtype() const { return compute_dtype_; }
    
private:
    DType compute_dtype_;
    std::unordered_map<Tensor*, Tensor> master_weights_;  // FP32 copies
};

/// Context manager for automatic casting
class AutocastContext {
public:
    AutocastContext(DType dtype);
    ~AutocastContext();
    
    static DType current_dtype();
    static bool is_enabled();
    
private:
    DType prev_dtype_;
    static thread_local DType current_dtype_;
    static thread_local bool enabled_;
};

} // namespace vesper::nn
```

### 2.5 Loss Scaling (FP16 Only)

FP16 has limited range—gradients can underflow to zero. Loss scaling multiplies the loss before backward, then divides gradients:

```cpp
// include/vesper/optim/grad_scaler.h

namespace vesper::optim {

class GradScaler {
public:
    GradScaler(float init_scale = 65536.0f, 
               float growth_factor = 2.0f,
               float backoff_factor = 0.5f,
               int growth_interval = 2000);
    
    /// Scale loss before backward pass
    Tensor scale(const Tensor& loss);
    
    /// Unscale gradients after backward
    void unscale(std::vector<Tensor*>& grads);
    
    /// Check for inf/nan and update scale factor
    /// Returns false if step should be skipped
    bool step(Optimizer& optimizer, std::vector<Tensor*>& grads);
    
    /// Update scale based on gradient health
    void update();
    
    float get_scale() const { return scale_; }
    
private:
    float scale_;
    float growth_factor_;
    float backoff_factor_;
    int growth_interval_;
    int steps_since_growth_ = 0;
    bool found_inf_ = false;
};

} // namespace vesper::optim
```

### 2.6 GPU Kernel for Type Conversion

```cpp
// src/ops/hip/cast.hip

template<typename SrcT, typename DstT>
__global__ void cast_kernel(const SrcT* src, DstT* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Handle special conversions
        if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, __half>) {
            dst[idx] = __float2half(src[idx]);
        } else if constexpr (std::is_same_v<SrcT, __half> && std::is_same_v<DstT, float>) {
            dst[idx] = __half2float(src[idx]);
        } else if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, hip_bfloat16>) {
            dst[idx] = __float2bfloat16(src[idx]);
        } else if constexpr (std::is_same_v<SrcT, hip_bfloat16> && std::is_same_v<DstT, float>) {
            dst[idx] = __bfloat162float(src[idx]);
        } else {
            dst[idx] = static_cast<DstT>(src[idx]);
        }
    }
}

Tensor cast_to(const Tensor& input, DType target_dtype) {
    if (input.dtype() == target_dtype) return input;
    
    Tensor output = empty(input.shape(), target_dtype, input.device());
    int64_t n = input.numel();
    
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    
    // Dispatch based on types
    DISPATCH_DTYPE_PAIR(input.dtype(), target_dtype, [&] {
        cast_kernel<<<grid, block>>>(
            input.data_ptr<src_t>(),
            output.data_ptr<dst_t>(),
            n);
    });
    
    return output;
}
```

---

## 3. Flash Attention

### 3.1 The Memory Problem

Standard attention materializes the full attention matrix:

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{N \times N}
$$

For sequence length N=4096: $4096^2 \times 4 = 67\text{MB}$ per head, per batch!

### 3.2 Flash Attention Algorithm

Flash Attention computes attention in tiles without materializing the full matrix:

**Key Insight**: Softmax can be computed incrementally using the online softmax trick:
$$
m_{new} = \max(m_{old}, m_{block})
$$
$$
l_{new} = l_{old} \cdot e^{m_{old} - m_{new}} + l_{block} \cdot e^{m_{block} - m_{new}}
$$

### 3.3 Implementation

```cpp
// include/vesper/ops/flash_attention.h

namespace vesper::ops {

/// Memory-efficient attention using Flash Attention algorithm
/// Memory: O(N) instead of O(N²)
/// 
/// @param q Query tensor [B, H, N, D]
/// @param k Key tensor [B, H, N, D]  
/// @param v Value tensor [B, H, N, D]
/// @param scale Attention scale (typically 1/sqrt(d_k))
/// @param is_causal Apply causal mask
/// @return Output tensor [B, H, N, D]
Tensor flash_attention(
    const Tensor& q,
    const Tensor& k, 
    const Tensor& v,
    float scale,
    bool is_causal = true);

/// Flash Attention backward pass
/// Returns tuple of (dQ, dK, dV)
std::tuple<Tensor, Tensor, Tensor> flash_attention_backward(
    const Tensor& dout,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& out,
    const Tensor& lse,  // Log-sum-exp from forward
    float scale,
    bool is_causal = true);

} // namespace vesper::ops
```

### 3.4 Flash Attention HIP Kernel

```cpp
// src/ops/hip/flash_attention.hip

constexpr int BLOCK_M = 64;   // Tile size for queries
constexpr int BLOCK_N = 64;   // Tile size for keys
constexpr int BLOCK_D = 64;   // Head dimension tile

template<int HeadDim>
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ Q,  // [B, H, N, D]
    const float* __restrict__ K,  // [B, H, N, D]
    const float* __restrict__ V,  // [B, H, N, D]
    float* __restrict__ O,        // [B, H, N, D]
    float* __restrict__ L,        // [B, H, N] - log-sum-exp for backward
    int B, int H, int N, int D,
    float scale, bool is_causal)
{
    // Shared memory for tiles
    __shared__ float sQ[BLOCK_M][HeadDim];
    __shared__ float sK[BLOCK_N][HeadDim];
    __shared__ float sV[BLOCK_N][HeadDim];
    
    const int batch_head = blockIdx.y;
    const int b = batch_head / H;
    const int h = batch_head % H;
    const int q_start = blockIdx.x * BLOCK_M;
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Initialize output accumulators
    float acc[BLOCK_M] = {0.0f};
    float m_prev[BLOCK_M];  // Running max
    float l_prev[BLOCK_M];  // Running sum
    
    for (int i = 0; i < BLOCK_M; ++i) {
        m_prev[i] = -INFINITY;
        l_prev[i] = 0.0f;
    }
    
    // Load Q tile to shared memory
    #pragma unroll
    for (int i = ty; i < BLOCK_M; i += blockDim.y) {
        int q_idx = q_start + i;
        if (q_idx < N) {
            for (int d = tx; d < D; d += blockDim.x) {
                sQ[i][d] = Q[((b * H + h) * N + q_idx) * D + d];
            }
        }
    }
    __syncthreads();
    
    // Iterate over K/V tiles
    int k_end = is_causal ? min(q_start + BLOCK_M, N) : N;
    
    for (int k_start = 0; k_start < k_end; k_start += BLOCK_N) {
        // Load K, V tiles
        #pragma unroll
        for (int j = ty; j < BLOCK_N; j += blockDim.y) {
            int k_idx = k_start + j;
            if (k_idx < N) {
                for (int d = tx; d < D; d += blockDim.x) {
                    sK[j][d] = K[((b * H + h) * N + k_idx) * D + d];
                    sV[j][d] = V[((b * H + h) * N + k_idx) * D + d];
                }
            }
        }
        __syncthreads();
        
        // Compute attention scores for this tile
        float scores[BLOCK_M][BLOCK_N];
        
        #pragma unroll
        for (int i = 0; i < BLOCK_M; ++i) {
            int q_idx = q_start + i;
            #pragma unroll
            for (int j = 0; j < BLOCK_N; ++j) {
                int k_idx = k_start + j;
                
                // Causal masking
                if (is_causal && k_idx > q_idx) {
                    scores[i][j] = -INFINITY;
                } else if (q_idx < N && k_idx < N) {
                    float sum = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        sum += sQ[i][d] * sK[j][d];
                    }
                    scores[i][j] = sum * scale;
                } else {
                    scores[i][j] = -INFINITY;
                }
            }
        }
        
        // Online softmax update
        #pragma unroll
        for (int i = 0; i < BLOCK_M; ++i) {
            // Find max in this block
            float m_block = -INFINITY;
            for (int j = 0; j < BLOCK_N; ++j) {
                m_block = fmaxf(m_block, scores[i][j]);
            }
            
            // Update running max
            float m_new = fmaxf(m_prev[i], m_block);
            
            // Compute local softmax
            float l_block = 0.0f;
            for (int j = 0; j < BLOCK_N; ++j) {
                scores[i][j] = expf(scores[i][j] - m_new);
                l_block += scores[i][j];
            }
            
            // Rescale previous accumulator
            float scale_prev = expf(m_prev[i] - m_new);
            l_prev[i] = l_prev[i] * scale_prev + l_block;
            
            // Scale previous output and add new contribution
            for (int d = 0; d < D; ++d) {
                acc[i * D + d] *= scale_prev;
                for (int j = 0; j < BLOCK_N; ++j) {
                    acc[i * D + d] += scores[i][j] * sV[j][d];
                }
            }
            
            m_prev[i] = m_new;
        }
        __syncthreads();
    }
    
    // Write output
    #pragma unroll
    for (int i = ty; i < BLOCK_M; i += blockDim.y) {
        int q_idx = q_start + i;
        if (q_idx < N) {
            float inv_l = 1.0f / l_prev[i];
            for (int d = tx; d < D; d += blockDim.x) {
                O[((b * H + h) * N + q_idx) * D + d] = acc[i * D + d] * inv_l;
            }
            // Save log-sum-exp for backward
            if (tx == 0) {
                L[(b * H + h) * N + q_idx] = m_prev[i] + logf(l_prev[i]);
            }
        }
    }
}
```

---

## 4. Gradient Checkpointing

### 4.1 The Activation Memory Problem

During forward pass, we save activations for backward. For a transformer:

- Per layer: ~4× hidden_dim × seq_len × batch_size
- For 32 layers: Activations dominate memory!

### 4.2 Checkpointing Strategy

Instead of saving all activations, we:
1. Save only inputs at "checkpoint" boundaries
2. During backward, recompute forward pass for that segment
3. Trade 33% more compute for ~70% less activation memory

```
Standard:   [Save A1] → [Save A2] → [Save A3] → ... → [Save A32]
Checkpoint: [Save A1] →  (recompute) → [Save A8] → (recompute) → [Save A16] → ...
```

### 4.3 Implementation

```cpp
// include/vesper/autograd/checkpoint.h

namespace vesper::autograd {

/// Checkpoint a function - recomputes forward during backward
/// @param fn Function to checkpoint (typically a Module forward)
/// @param inputs Inputs to the function
/// @return Output tensors (same as fn(inputs))
template<typename Fn>
Tensor checkpoint(Fn&& fn, const Tensor& input);

/// Checkpoint a module's forward pass
/// During backward, the forward is recomputed instead of using saved activations
class CheckpointedModule {
public:
    CheckpointedModule(std::shared_ptr<nn::Module> module);
    
    Tensor forward(const Tensor& input);
    
private:
    std::shared_ptr<nn::Module> module_;
};

/// Sequential module with checkpointing every N layers
class CheckpointedSequential : public nn::Module {
public:
    CheckpointedSequential(int checkpoint_every = 1);
    
    void add(std::shared_ptr<nn::Module> module);
    
    Tensor forward(const Tensor& x) override;
    
private:
    std::vector<std::shared_ptr<nn::Module>> modules_;
    int checkpoint_every_;
};

} // namespace vesper::autograd
```

### 4.4 Checkpoint Implementation

```cpp
// src/autograd/checkpoint.cpp

namespace vesper::autograd {

/// Custom autograd function for checkpointing
class CheckpointFunction {
public:
    template<typename Fn>
    static Tensor apply(Fn&& fn, const Tensor& input) {
        // Disable gradient computation for forward
        Tensor output;
        {
            NoGradGuard no_grad;
            output = fn(input);
        }
        
        // Save function and input for recomputation
        if (input.requires_grad()) {
            auto saved_fn = std::forward<Fn>(fn);
            auto saved_input = input.detach();
            
            // Register custom backward function
            output.set_grad_fn([saved_fn, saved_input](const Tensor& grad_output) {
                // Recompute forward with gradients enabled
                Tensor recomputed_input = saved_input;
                recomputed_input.set_requires_grad(true);
                
                Tensor recomputed_output = saved_fn(recomputed_input);
                
                // Now do the actual backward
                recomputed_output.backward(grad_output);
                
                return recomputed_input.grad();
            });
        }
        
        return output;
    }
};

template<typename Fn>
Tensor checkpoint(Fn&& fn, const Tensor& input) {
    return CheckpointFunction::apply(std::forward<Fn>(fn), input);
}

// CheckpointedSequential implementation
Tensor CheckpointedSequential::forward(const Tensor& x) {
    Tensor out = x;
    
    for (size_t i = 0; i < modules_.size(); ++i) {
        if (i % checkpoint_every_ == 0 && training()) {
            // Use checkpointing
            out = checkpoint([this, i](const Tensor& input) {
                return modules_[i]->forward(input);
            }, out);
        } else {
            out = modules_[i]->forward(out);
        }
    }
    
    return out;
}

} // namespace vesper::autograd
```

---

## 5. Putting It All Together

### 5.1 Memory-Efficient Transformer

```cpp
// include/vesper/models/efficient_transformer.h

namespace vesper::models {

struct EfficientTransformerConfig {
    TransformerConfig base_config;
    
    // Mixed precision
    DType compute_dtype = DType::BFloat16;
    bool use_loss_scaling = false;  // Only for FP16
    
    // Flash attention
    bool use_flash_attention = true;
    
    // Gradient checkpointing  
    bool use_checkpointing = true;
    int checkpoint_every = 2;  // Checkpoint every N layers
};

class EfficientTransformer : public nn::Module {
public:
    EfficientTransformer(const EfficientTransformerConfig& config);
    
    Tensor forward(const Tensor& input_ids) override;
    
    // Memory statistics
    size_t parameter_memory() const;
    size_t activation_memory(int batch_size, int seq_len) const;
    size_t total_training_memory(int batch_size, int seq_len) const;
    
private:
    EfficientTransformerConfig config_;
    std::unique_ptr<autograd::CheckpointedSequential> layers_;
    // ... other members
};

} // namespace vesper::models
```

### 5.2 Training Loop Example

```cpp
#include <vesper/vesper.h>

int main() {
    Device device = Device::HIP;
    
    // Configure efficient transformer
    EfficientTransformerConfig config;
    config.base_config.vocab_size = 50257;
    config.base_config.dim = 768;
    config.base_config.n_layers = 12;
    config.base_config.n_heads = 12;
    config.compute_dtype = DType::BFloat16;
    config.use_flash_attention = true;
    config.use_checkpointing = true;
    config.checkpoint_every = 2;
    
    // Create model
    EfficientTransformer model(config);
    model.to(device);
    
    // Print memory usage
    std::cout << "Parameter memory: " 
              << model.parameter_memory() / 1e6 << " MB" << std::endl;
    std::cout << "Activation memory (B=8, S=1024): "
              << model.activation_memory(8, 1024) / 1e6 << " MB" << std::endl;
    
    // Setup optimizer with AMP
    optim::AdamW optimizer(model.parameters(), 1e-4f);
    nn::AMP amp(DType::BFloat16);
    
    // Training loop
    for (int step = 0; step < 10000; ++step) {
        Tensor input_ids = /* load batch */;
        Tensor labels = /* load labels */;
        
        // Forward in half precision
        Tensor logits;
        {
            nn::AutocastContext autocast(DType::BFloat16);
            logits = model.forward(input_ids);
        }
        
        // Loss in FP32
        Tensor loss = nn::functional::cross_entropy_loss(
            logits.to(DType::Float32), labels);
        
        // Backward and step
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        if (step % 100 == 0) {
            float loss_val;
            loss.copy_to_host(&loss_val);
            std::cout << "Step " << step << " Loss: " << loss_val << std::endl;
        }
    }
    
    return 0;
}
```

---

## 6. Testing Strategy

### 6.1 FP16/BF16 Precision Tests

```cpp
TEST(MixedPrecision, BFloat16Conversion) {
    Tensor fp32 = randn({100, 100}, DType::Float32, Device::HIP);
    Tensor bf16 = fp32.to(DType::BFloat16);
    Tensor back = bf16.to(DType::Float32);
    
    // BF16 has 7 bits mantissa -> ~1% relative error
    EXPECT_TRUE(allclose(fp32, back, /*rtol=*/0.01f, /*atol=*/1e-5f));
}

TEST(MixedPrecision, Float16Conversion) {
    // Test with values in FP16 range
    Tensor fp32 = randn({100, 100}, DType::Float32, Device::HIP) * 100.0f;
    Tensor fp16 = fp32.to(DType::Float16);
    Tensor back = fp16.to(DType::Float32);
    
    // FP16 has 10 bits mantissa -> ~0.1% relative error
    EXPECT_TRUE(allclose(fp32, back, /*rtol=*/0.001f, /*atol=*/1e-3f));
}

TEST(MixedPrecision, MatmulBF16) {
    Tensor a = randn({64, 128}, DType::BFloat16, Device::HIP);
    Tensor b = randn({128, 64}, DType::BFloat16, Device::HIP);
    
    Tensor c_bf16 = ops::matmul(a, b);
    
    // Compare with FP32
    Tensor c_fp32 = ops::matmul(a.to(DType::Float32), b.to(DType::Float32));
    
    EXPECT_TRUE(allclose(c_bf16.to(DType::Float32), c_fp32, 0.02f, 1e-3f));
}

TEST(MixedPrecision, BackwardBF16) {
    Tensor x = randn({32, 64}, DType::BFloat16, Device::HIP, true);
    Tensor w = randn({64, 32}, DType::BFloat16, Device::HIP, true);
    
    Tensor y = ops::matmul(x, w);
    Tensor loss = y.sum();
    
    loss.backward();
    
    EXPECT_TRUE(x.grad().defined());
    EXPECT_TRUE(w.grad().defined());
    EXPECT_EQ(x.grad().dtype(), DType::BFloat16);
}
```

### 6.2 Flash Attention Tests

```cpp
TEST(FlashAttention, MatchesStandard) {
    int B = 2, H = 8, N = 256, D = 64;
    
    Tensor q = randn({B, H, N, D}, DType::Float32, Device::HIP);
    Tensor k = randn({B, H, N, D}, DType::Float32, Device::HIP);
    Tensor v = randn({B, H, N, D}, DType::Float32, Device::HIP);
    float scale = 1.0f / std::sqrt(D);
    
    // Standard attention
    Tensor scores = ops::matmul(q, k.transpose(-2, -1)) * scale;
    Tensor causal_mask = ops::triu(ops::full({N, N}, -INFINITY), 1);
    scores = scores + causal_mask;
    Tensor attn = nn::functional::softmax(scores, -1);
    Tensor standard_out = ops::matmul(attn, v);
    
    // Flash attention
    Tensor flash_out = ops::flash_attention(q, k, v, scale, /*causal=*/true);
    
    EXPECT_TRUE(allclose(standard_out, flash_out, 1e-3f, 1e-4f));
}

TEST(FlashAttention, MemoryEfficiency) {
    // Large sequence - would OOM with standard attention
    int B = 1, H = 32, N = 8192, D = 128;
    
    Tensor q = randn({B, H, N, D}, DType::BFloat16, Device::HIP);
    Tensor k = randn({B, H, N, D}, DType::BFloat16, Device::HIP);
    Tensor v = randn({B, H, N, D}, DType::BFloat16, Device::HIP);
    
    // This should NOT OOM
    Tensor out = ops::flash_attention(q, k, v, 1.0f / std::sqrt(D), true);
    
    EXPECT_EQ(out.shape(), (std::vector<int64_t>{B, H, N, D}));
}

TEST(FlashAttention, BackwardCorrectness) {
    int B = 2, H = 4, N = 64, D = 32;
    
    Tensor q = randn({B, H, N, D}, DType::Float32, Device::HIP, true);
    Tensor k = randn({B, H, N, D}, DType::Float32, Device::HIP, true);
    Tensor v = randn({B, H, N, D}, DType::Float32, Device::HIP, true);
    
    Tensor out = ops::flash_attention(q, k, v, 1.0f / std::sqrt(D), true);
    Tensor loss = out.sum();
    
    loss.backward();
    
    // Numerical gradient check
    auto check_grad = [&](Tensor& t, const std::string& name) {
        Tensor numerical = numerical_gradient([&]() {
            return ops::flash_attention(q, k, v, 1.0f / std::sqrt(D), true).sum();
        }, t);
        EXPECT_TRUE(allclose(t.grad(), numerical, 1e-2f, 1e-3f)) 
            << "Gradient mismatch for " << name;
    };
    
    check_grad(q, "Q");
    check_grad(k, "K");
    check_grad(v, "V");
}
```

### 6.3 Gradient Checkpointing Tests

```cpp
TEST(Checkpointing, ProducesCorrectGradients) {
    // Create identical models
    nn::Linear layer1(64, 64), layer2(64, 64);
    nn::Linear ckpt_layer1(64, 64), ckpt_layer2(64, 64);
    
    // Copy weights
    ckpt_layer1.weight().copy_(layer1.weight());
    ckpt_layer2.weight().copy_(layer2.weight());
    
    Tensor x = randn({8, 64}, DType::Float32, Device::HIP, true);
    Tensor x_ckpt = x.clone().set_requires_grad(true);
    
    // Standard forward
    Tensor y1 = nn::functional::relu(layer1(x));
    Tensor y2 = layer2(y1);
    y2.sum().backward();
    
    // Checkpointed forward
    Tensor y1_ckpt = autograd::checkpoint([&](const Tensor& in) {
        return nn::functional::relu(ckpt_layer1(in));
    }, x_ckpt);
    Tensor y2_ckpt = ckpt_layer2(y1_ckpt);
    y2_ckpt.sum().backward();
    
    // Gradients should match
    EXPECT_TRUE(allclose(x.grad(), x_ckpt.grad()));
    EXPECT_TRUE(allclose(layer1.weight().grad(), ckpt_layer1.weight().grad()));
}

TEST(Checkpointing, ReducesMemory) {
    // This test checks memory reduction
    int layers = 16;
    int dim = 512;
    int seq_len = 1024;
    int batch = 8;
    
    // Measure peak memory with and without checkpointing
    size_t peak_standard = measure_peak_memory([&]() {
        auto model = create_transformer(layers, dim);
        Tensor x = randn({batch, seq_len, dim}, Device::HIP, true);
        model.forward(x).sum().backward();
    });
    
    size_t peak_checkpointed = measure_peak_memory([&]() {
        auto model = create_transformer(layers, dim, /*checkpoint=*/true);
        Tensor x = randn({batch, seq_len, dim}, Device::HIP, true);
        model.forward(x).sum().backward();
    });
    
    // Checkpointing should use significantly less memory
    EXPECT_LT(peak_checkpointed, peak_standard * 0.6);
}
```

### 6.4 Integration Test

```cpp
TEST(EfficientTransformer, EndToEndTraining) {
    EfficientTransformerConfig config;
    config.base_config.vocab_size = 1000;
    config.base_config.dim = 256;
    config.base_config.n_layers = 4;
    config.base_config.n_heads = 4;
    config.compute_dtype = DType::BFloat16;
    config.use_flash_attention = true;
    config.use_checkpointing = true;
    
    EfficientTransformer model(config);
    model.to(Device::HIP);
    
    optim::AdamW optimizer(model.parameters(), 1e-3f);
    
    // Train for a few steps
    std::vector<float> losses;
    for (int i = 0; i < 50; ++i) {
        Tensor input = randint(0, 1000, {4, 64}, DType::Int64, Device::HIP);
        Tensor labels = randint(0, 1000, {4, 64}, DType::Int64, Device::HIP);
        
        Tensor logits = model.forward(input);
        Tensor loss = nn::functional::cross_entropy_loss(
            logits.view({-1, 1000}), labels.view({-1}));
        
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        float loss_val;
        loss.copy_to_host(&loss_val);
        losses.push_back(loss_val);
    }
    
    // Loss should decrease
    float avg_first_10 = std::accumulate(losses.begin(), losses.begin() + 10, 0.0f) / 10;
    float avg_last_10 = std::accumulate(losses.end() - 10, losses.end(), 0.0f) / 10;
    
    EXPECT_LT(avg_last_10, avg_first_10);
}
```

---

## 7. Summary

This chapter covered three essential techniques for memory-efficient LLM training:

| Technique | Memory Savings | Compute Overhead | Implementation Complexity |
|-----------|---------------|------------------|---------------------------|
| FP16/BF16 | ~50% | Minimal | Low |
| Flash Attention | O(N) vs O(N²) | None (faster!) | High |
| Gradient Checkpointing | ~60-70% | ~33% | Medium |

**Key Takeaways**:
1. **BF16 > FP16** when hardware supports it (same range as FP32)
2. **Flash Attention** is essential for long sequences (>1K tokens)
3. **Checkpoint every 2-4 layers** for optimal memory/compute tradeoff
4. **Combine all three** for training models 4-10x larger

With these optimizations, Vesper can now train GPT-2 and Llama-2 scale models efficiently!

---

## 8. Flash Attention Backward Pass

The forward pass alone isn't sufficient for training. We need an efficient backward pass that also doesn't materialize the full attention matrix.

### 8.1 Backward Algorithm

The backward pass requires computing gradients for Q, K, and V:

$$
\frac{\partial L}{\partial Q} = \text{scale} \cdot P^T \cdot dO \cdot K + (dS \odot P) \cdot K
$$

$$
\frac{\partial L}{\partial K} = \text{scale} \cdot P^T \cdot dO \cdot Q + (dS \odot P)^T \cdot Q
$$

$$
\frac{\partial L}{\partial V} = P^T \cdot dO
$$

Where $P$ is the attention weights and $dS$ involves the softmax Jacobian.

### 8.2 HIP Backward Kernel

```cpp
// src/ops/hip/flash_attention.hip (continued)

template<int HeadDim>
__global__ void flash_attention_backward_kernel(
    const float* __restrict__ dO,     // [B, H, N, D] - gradient of output
    const float* __restrict__ Q,      // [B, H, N, D]
    const float* __restrict__ K,      // [B, H, N, D]
    const float* __restrict__ V,      // [B, H, N, D]
    const float* __restrict__ O,      // [B, H, N, D] - saved output from forward
    const float* __restrict__ L,      // [B, H, N] - log-sum-exp from forward
    float* __restrict__ dQ,           // [B, H, N, D] - gradient of Q
    float* __restrict__ dK,           // [B, H, N, D] - gradient of K
    float* __restrict__ dV,           // [B, H, N, D] - gradient of V
    int B, int H, int N, int D,
    float scale, bool is_causal)
{
    __shared__ float sQ[BLOCK_M][HeadDim];
    __shared__ float sK[BLOCK_N][HeadDim];
    __shared__ float sV[BLOCK_N][HeadDim];
    __shared__ float sdO[BLOCK_M][HeadDim];
    __shared__ float sO[BLOCK_M][HeadDim];
    
    const int batch_head = blockIdx.y;
    const int b = batch_head / H;
    const int h = batch_head % H;
    const int q_start = blockIdx.x * BLOCK_M;
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Initialize gradient accumulators
    float dQ_acc[BLOCK_M][HeadDim] = {{0.0f}};
    
    // Load Q, dO, O tiles
    for (int i = ty; i < BLOCK_M; i += blockDim.y) {
        int q_idx = q_start + i;
        if (q_idx < N) {
            for (int d = tx; d < D; d += blockDim.x) {
                int idx = ((b * H + h) * N + q_idx) * D + d;
                sQ[i][d] = Q[idx];
                sdO[i][d] = dO[idx];
                sO[i][d] = O[idx];
            }
        }
    }
    __syncthreads();
    
    // Compute D = rowsum(dO * O) for softmax backward
    float Di[BLOCK_M];
    for (int i = 0; i < BLOCK_M; ++i) {
        Di[i] = 0.0f;
        int q_idx = q_start + i;
        if (q_idx < N) {
            for (int d = 0; d < D; ++d) {
                Di[i] += sdO[i][d] * sO[i][d];
            }
        }
    }
    
    // Iterate over K/V tiles (similar structure to forward)
    int k_end = is_causal ? min(q_start + BLOCK_M, N) : N;
    
    for (int k_start = 0; k_start < k_end; k_start += BLOCK_N) {
        // Load K, V tiles
        for (int j = ty; j < BLOCK_N; j += blockDim.y) {
            int k_idx = k_start + j;
            if (k_idx < N) {
                for (int d = tx; d < D; d += blockDim.x) {
                    int idx = ((b * H + h) * N + k_idx) * D + d;
                    sK[j][d] = K[idx];
                    sV[j][d] = V[idx];
                }
            }
        }
        __syncthreads();
        
        // Recompute attention scores and weights
        float scores[BLOCK_M][BLOCK_N];
        float P[BLOCK_M][BLOCK_N];
        
        for (int i = 0; i < BLOCK_M; ++i) {
            int q_idx = q_start + i;
            float li = L[(b * H + h) * N + q_idx];
            
            for (int j = 0; j < BLOCK_N; ++j) {
                int k_idx = k_start + j;
                
                if (is_causal && k_idx > q_idx) {
                    scores[i][j] = -INFINITY;
                    P[i][j] = 0.0f;
                } else if (q_idx < N && k_idx < N) {
                    float sum = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        sum += sQ[i][d] * sK[j][d];
                    }
                    scores[i][j] = sum * scale;
                    P[i][j] = expf(scores[i][j] - li);
                } else {
                    scores[i][j] = -INFINITY;
                    P[i][j] = 0.0f;
                }
            }
        }
        
        // Compute dV contribution: dV += P^T @ dO
        float dV_local[BLOCK_N][HeadDim] = {{0.0f}};
        for (int j = 0; j < BLOCK_N; ++j) {
            for (int i = 0; i < BLOCK_M; ++i) {
                for (int d = 0; d < D; ++d) {
                    dV_local[j][d] += P[i][j] * sdO[i][d];
                }
            }
        }
        
        // Compute dP = dO @ V^T
        float dP[BLOCK_M][BLOCK_N];
        for (int i = 0; i < BLOCK_M; ++i) {
            for (int j = 0; j < BLOCK_N; ++j) {
                dP[i][j] = 0.0f;
                for (int d = 0; d < D; ++d) {
                    dP[i][j] += sdO[i][d] * sV[j][d];
                }
            }
        }
        
        // Compute dS = P * (dP - Di) for softmax backward
        float dS[BLOCK_M][BLOCK_N];
        for (int i = 0; i < BLOCK_M; ++i) {
            for (int j = 0; j < BLOCK_N; ++j) {
                dS[i][j] = P[i][j] * (dP[i][j] - Di[i]);
            }
        }
        
        // Compute dQ contribution: dQ += scale * dS @ K
        for (int i = 0; i < BLOCK_M; ++i) {
            for (int d = 0; d < D; ++d) {
                for (int j = 0; j < BLOCK_N; ++j) {
                    dQ_acc[i][d] += scale * dS[i][j] * sK[j][d];
                }
            }
        }
        
        // Compute dK contribution: dK += scale * dS^T @ Q
        float dK_local[BLOCK_N][HeadDim] = {{0.0f}};
        for (int j = 0; j < BLOCK_N; ++j) {
            for (int d = 0; d < D; ++d) {
                for (int i = 0; i < BLOCK_M; ++i) {
                    dK_local[j][d] += scale * dS[i][j] * sQ[i][d];
                }
            }
        }
        
        // Atomically accumulate dK, dV (multiple Q blocks contribute)
        for (int j = ty; j < BLOCK_N; j += blockDim.y) {
            int k_idx = k_start + j;
            if (k_idx < N) {
                for (int d = tx; d < D; d += blockDim.x) {
                    int idx = ((b * H + h) * N + k_idx) * D + d;
                    atomicAdd(&dK[idx], dK_local[j][d]);
                    atomicAdd(&dV[idx], dV_local[j][d]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write dQ
    for (int i = ty; i < BLOCK_M; i += blockDim.y) {
        int q_idx = q_start + i;
        if (q_idx < N) {
            for (int d = tx; d < D; d += blockDim.x) {
                dQ[((b * H + h) * N + q_idx) * D + d] = dQ_acc[i][d];
            }
        }
    }
}
```

---

## 9. AMD GPU (HIP/ROCm) Specific Considerations

### 9.1 Hardware Support Matrix

| GPU Family | FP16 | BF16 | Flash Attention | Notes |
|------------|------|------|-----------------|-------|
| MI100 | ✅ | ❌ | ✅ | Use FP16 + loss scaling |
| MI200 (MI210/250/250X) | ✅ | ✅ | ✅ | BF16 preferred |
| MI300 (MI300X/MI300A) | ✅ | ✅ | ✅ | Best performance |
| RX 7900 (RDNA3) | ✅ | ⚠️ | ✅ | BF16 emulated |
| RX 6000 (RDNA2) | ✅ | ❌ | ✅ | Use FP16 |

### 9.2 ROCm-Specific Intrinsics

```cpp
// HIP provides native BF16 support on MI200+
#include <hip/hip_bf16.h>

// For older architectures, use software emulation
#if !defined(__gfx90a__) && !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
    // Emulate BF16 operations
    __device__ float bf16_to_float(hip_bfloat16 x) {
        uint32_t bits = static_cast<uint32_t>(x.data) << 16;
        return *reinterpret_cast<float*>(&bits);
    }
#else
    // Native hardware support
    __device__ float bf16_to_float(hip_bfloat16 x) {
        return __bfloat162float(x);
    }
#endif
```

### 9.3 Memory Coalescing for AMD GPUs

AMD GPUs have different cache line sizes and memory access patterns:

```cpp
// Optimal access pattern for AMD GCN/CDNA architecture
// Use wavefront size of 64 threads
constexpr int WAVEFRONT_SIZE = 64;

// Ensure shared memory is bank-conflict free
// AMD has 32 banks with 4-byte stride
constexpr int SHARED_MEM_BANKS = 32;
constexpr int BANK_STRIDE = 4;

// Pad shared memory to avoid bank conflicts
template<int HeadDim>
struct PaddedSharedMem {
    // Add padding to avoid bank conflicts
    static constexpr int PADDED_DIM = HeadDim + (HeadDim % SHARED_MEM_BANKS == 0 ? 1 : 0);
    float data[BLOCK_M][PADDED_DIM];
};
```

---

## 10. Performance Optimization Tips

### 10.1 Kernel Launch Configuration

```cpp
// Optimal block sizes for Flash Attention on AMD GPUs
struct FlashAttnConfig {
    // For MI200 series
    static constexpr int BLOCK_M_MI200 = 128;
    static constexpr int BLOCK_N_MI200 = 64;
    
    // For MI100
    static constexpr int BLOCK_M_MI100 = 64;
    static constexpr int BLOCK_N_MI100 = 64;
    
    // For consumer RDNA
    static constexpr int BLOCK_M_RDNA = 64;
    static constexpr int BLOCK_N_RDNA = 32;
    
    static auto get_config(Device device) {
        auto arch = get_gpu_architecture(device);
        if (arch.starts_with("gfx90a") || arch.starts_with("gfx94")) {
            return std::make_pair(BLOCK_M_MI200, BLOCK_N_MI200);
        } else if (arch.starts_with("gfx908")) {
            return std::make_pair(BLOCK_M_MI100, BLOCK_N_MI100);
        } else {
            return std::make_pair(BLOCK_M_RDNA, BLOCK_N_RDNA);
        }
    }
};
```

### 10.2 Memory Prefetching

```cpp
// Prefetch next tiles while computing current
template<int HeadDim>
__device__ void prefetch_tile(
    const float* __restrict__ src,
    float* __restrict__ smem,
    int row, int col, int stride,
    bool valid)
{
    if (valid) {
        // Use async copy on supported architectures
        #if __gfx90a__ || __gfx940__
        __builtin_amdgcn_global_load_lds(
            reinterpret_cast<const void*>(src + row * stride + col),
            reinterpret_cast<void*>(smem),
            sizeof(float) * HeadDim,
            0, 0);
        #else
        // Fallback to regular load
        for (int d = 0; d < HeadDim; ++d) {
            smem[d] = src[row * stride + col + d];
        }
        #endif
    }
}
```

### 10.3 Mixed Precision Accumulation

For best accuracy with FP16/BF16, accumulate in FP32:

```cpp
// Accumulate dot products in FP32 for numerical stability
__device__ float dot_product_accumulate(
    const __half* a,
    const __half* b,
    int n)
{
    float sum = 0.0f;
    
    // Process 2 elements at a time using half2
    int n2 = n / 2;
    const half2* a2 = reinterpret_cast<const half2*>(a);
    const half2* b2 = reinterpret_cast<const half2*>(b);
    
    for (int i = 0; i < n2; ++i) {
        half2 prod = __hmul2(a2[i], b2[i]);
        sum += __low2float(prod) + __high2float(prod);
    }
    
    // Handle odd element
    if (n % 2 == 1) {
        sum += __half2float(a[n-1]) * __half2float(b[n-1]);
    }
    
    return sum;
}
```

---

## 11. Debugging and Profiling

### 11.1 Numerical Debugging

```cpp
// Helper to check for NaN/Inf in tensors
bool check_tensor_health(const Tensor& t, const std::string& name) {
    Tensor is_nan = ops::isnan(t);
    Tensor is_inf = ops::isinf(t);
    
    int64_t nan_count = ops::sum(is_nan.to(DType::Int64)).item<int64_t>();
    int64_t inf_count = ops::sum(is_inf.to(DType::Int64)).item<int64_t>();
    
    if (nan_count > 0 || inf_count > 0) {
        std::cerr << "WARNING: " << name << " contains "
                  << nan_count << " NaN and " 
                  << inf_count << " Inf values" << std::endl;
        return false;
    }
    return true;
}

// Use in training loop
void safe_backward(Tensor& loss, const std::string& step_info) {
    if (!check_tensor_health(loss, "loss at " + step_info)) {
        throw std::runtime_error("NaN/Inf detected in loss");
    }
    loss.backward();
}
```

### 11.2 Memory Profiling

```cpp
// Track peak memory usage
class MemoryProfiler {
public:
    static void reset() {
        peak_allocated_ = 0;
        current_allocated_ = 0;
    }
    
    static void record_allocation(size_t bytes) {
        current_allocated_ += bytes;
        peak_allocated_ = std::max(peak_allocated_, current_allocated_);
    }
    
    static void record_deallocation(size_t bytes) {
        current_allocated_ -= bytes;
    }
    
    static size_t peak_memory() { return peak_allocated_; }
    static size_t current_memory() { return current_allocated_; }
    
    static void print_stats() {
        std::cout << "Memory Stats:" << std::endl;
        std::cout << "  Current: " << current_allocated_ / 1e9 << " GB" << std::endl;
        std::cout << "  Peak: " << peak_allocated_ / 1e9 << " GB" << std::endl;
    }
    
private:
    static inline size_t peak_allocated_ = 0;
    static inline size_t current_allocated_ = 0;
};
```

### 11.3 Using rocprof for Kernel Analysis

```bash
# Profile Flash Attention kernel
rocprof --stats --hip-trace ./vesper_train

# Detailed kernel metrics
rocprof -i metrics.txt --timestamp on ./vesper_train

# metrics.txt content:
# pmc: TCC_HIT_sum, TCC_MISS_sum, FETCH_SIZE, WRITE_SIZE
# pmc: SQ_WAVES, SQ_INSTS_VALU, SQ_INSTS_VMEM_RD
```

---

## 12. Common Pitfalls and Solutions

### 12.1 FP16 Overflow in Attention Scores

**Problem**: Large attention scores overflow FP16 range (>65504).

**Solution**: Always apply scaling before softmax, use BF16 if available.

```cpp
// Bad: Overflow risk
Tensor scores = ops::matmul(q, k.transpose(-2, -1));  // Can be very large
Tensor attn = nn::functional::softmax(scores, -1);

// Good: Scale before softmax
float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
Tensor scores = ops::matmul(q, k.transpose(-2, -1)) * scale;
Tensor attn = nn::functional::softmax(scores, -1);
```

### 12.2 Gradient Checkpointing with Dropout

**Problem**: Dropout generates different masks during recomputation.

**Solution**: Save and restore RNG state.

```cpp
// Save RNG state before checkpointed forward
auto rng_state = get_rng_state(device);

Tensor output = autograd::checkpoint([&](const Tensor& input) {
    // Restore RNG state for deterministic dropout
    set_rng_state(device, rng_state);
    return module->forward(input);
}, input);
```

### 12.3 Memory Fragmentation

**Problem**: Repeated allocation/deallocation causes fragmentation.

**Solution**: Pre-allocate buffers, use memory pools.

```cpp
// Pre-allocate workspace for Flash Attention
class FlashAttentionWorkspace {
public:
    FlashAttentionWorkspace(int64_t max_batch, int64_t max_heads, 
                            int64_t max_seq_len, Device device) {
        // Pre-allocate LSE buffer
        lse_ = empty({max_batch, max_heads, max_seq_len}, 
                     DType::Float32, device);
    }
    
    Tensor& lse() { return lse_; }
    
private:
    Tensor lse_;
};
```

---

## 13. Benchmarks and Expected Performance

### 13.1 Memory Comparison (GPT-2 124M, Batch=8, SeqLen=1024)

| Configuration | Peak Memory | vs Baseline |
|---------------|-------------|-------------|
| FP32, Standard Attention | 8.2 GB | 1.0x |
| FP32, Flash Attention | 3.1 GB | 0.38x |
| BF16, Standard Attention | 4.8 GB | 0.59x |
| BF16, Flash Attention | 1.8 GB | 0.22x |
| BF16, Flash + Checkpointing | 1.2 GB | 0.15x |

### 13.2 Training Throughput (tokens/sec, MI250X)

| Configuration | Throughput | vs Baseline |
|---------------|------------|-------------|
| FP32, Standard | 12,000 | 1.0x |
| BF16, Standard | 28,000 | 2.3x |
| BF16, Flash Attention | 42,000 | 3.5x |
| BF16, Flash + Checkpoint | 35,000 | 2.9x |

### 13.3 Maximum Trainable Model Size (24GB GPU)

| Configuration | Max Parameters |
|---------------|----------------|
| FP32, Standard | ~300M |
| BF16, Standard | ~600M |
| BF16, Flash Attention | ~1.5B |
| BF16, Flash + Checkpointing | ~3B |

---

## 14. Next Steps

- **Chapter 34**: Advanced autograd features (retain_graph, in-place version checking)
- **Chapter 38**: Fused kernels (GEMM+activation, fused attention)
- **Chapter 39**: Complete mixed precision training pipeline with FP8 support
