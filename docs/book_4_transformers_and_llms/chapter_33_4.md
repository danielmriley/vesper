```markdown
# Chapter 33.4: SwiGLU and FFN Variants

## 1. Introduction

The Feed-Forward Network (FFN) is the "workhorse" of the Transformer, processing each token's representation independently after attention. While the original Transformer used a simple two-layer MLP with ReLU, modern LLMs have converged on **SwiGLU** — a gated linear unit with the Swish activation.

### Evolution of Transformer FFN

| Model | FFN Type | Hidden Multiplier | Activation |
|-------|----------|-------------------|------------|
| Transformer (2017) | MLP | 4x | ReLU |
| GPT-2/3 | MLP | 4x | GELU |
| PaLM | SwiGLU | 8/3x (~2.67x) | Swish |
| Llama 1/2/3 | SwiGLU | 8/3x | SiLU |
| Mistral | SwiGLU | Variable | SiLU |

## 2. Mathematical Foundation

### 2.1 Original FFN (MLP)

$$
\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2
$$

With shapes:
- $W_1$: `[hidden_dim, d_model]` (typically hidden_dim = 4 × d_model)
- $W_2$: `[d_model, hidden_dim]`

### 2.2 Gated Linear Units (GLU)

GLU introduces a multiplicative gate:

$$
\text{GLU}(x) = (W_1 \cdot x) \otimes \sigma(W_2 \cdot x)
$$

Where $\sigma$ is the sigmoid function and $\otimes$ is element-wise multiplication.

### 2.3 SwiGLU

SwiGLU replaces sigmoid with **Swish** (also called **SiLU**):

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

The full SwiGLU FFN:

$$
\text{SwiGLU}(x) = (W_{\text{gate}} \cdot x \otimes \text{Swish}(W_{\text{up}} \cdot x)) \cdot W_{\text{down}}
$$

Or equivalently (Llama notation):

$$
\text{FFN}(x) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} \cdot x) \otimes (W_{\text{up}} \cdot x))
$$

### 2.4 Why 8/3 Hidden Dimension?

With GLU-style FFNs, we have 3 weight matrices instead of 2. To maintain parameter count parity with the original 4x FFN:

$$
\text{Original: } 2 \times d \times 4d = 8d^2
$$
$$
\text{GLU: } 3 \times d \times h = 8d^2 \implies h = \frac{8d}{3} \approx 2.67d
$$

In practice, Llama rounds to nice multiples (e.g., for d=4096, hidden=11008).

## 3. Implementation Plan

### 3.1 SiLU Activation

```cpp
// include/vesper/ops/activations.h

namespace vesper::ops {

// SiLU / Swish activation
// silu(x) = x * sigmoid(x)
Tensor silu(const Tensor& x);

// In-place version
void silu_(Tensor& x);

} // namespace vesper::ops
```

### 3.2 SiLU Kernel

```cpp
// src/ops/hip/activation.hip

__global__ void silu_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid_x;
    }
}

// Vectorized version
__global__ void silu_kernel_vec4(const float4* __restrict__ input,
                                  float4* __restrict__ output,
                                  int64_t vec_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_size) {
        float4 x = input[idx];
        float4 result;
        
        result.x = x.x / (1.0f + expf(-x.x));
        result.y = x.y / (1.0f + expf(-x.y));
        result.z = x.z / (1.0f + expf(-x.z));
        result.w = x.w / (1.0f + expf(-x.w));
        
        output[idx] = result;
    }
}
```

### 3.3 SwiGLU MLP Module

```cpp
// include/vesper/nn/swiglu_mlp.h

namespace vesper::nn {

class SwiGLUMLP : public Module {
public:
    SwiGLUMLP(int64_t d_model, int64_t hidden_dim, bool bias = false);
    
    Tensor forward(const Tensor& x) override;
    
private:
    Linear w_gate_;  // Gate projection
    Linear w_up_;    // Up projection  
    Linear w_down_;  // Down projection
    int64_t hidden_dim_;
};

} // namespace vesper::nn
```

### 3.4 SwiGLU Implementation

```cpp
// src/nn/swiglu_mlp.cpp

namespace vesper::nn {

SwiGLUMLP::SwiGLUMLP(int64_t d_model, int64_t hidden_dim, bool bias)
    : hidden_dim_(hidden_dim)
{
    // Llama-style naming
    w_gate_ = register_module("gate_proj", Linear(d_model, hidden_dim, bias));
    w_up_   = register_module("up_proj",   Linear(d_model, hidden_dim, bias));
    w_down_ = register_module("down_proj", Linear(hidden_dim, d_model, bias));
}

Tensor SwiGLUMLP::forward(const Tensor& x) {
    // x: [Batch, SeqLen, D]
    
    // 1. Gate and Up projections (can be done in parallel)
    Tensor gate = w_gate_(x);   // [B, S, Hidden]
    Tensor up = w_up_(x);       // [B, S, Hidden]
    
    // 2. SiLU activation on gate
    ops::silu_(gate);           // In-place
    
    // 3. Element-wise multiply
    Tensor hidden = gate * up;  // [B, S, Hidden]
    
    // 4. Down projection
    return w_down_(hidden);     // [B, S, D]
}

} // namespace vesper::nn
```

### 3.5 Fused Gate-Up Projection (Optimization)

For efficiency, we can combine gate and up projections into a single GEMM:

```cpp
// Optimized SwiGLU with fused projections
class SwiGLUMLPFused : public Module {
public:
    SwiGLUMLPFused(int64_t d_model, int64_t hidden_dim, bool bias = false)
        : hidden_dim_(hidden_dim)
    {
        // Combined gate+up projection
        w_gate_up_ = register_module("gate_up_proj", 
            Linear(d_model, 2 * hidden_dim, bias));
        w_down_ = register_module("down_proj", 
            Linear(hidden_dim, d_model, bias));
    }
    
    Tensor forward(const Tensor& x) override {
        // 1. Fused gate+up projection
        Tensor gate_up = w_gate_up_(x);  // [B, S, 2*Hidden]
        
        // 2. Split along last dim
        auto chunks = gate_up.chunk(2, /*dim=*/-1);
        Tensor gate = chunks[0];  // [B, S, Hidden]
        Tensor up = chunks[1];    // [B, S, Hidden]
        
        // 3. SiLU on gate, multiply with up
        ops::silu_(gate);
        Tensor hidden = gate * up;
        
        // 4. Down projection
        return w_down_(hidden);
    }
    
private:
    Linear w_gate_up_;
    Linear w_down_;
    int64_t hidden_dim_;
};
```

## 4. Other FFN Variants

### 4.1 GeGLU (GELU-Gated)

Used in some models like Gemma:

```cpp
Tensor geglu(const Tensor& x, const Tensor& gate) {
    return ops::gelu(gate) * x;
}
```

### 4.2 ReGLU (ReLU-Gated)

Simpler variant:

```cpp
Tensor reglu(const Tensor& x, const Tensor& gate) {
    return ops::relu(gate) * x;
}
```

### 4.3 GEGLU Kernel (Fused)

```cpp
__global__ void geglu_fused_kernel(
    const float* __restrict__ gate_up,  // [N, 2*hidden]
    float* __restrict__ output,          // [N, hidden]
    int64_t N, int64_t hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * hidden_dim) return;
    
    int n = idx / hidden_dim;
    int h = idx % hidden_dim;
    
    // Gate is first half, up is second half
    float gate_val = gate_up[n * 2 * hidden_dim + h];
    float up_val = gate_up[n * 2 * hidden_dim + hidden_dim + h];
    
    // GELU approximation
    float gelu_gate = 0.5f * gate_val * (1.0f + tanhf(
        0.7978845608f * (gate_val + 0.044715f * gate_val * gate_val * gate_val)));
    
    output[idx] = gelu_gate * up_val;
}
```

## 5. Backward Pass

### 5.1 SiLU Backward

The derivative of SiLU:

$$
\frac{d}{dx}\text{SiLU}(x) = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x))
$$
$$
= \sigma(x)(1 + x(1 - \sigma(x)))
$$

```cpp
__global__ void silu_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        
        // d_silu/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        float grad = sigmoid_x * (1.0f + x * (1.0f - sigmoid_x));
        
        grad_in[idx] = grad_out[idx] * grad;
    }
}
```

### 5.2 SwiGLU Backward

For `output = SiLU(gate) * up`:

```cpp
// grad_gate = grad_out * up * d_silu(gate)
// grad_up = grad_out * SiLU(gate)

void swiglu_backward(const Tensor& grad_out, const Tensor& gate, 
                     const Tensor& up, Tensor& grad_gate, Tensor& grad_up) {
    // SiLU(gate) for grad_up computation
    Tensor silu_gate = ops::silu(gate);
    
    // d_SiLU/d_gate
    Tensor sigmoid_gate = ops::sigmoid(gate);
    Tensor d_silu = sigmoid_gate * (1.0f + gate * (1.0f - sigmoid_gate));
    
    // Gradients
    grad_gate = grad_out * up * d_silu;
    grad_up = grad_out * silu_gate;
}
```

## 6. Autograd Integration

```cpp
// src/autograd/swiglu.cpp

class SiLUFunction : public autograd::Function {
public:
    static Tensor forward(AutogradContext& ctx, const Tensor& input) {
        ctx.save_for_backward({input});
        return ops::silu(input);
    }
    
    static std::vector<Tensor> backward(AutogradContext& ctx, 
                                          const std::vector<Tensor>& grads) {
        auto saved = ctx.get_saved_tensors();
        Tensor input = saved[0];
        Tensor grad_out = grads[0];
        
        Tensor sigmoid_x = ops::sigmoid(input);
        Tensor grad_in = grad_out * sigmoid_x * (1.0f + input * (1.0f - sigmoid_x));
        
        return {grad_in};
    }
};
```

## 7. Testing Strategy

### 7.1 Unit Tests

```cpp
// tests/nn/test_swiglu.cpp

TEST(SiLU, ForwardCorrectness) {
    Tensor x = tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    Tensor y = ops::silu(x);
    
    // Manual computation
    // silu(x) = x * sigmoid(x)
    auto sigmoid = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    
    float* data = y.data_ptr<float>();
    EXPECT_NEAR(data[0], -2.0f * sigmoid(-2.0f), 1e-5);  // -0.2384
    EXPECT_NEAR(data[1], -1.0f * sigmoid(-1.0f), 1e-5);  // -0.2689
    EXPECT_NEAR(data[2], 0.0f, 1e-5);                     // 0.0
    EXPECT_NEAR(data[3], 1.0f * sigmoid(1.0f), 1e-5);    // 0.7311
    EXPECT_NEAR(data[4], 2.0f * sigmoid(2.0f), 1e-5);    // 1.7616
}

TEST(SiLU, MonotonicityProperty) {
    // SiLU is monotonically increasing for x > ~-1.278
    Tensor x = linspace(0.0f, 10.0f, 100);
    Tensor y = ops::silu(x);
    
    float* data = y.data_ptr<float>();
    for (int i = 1; i < 100; ++i) {
        EXPECT_GT(data[i], data[i-1]);
    }
}

TEST(SwiGLUMLP, OutputShape) {
    SwiGLUMLP mlp(512, 1376);  // Typical hidden = 8/3 * 512 ≈ 1365, rounded
    
    Tensor x = randn({2, 16, 512});
    Tensor y = mlp(x);
    
    EXPECT_EQ(y.shape(), std::vector<int64_t>({2, 16, 512}));
}

TEST(SwiGLUMLP, ParameterCount) {
    int64_t d = 512;
    int64_t h = 1376;
    
    SwiGLUMLP mlp(d, h);
    
    int64_t expected_params = 3 * d * h;  // gate + up + down (no bias)
    int64_t actual_params = 0;
    for (auto& p : mlp.parameters()) {
        actual_params += p.numel();
    }
    
    EXPECT_EQ(actual_params, expected_params);
}

TEST(SwiGLUMLP, GradientFlow) {
    SwiGLUMLP mlp(64, 170);
    Tensor x = randn({1, 4, 64}, /*requires_grad=*/true);
    
    Tensor y = mlp(x);
    Tensor loss = y.sum();
    loss.backward();
    
    // All parameters should have gradients
    for (auto& p : mlp.parameters()) {
        EXPECT_TRUE(p.grad().defined());
        EXPECT_FALSE(p.grad().isnan().any().item<bool>());
    }
    
    // Input should have gradient
    EXPECT_TRUE(x.grad().defined());
}
```

### 7.2 Stress Tests

```cpp
TEST(SwiGLU, StressTest_LargeScale) {
    // Llama 7B dimensions
    int64_t d_model = 4096;
    int64_t hidden = 11008;
    int64_t batch = 4;
    int64_t seq_len = 2048;
    
    SwiGLUMLP mlp(d_model, hidden);
    mlp.to(Device::HIP);
    
    Tensor x = randn({batch, seq_len, d_model}, Device::HIP);
    
    // Warm-up
    Tensor y = mlp(x);
    hipDeviceSynchronize();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        y = mlp(x);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "SwiGLU [4, 2048, 4096] x10: " << ms << " ms" << std::endl;
    
    // Should be fast (< 1 second for 10 iterations on modern GPU)
    EXPECT_LT(ms, 1000);
}

TEST(SwiGLU, StressTest_NumericalStability) {
    // Test with extreme values
    SwiGLUMLP mlp(64, 170);
    
    // Large positive values
    Tensor x_pos = ones({1, 4, 64}) * 100.0f;
    Tensor y_pos = mlp(x_pos);
    EXPECT_FALSE(y_pos.isnan().any().item<bool>());
    EXPECT_FALSE(y_pos.isinf().any().item<bool>());
    
    // Large negative values
    Tensor x_neg = ones({1, 4, 64}) * -100.0f;
    Tensor y_neg = mlp(x_neg);
    EXPECT_FALSE(y_neg.isnan().any().item<bool>());
    EXPECT_FALSE(y_neg.isinf().any().item<bool>());
    
    // Mixed extreme values
    Tensor x_mixed = randn({1, 4, 64}) * 50.0f;
    Tensor y_mixed = mlp(x_mixed);
    EXPECT_FALSE(y_mixed.isnan().any().item<bool>());
}

TEST(SwiGLU, StressTest_BackwardMemory) {
    // Test that gradients don't cause memory explosion
    SwiGLUMLP mlp(1024, 2730);
    mlp.to(Device::HIP);
    
    size_t initial_memory = get_hip_memory_usage();
    
    for (int i = 0; i < 100; ++i) {
        Tensor x = randn({8, 512, 1024}, Device::HIP, /*requires_grad=*/true);
        Tensor y = mlp(x);
        Tensor loss = y.mean();
        loss.backward();
        
        mlp.zero_grad();
    }
    
    size_t final_memory = get_hip_memory_usage();
    
    // Memory should not grow significantly (allow 10% variance)
    EXPECT_LT(final_memory, initial_memory * 1.1);
}
```

### 7.3 Comparison Tests

```cpp
TEST(SwiGLU, CompareWithPyTorch) {
    // Values computed from PyTorch reference
    Tensor gate = tensor({{1.0f, 2.0f}, {-1.0f, 0.5f}});
    Tensor up = tensor({{0.5f, 1.0f}, {2.0f, -1.0f}});
    
    // PyTorch: F.silu(gate) * up
    // Expected (pre-computed):
    Tensor expected = tensor({{0.3655f, 1.7616f}, {-0.5378f, -0.3112f}});
    
    Tensor silu_gate = ops::silu(gate);
    Tensor result = silu_gate * up;
    
    EXPECT_TRUE(allclose(result, expected, 1e-4, 1e-4));
}
```

## 8. Fused SwiGLU Kernel (Advanced)

For maximum performance, fuse the entire gate-silu-multiply operation:

```cpp
// Fused: output = SiLU(gate_proj(x)) * up_proj(x)
// Assumes gate and up are already computed and interleaved

__global__ void swiglu_fused_kernel(
    const float* __restrict__ gate_up,   // [N, 2*hidden]
    float* __restrict__ output,           // [N, hidden]
    int64_t N, int64_t hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * hidden_dim) return;
    
    int n = idx / hidden_dim;
    int h = idx % hidden_dim;
    
    // Interleaved layout: [gate0, up0, gate1, up1, ...]
    // Or split layout: [gate0..gateH, up0..upH]
    // Using split layout:
    int gate_idx = n * 2 * hidden_dim + h;
    int up_idx = n * 2 * hidden_dim + hidden_dim + h;
    
    float gate_val = gate_up[gate_idx];
    float up_val = gate_up[up_idx];
    
    // SiLU
    float sigmoid_gate = 1.0f / (1.0f + expf(-gate_val));
    float silu_gate = gate_val * sigmoid_gate;
    
    output[idx] = silu_gate * up_val;
}
```

## 9. Performance Considerations

### Memory Bandwidth Analysis

For a single SwiGLU forward pass with batch B, seq S, dim D, hidden H:

| Operation | Read | Write | Total |
|-----------|------|-------|-------|
| Gate proj | B×S×D + D×H | B×S×H | ~2×B×S×H |
| Up proj | B×S×D + D×H | B×S×H | ~2×B×S×H |
| SiLU | B×S×H | B×S×H | 2×B×S×H |
| Multiply | 2×B×S×H | B×S×H | 3×B×S×H |
| Down proj | B×S×H + H×D | B×S×D | ~2×B×S×H |

**Total: ~11 × B × S × H memory operations**

With fusion, we can reduce SiLU+Multiply to a single fused kernel, saving ~3 × B × S × H.

## 10. Summary

SwiGLU is the state-of-the-art FFN architecture for LLMs, combining:
- **Gating**: Multiplicative interaction allows dynamic routing of information
- **Swish/SiLU**: Smooth, non-monotonic activation with beneficial gradient properties
- **Parameter Efficiency**: 3 matrices with 8/3x hidden maintains similar parameter count

Key implementation points:
1. Fuse gate+up projection into single GEMM when possible
2. Implement vectorized SiLU kernel for GPU
3. Consider fused SwiGLU kernel for memory bandwidth savings
4. Handle large inputs gracefully (no overflow in exp)

With RoPE (Chapter 33.3) and SwiGLU, Vesper now has the core components for Llama-style models.

```
