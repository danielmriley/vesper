# Chapter 30: Normalization and Stability

## 1. Introduction

Deep transformers are notoriously difficult to train without proper normalization and stable
activation functions. Gradient magnitudes can explode or vanish across hundreds of layers,
causing training instability. This chapter covers the essential components for stable training:

- **Layer Normalization**: Standard normalization for transformers (BERT, GPT-2).
- **RMS Normalization**: A faster variant used in Llama, Gopher, and Chinchilla.
- **Stable Softmax**: Numerically robust softmax for attention mechanisms.
- **GELU Activation**: The default non-linearity in modern transformers.

---

## 2. Layer Normalization (`nn::LayerNorm`)

### Mathematical Formulation

Normalizes input across the last $D$ dimensions (the `normalized_shape`):

$$ y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta $$

Where:
- $\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$ — mean over normalized dimensions
- $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2$ — variance over normalized dimensions
- $\gamma, \beta$ — learnable affine parameters (scale and shift)
- $\epsilon$ — small constant for numerical stability (default: `1e-5`)

### Backward Pass

The gradient computation requires chain rule through the normalization:

$$ \frac{\partial L}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}} \left( \frac{\partial L}{\partial y_i} - \frac{1}{n}\sum_j \frac{\partial L}{\partial y_j} - \frac{\hat{x}_i}{n} \sum_j \frac{\partial L}{\partial y_j} \hat{x}_j \right) $$

Where $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$ is the normalized input.

### API

```cpp
class LayerNorm : public Module {
public:
    LayerNorm(std::vector<int64_t> normalized_shape, float eps = 1e-5,
              bool elementwise_affine = true);
    Tensor forward(const Tensor& input) override;

    std::vector<int64_t> normalized_shape;
    float eps;
    bool elementwise_affine;
    Tensor weight;  // gamma, shape: normalized_shape
    Tensor bias;    // beta, shape: normalized_shape
};
```

### Usage

```cpp
auto ln = nn::LayerNorm({512});            // Normalize last dim of size 512
Tensor x = randn({32, 10, 512});           // [batch, seq, hidden]
Tensor y = ln.forward(x);                  // Output: mean≈0, std≈1 over last dim
```

---

## 3. RMS Normalization (`nn::RMSNorm`)

### Mathematical Formulation

RMSNorm omits mean subtraction, using only root-mean-square for normalization:

$$ y = \frac{x}{\text{RMS}(x)} \cdot \gamma $$

$$ \text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon} $$

**Advantages over LayerNorm**:
- ~10-15% faster (no mean computation or subtraction)
- Empirically performs comparably on LLM tasks
- Used in Llama, Llama 2, Mistral, and other modern architectures

### Backward Pass

$$ \frac{\partial L}{\partial x_i} = \frac{\gamma}{\text{RMS}(x)} \left( \frac{\partial L}{\partial y_i} - \frac{x_i}{n \cdot \text{RMS}(x)^2} \sum_j \frac{\partial L}{\partial y_j} x_j \right) $$

### API

```cpp
class RMSNorm : public Module {
public:
    RMSNorm(std::vector<int64_t> normalized_shape, float eps = 1e-5,
            bool elementwise_affine = true);
    Tensor forward(const Tensor& input) override;

    std::vector<int64_t> normalized_shape;
    float eps;
    bool elementwise_affine;
    Tensor weight;  // gamma only, no bias
};
```

---

## 4. Stable Softmax

### The Problem

Standard softmax is prone to overflow with large inputs:

$$ \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} $$

For $x_i = 1000$, $e^{1000}$ overflows to `inf`.

### The Solution

Subtract the maximum value before exponentiation (mathematically equivalent):

$$ \text{Softmax}(x_i) = \frac{e^{x_i - M}}{\sum_j e^{x_j - M}}, \quad M = \max_k(x_k) $$

### Kernel Implementation Strategy

For GPU kernels, use a two-pass or fused approach:

```
Pass 1: Find max M along dim
Pass 2: Compute exp(x - M), sum, and normalize
```

For attention with large sequence lengths, consider online softmax (single pass).

### API

```cpp
namespace nn::functional {
    Tensor softmax(const Tensor& input, int64_t dim);
    Tensor log_softmax(const Tensor& input, int64_t dim);  // For cross-entropy
}
```

---

## 5. GELU Activation

### Mathematical Formulation

Gaussian Error Linear Unit smoothly gates inputs:

**Exact form** (using error function):
$$ \text{GELU}(x) = 0.5x \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right) $$

**Tanh approximation** (faster, used in GPT-2):
$$ \text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right) $$

### Key Properties

| $x$   | $\text{GELU}(x)$ | Gradient |
|-------|------------------|----------|
| 0     | 0                | 0.5      |
| 1     | ≈0.841           | ≈1.08    |
| -1    | ≈-0.159          | ≈-0.08   |
| $+\infty$ | $x$          | 1        |
| $-\infty$ | 0            | 0        |

### API

```cpp
namespace nn::functional {
    Tensor gelu(const Tensor& input);           // Default: tanh approx
    Tensor gelu_tanh(const Tensor& input);      // Explicit tanh approx
    Tensor gelu_erf(const Tensor& input);       // Exact erf form
}
```

---

## 6. Implementation Details

### Kernel Considerations (HIP/CUDA)

| Component | CPU | GPU (HIP/CUDA) |
|-----------|-----|----------------|
| LayerNorm | Loop over batch, vectorize over features | Block-reduce for mean/var, shared memory |
| RMSNorm   | Similar to LayerNorm | Simpler: only sum-of-squares reduction |
| Softmax   | Loop, stable subtraction | Warp-reduce for max, parallel exp/sum |
| GELU      | Element-wise `erf()` or `tanh()` | Fused kernel, use fast intrinsics |

### Memory Layout

All normalization ops assume the normalized dimensions are **contiguous** in memory
(i.e., the last dimensions). For a tensor of shape `[B, S, H]`:
- `LayerNorm({H})` normalizes over the hidden dimension.
- `LayerNorm({S, H})` normalizes over sequence and hidden (less common).

---

## 7. Comprehensive Testing Strategy

### 7.1 LayerNorm Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_layer_norm_output_stats` | Random input, verify output mean≈0, var≈1 | `abs(mean) < 1e-5`, `abs(var - 1) < 1e-4` |
| `test_layer_norm_known_values` | Fixed input `[1,2,3,4,5]`, compare to manual calc | Exact match to hand-computed values |
| `test_layer_norm_affine` | Set gamma=2, beta=1, verify scaling | Output = 2*normalized + 1 |
| `test_layer_norm_no_affine` | `elementwise_affine=false` | No weight/bias registered |
| `test_layer_norm_backward` | Gradient check with finite differences | `abs(numerical - analytical) < 1e-3` |
| `test_layer_norm_consistency` | CPU vs HIP output match | `max_diff < 1e-4` |
| `test_layer_norm_3d_input` | Shape `[B, S, H]` | Correct broadcasting |
| `test_layer_norm_multi_dim` | `normalized_shape={S, H}` | Normalizes over 2 dims |

### 7.2 RMSNorm Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_rms_norm_known_values` | Input `[3, 4]`, RMS=√12.5 | Exact match |
| `test_rms_norm_vs_layer_norm` | Zero-mean input, compare outputs | Should differ (no mean subtraction in RMS) |
| `test_rms_norm_backward` | Gradient check | Finite diff match |
| `test_rms_norm_consistency` | CPU vs HIP | `max_diff < 1e-4` |
| `test_rms_norm_large_values` | Input with large magnitudes | No overflow, correct RMS |

### 7.3 Softmax Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_softmax_sum_to_one` | Random input | `sum(output, dim) == 1.0` |
| `test_softmax_known_values` | `[1, 2, 3]` → expected probs | Match to `exp(x)/sum(exp(x))` |
| `test_softmax_uniform` | `[10, 10, 10]` | Output = `[1/3, 1/3, 1/3]` |
| `test_softmax_stability_large` | Input with values ~1000 | No inf/nan, correct output |
| `test_softmax_stability_mixed` | `[1000, 0, -1000]` | First element ≈ 1.0 |
| `test_softmax_dim0` | Softmax along dim=0 | Correct axis handling |
| `test_softmax_backward` | Gradient check | Jacobian validation |
| `test_softmax_consistency` | CPU vs HIP | `max_diff < 1e-4` |

### 7.4 GELU Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_gelu_known_values` | x=0,1,-1 | GELU(0)=0, GELU(1)≈0.841, GELU(-1)≈-0.159 |
| `test_gelu_symmetry` | Compare GELU(x) and GELU(-x) | Not symmetric (unlike ReLU) |
| `test_gelu_large_positive` | x=10 | Output ≈ x |
| `test_gelu_large_negative` | x=-10 | Output ≈ 0 |
| `test_gelu_backward` | Gradient check at x=0,1,-1 | grad(0)=0.5, grad(1)≈1.08 |
| `test_gelu_tanh_vs_erf` | Compare approximations | `max_diff < 1e-3` |
| `test_gelu_consistency` | CPU vs HIP | `max_diff < 1e-4` |

### 7.5 Integration Tests

| Test Case | Description |
|-----------|-------------|
| `test_transformer_block_stability` | Stack 12 LayerNorm + Attention + GELU |
| `test_gradient_flow_deep` | 24-layer forward/backward, no nan/inf |
| `test_mixed_precision` | Float16 input with Float32 accumulation |

---

## 8. Example: Full Transformer Pre-Norm Block

```cpp
// Pre-LayerNorm architecture (GPT-2 style)
class TransformerBlock : public Module {
    LayerNorm ln1, ln2;
    MultiHeadAttention attn;
    MLP mlp;

public:
    TransformerBlock(int hidden, int heads, int mlp_dim)
        : ln1({hidden}), ln2({hidden}),
          attn(hidden, heads), mlp(hidden, mlp_dim) {}

    Tensor forward(const Tensor& x) override {
        // Pre-norm: normalize before attention
        auto h = x + attn(ln1(x));
        return h + mlp(ln2(h));
    }
};
```

---

## 9. References

1. Ba, Kiros, Hinton. "Layer Normalization" (2016)
2. Zhang, Sennrich. "Root Mean Square Layer Normalization" (2019)
3. Hendrycks, Gimpel. "Gaussian Error Linear Units (GELUs)" (2016)
4. Rabe, Staats. "Self-Attention Does Not Need O(n²) Memory" (2021) — Online softmax
