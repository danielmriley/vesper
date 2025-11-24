# Chapter 30: Normalization and Stability

## 1. Introduction

Deep transformers are notoriously difficult to train without proper normalization and stable activation functions. This chapter implements the essential components for stability.

## 2. Components

### Layer Normalization (`nn.LayerNorm`)
Normalizes input across the feature dimension (independent of batch size).
$$ y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta $$
-   $\mu, \sigma^2$: Mean and variance of $x$ along the last dimension.
-   $\gamma, \beta$: Learnable affine parameters.

### RMS Normalization (`nn.RMSNorm`)
A simplified LayerNorm used in Llama, Gopher, and Chinchilla. It re-scales invariance but doesn't re-center (no mean subtraction).
$$ y = \frac{x}{\text{RMS}(x)} \cdot \gamma $$
$$ \text{RMS}(x) = \sqrt{\frac{1}{n} \sum x_i^2 + \epsilon} $$

**Backward Pass Note**: The gradient for RMSNorm is simpler than LayerNorm but still requires careful derivation.
$$ \frac{\partial L}{\partial x} = \frac{1}{\text{RMS}(x)} \left( \frac{\partial L}{\partial y} - \frac{x}{\text{RMS}(x)^2} \sum \left( \frac{\partial L}{\partial y} \cdot x \right) \frac{1}{n} \right) \cdot \gamma $$

### Stable Softmax
The standard softmax is prone to overflow if inputs are large.
$$ \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}} $$
**Stable trick**: Subtract the max value $M = \max(x)$ from all inputs.
$$ \text{Softmax}(x_i) = \frac{e^{x_i - M}}{\sum e^{x_j - M}} $$

### GELU Activation
Gaussian Error Linear Unit. Approximates $x \Phi(x)$.
$$ \text{GELU}(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3))) $$
**Exact GELU**: Some models use the exact error function formulation:
$$ \text{GELU}(x) = 0.5x (1 + \text{erf}(x / \sqrt{2})) $$
We should implement both `gelu_tanh` (approx) and `gelu_erf` (exact) variants.


## 3. Implementation Plan

### `nn::LayerNorm`
```cpp
class LayerNorm : public Module {
public:
    LayerNorm(std::vector<int64_t> normalized_shape, float eps=1e-5);
    Tensor forward(const Tensor& input) override;
    Tensor weight; // gamma
    Tensor bias;   // beta
};
```

### `ops::softmax`
Implement a kernel that:
1.  Finds max along dim.
2.  Subtracts max.
3.  Exps.
4.  Sums.
5.  Divides.

## 4. Usage Example

```cpp
auto ln = nn::LayerNorm({512});
Tensor x = randn({32, 10, 512});
Tensor y = ln(x); // Output has mean 0, std 1 over last dim
```

## 5. Testing Strategy

1.  **LayerNorm**: Verify output mean is close to 0 and variance close to 1 for random input (with gamma=1, beta=0).
2.  **Softmax**: Verify output sums to 1.0 along the specified dimension.
