# Chapter 26: Weight Initialization (`nn.init`)

## 1. Introduction

Weight initialization is a critical step in training deep neural networks. If weights are initialized too small, the signal shrinks as it passes through each layer until it's too tiny to be useful (vanishing gradients). If weights are initialized too large, the signal grows until it's massive and unusable (exploding gradients).

In this chapter, we implement a suite of initialization strategies in the `vesper::nn::init` namespace. These functions modify tensors in-place.

## 2. Mathematical Foundations

The goal of initialization is often to preserve the variance of activations and gradients across layers.

### Fan-in and Fan-out
- **Fan-in**: The number of input units to the layer.
- **Fan-out**: The number of output units from the layer.

For a linear layer with weights of shape `(out_features, in_features)`, `fan_in = in_features` and `fan_out = out_features`.
For a convolutional layer with weights of shape `(out_channels, in_channels, kH, kW)`, `fan_in = in_channels * kH * kW` and `fan_out = out_channels * kH * kW`.

### Xavier (Glorot) Initialization
Designed for layers with **Sigmoid** or **Tanh** activation functions. It draws samples from a distribution with variance:
$$ \text{Var}(W) = \frac{2}{\text{fan\_in} + \text{fan\_out}} $$

### Kaiming (He) Initialization
Designed for layers with **ReLU** or **LeakyReLU** activation functions. It draws samples from a distribution with variance:
$$ \text{Var}(W) = \frac{2}{\text{fan\_in}} $$
(assuming `mode='fan_in'`).

## 3. Implementation Plan

We will implement the following functions in `include/vesper/nn/init.h` and `src/nn/init.cpp`:

### Basic Initializers
- `uniform_(Tensor& tensor, float min, float max)`
- `normal_(Tensor& tensor, float mean, float std)`
- `constant_(Tensor& tensor, float value)`
- `ones_(Tensor& tensor)`
- `zeros_(Tensor& tensor)`

### Advanced Initializers
- `xavier_uniform_(Tensor& tensor, float gain=1.0)`
- `xavier_normal_(Tensor& tensor, float gain=1.0)`
- `kaiming_uniform_(Tensor& tensor, float a=0, mode="fan_in", nonlinearity="leaky_relu")`
- `kaiming_normal_(Tensor& tensor, float a=0, mode="fan_in", nonlinearity="leaky_relu")`

## 4. Usage Examples

```cpp
#include <vesper/nn/linear.h>
#include <vesper/nn/init.h>

// Initialize a Linear layer for ReLU network
auto fc = vesper::nn::Linear(784, 256);

// Weights: Kaiming Uniform
vesper::nn::init::kaiming_uniform_(fc.weight, std::sqrt(5.0)); 

// Bias: Zeros
vesper::nn::init::zeros_(fc.bias);
```

## 5. Testing Strategy

We verify initialization by checking the statistical properties of the initialized tensors.

```cpp
// Test Kaiming Normal
Tensor w = empty({1000, 1000}, DType::Float32, Device::CPU);
nn::init::kaiming_normal_(w);

// Expected std dev = sqrt(2 / fan_in) = sqrt(2/1000) ≈ 0.0447
float mean = w.mean().item<float>();
float std = w.std().item<float>();

assert(std::abs(mean) < 1e-2);
assert(std::abs(std - 0.0447) < 1e-3);
```
