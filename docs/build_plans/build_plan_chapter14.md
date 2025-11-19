
# Vesper Build Plan - Chapter 14: Measuring Performance: Loss Functions (MSE)

## 1. Goal

Implement the Mean Squared Error (MSE) loss function. This chapter is now significantly simpler, as it can directly compose the element-wise operations (`sub`, `mul`) and reduction (`sum`) that were established and verified in previous chapters.

## 2. Prerequisites

-   Chapter 5 & 5.1: Element-wise `sub` and `mul` operations with autograd support.
-   Chapter 8: The `sum` reduction operation with autograd support.

## 3. The MSE Formula

The MSE loss is defined as `loss = mean((y_pred - y_true)^2)`. The Vesper implementation will be a direct translation of this formula using our library's ops.

## 4. Detailed Steps

### Step 4.1: Implement `mse_loss` Functional

Add the function to `include/vesper/nn/functional.h` and implement it in `src/nn/functional.cpp`.

```cpp
// src/nn/functional.cpp
#include <vesper/ops/reduction.h> // for sum

Tensor mse_loss(const Tensor& y_pred, const Tensor& y_true) {
    // 1. diff = y_pred - y_true
    Tensor diff = ops::sub(y_pred, y_true);

    // 2. sq_diff = diff * diff
    Tensor sq_diff = ops::mul(diff, diff);

    // 3. sum_sq = sum(sq_diff)
    Tensor sum_sq = ops::sum(sq_diff);

    // 4. loss = sum_sq / n
    // We reuse scalar multiplication. Division is multiplication by the reciprocal.
    float n_reciprocal = 1.0f / static_cast<float>(y_pred.numel());
    Tensor loss = ops::mul(sum_sq, n_reciprocal);

    return loss;
}
```
Because each op (`sub`, `mul`, `sum`) is already autograd-aware, the `loss` tensor will automatically be the root of a valid computational graph.

### Step 4.2: Create the `MSELoss` Module

Create `include/vesper/nn/loss.h` for the module wrapper.
```cpp
// include/vesper/nn/loss.h
#pragma once
#include <vesper/nn/module.h>
#include <vesper/nn/functional.h>

namespace vesper::nn {

class MSELoss : public Module {
public:
    MSELoss() = default;

    // The forward pass takes a prediction and a target
    Tensor forward(const Tensor& y_pred, const Tensor& y_true) {
        return functional::mse_loss(y_pred, y_true);
    }
    
    // Override the base forward to guide the user
    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("MSELoss requires two inputs: prediction and target.");
    }
};

} // namespace vesper::nn
```

## 5. Verification

The verification test remains the same as in the original plan, but its implementation is now simpler and more robust because it relies on previously tested components.

Create `tests/test_loss.cpp` and add it to `CMakeLists.txt`.
```cpp
// tests/test_loss.cpp
#include <vesper/nn/loss.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

void test_mse_loss() {
    std::cout << "Testing MSELoss..." << std::endl;

    auto y_pred_t = vesper::empty({4}, vesper::DType::Float32, vesper::Device::CPU, true);
    auto y_true_t = vesper::empty({4}, vesper::DType::Float32, vesper::Device::CPU, false);
    // ... setup data ...

    // 1. Forward Pass Verification
    // ... manual calculation ...
    auto loss_module = vesper::nn::MSELoss();
    auto loss_t = loss_module.forward(y_pred_t, y_true_t);
    // ... assert forward pass is correct ...

    // 2. Backward Pass Verification
    loss_t.backward();
    // ... manual gradient calculation ...
    // ... assert y_pred_t.grad() is correct ...

    std::cout << "MSELoss test passed!" << std::endl;
}

int main() { test_mse_loss(); return 0; }
```
A passing test confirms that multiple autograd-aware operations can be composed to create a valid, trainable loss function.
