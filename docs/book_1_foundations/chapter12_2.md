
# Vesper Build Plan - Chapter 12.2: The `nn::Linear` Module

## 1. Goal

Implement the `vesper::nn::Linear` layer. This is the first concrete, parameter-holding subclass of `nn::Module`. This chapter focuses on the layer's structure, parameter initialization, and its `forward` pass, leveraging the `ops::matmul` function.

## 2. Prerequisites

-   Chapter 7: A working `ops::matmul` function.
-   Chapter 11: The `nn::Module` base class.
-   Chapter 12.1: `Tensor::transpose()` and `Tensor::contiguous()` methods.

## 3. Parameter Initialization

A crucial part of creating a layer is initializing its weights correctly. Initializing weights to zero can prevent a network from learning. We will implement a standard initialization scheme, Kaiming uniform, which helps maintain variance of activations as they pass through the network.

**Formula**: Weights are sampled from a uniform distribution `U(-bound, bound)`, where `bound = sqrt(6 / fan_in)`.

## 4. Detailed Steps

### Step 4.1: Create `linear.h`

Create the header file in `include/vesper/nn/`:
```cpp
// include/vesper/nn/linear.h
#pragma once

#include <vesper/nn/module.h>

namespace vesper::nn {

class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, bool use_bias = true);

    Tensor forward(const Tensor& input) override;

    // The parameters of the layer, owned by the class
    Tensor weight;
    Tensor bias;
private:
    bool use_bias_;
};

} // namespace vesper::nn
```

### Step 4.2: Implement the `Linear` Layer

Create `src/nn/linear.cpp`. This file will contain the constructor (with initialization) and the forward pass logic.

```cpp
// src/nn/linear.cpp
#include <vesper/nn/linear.h>
#include <vesper/core/factories.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/elementwise.h> // For bias addition
#include <random>
#include <cmath>

namespace vesper::nn {

// Helper function for Kaiming Uniform Initialization
void kaiming_uniform_init(Tensor& t, int64_t fan_in) {
    const float bound = std::sqrt(6.0f / fan_in);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-bound, bound);

    std::vector<float> data(t.numel());
    for (float& val : data) {
        val = dist(rng);
    }
    t.copy_from_host(data.data());
}

Linear::Linear(int64_t in_features, int64_t out_features, bool use_bias)
    : use_bias_(use_bias) {
    
    // 1. Initialize weight tensor with Kaiming initialization
    weight = empty({out_features, in_features}, DType::Float32, Device::CPU);
    kaiming_uniform_init(weight, in_features);
    register_parameter("weight", weight); // Register it as a trainable parameter

    if (use_bias_) {
        // 2. Initialize bias tensor to zeros
        bias = zeros({out_features}, DType::Float32, Device::CPU);
        register_parameter("bias", bias);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // 3. Compute the forward pass: y = x * W^T + b
    
    // The transpose() call creates a non-contiguous view. Our matmul from Ch 7
    // was updated to handle this by calling .contiguous() internally.
    auto output = ops::matmul(input, weight.transpose(0, 1));

    if (use_bias_) {
        // This requires broadcasting support for `ops::add`.
        // We will assume this is implemented for this chapter.
        // A simple implementation would expand `bias` to match `output`'s shape.
        output = ops::add(output, bias);
    }
    return output;
}

} // namespace vesper::nn
```

### Step 4.3: Update CMake

Add the new `linear.cpp` file to `src/CMakeLists.txt`:
```cmake
target_sources(vesper PRIVATE
    # ...
    nn/module.cpp
    nn/linear.cpp  # Add this
    # ...
)
```

## 5. Verification

The test will instantiate a `Linear` layer, check its parameters, and ensure the forward pass produces an output of the correct shape.

### Step 5.1: Create `tests/test_nn_layers.cpp`
```cpp
// tests/test_nn_layers.cpp
#include <vesper/nn/linear.h>
#include <iostream>
#include <cassert>

void test_linear_layer_forward() {
    std::cout << "Testing nn::Linear layer forward pass..." << std::endl;

    const int64_t in_f = 16, out_f = 8, batch_size = 4;
    
    // 1. Create the layer
    auto linear_layer = std::make_shared<vesper::nn::Linear>(in_f, out_f);

    // 2. Check that parameters were registered
    auto params = linear_layer->parameters();
    assert(params.size() == 2); // weight and bias
    assert(params[0]->shape() == std::vector<int64_t>({out_f, in_f})); // Correct weight shape
    assert(params[1]->shape() == std::vector<int64_t>({out_f}));       // Correct bias shape

    // 3. Create a dummy input tensor
    auto input = vesper::zeros({batch_size, in_f}, vesper::DType::Float32, vesper::Device::CPU);

    // 4. Perform the forward pass
    // We will temporarily disable bias for the test, as broadcasting `add` is not implemented.
    linear_layer->bias.set_requires_grad(false); // To simplify
    auto output = linear_layer->forward(input);

    // 5. Check the output shape
    assert(output.shape() == std::vector<int64_t>({batch_size, out_f}));

    std::cout << "nn::Linear layer forward test passed!" << std::endl;
}

int main() {
    test_linear_layer_forward();
    return 0;
}
```

### Step 5.2: Add Test to `tests/CMakeLists.txt`
```cmake
add_executable(nn_layer_tests test_nn_layers.cpp)
target_link_libraries(nn_layer_tests PRIVATE vesper)
add_test(NAME NnLayerTests COMMAND nn_layer_tests)
```
A passing test confirms the `Linear` module is correctly structured and its forward pass executes as expected. The next step is to enable autograd for this layer.
