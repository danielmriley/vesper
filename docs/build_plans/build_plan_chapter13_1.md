
# Vesper Build Plan - Chapter 13.1: The Sigmoid Activation

## 1. Goal

Implement the Sigmoid activation function, including its forward pass kernel and its autograd-aware backward pass. The Sigmoid function is defined as `y = 1 / (1 + exp(-x))`. A key feature is that its derivative can be expressed purely in terms of its output: `dy/dx = y * (1 - y)`.

## 2. Prerequisites

-   Chapter 5.1: `ops::sub` and `ops::mul` are implemented.
-   Chapter 10: The autograd engine is functional.

## 3. Detailed Steps

### Step 3.1: Create `functional.h` and Activation Kernels

Create `include/vesper/nn/functional.h` for the declarations.
```cpp
// include/vesper/nn/functional.h
#pragma once
#include <vesper/core/tensor.h>

namespace vesper::nn::functional {

Tensor sigmoid(const Tensor& input);
void sigmoid_hip_dispatch(const Tensor& input, Tensor& output);
// ... other activations will be added here ...

}
```

Create `src/ops/hip/activation.hip` for the kernels.
```cpp
// src/ops/hip/activation.hip
#include <vesper/nn/functional.h>
#include <hip/hip_runtime.h>
#include <cmath>

namespace vesper::nn::functional {

template <typename T>
__global__ void sigmoid_kernel(const T* in, T* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<T>(1) / (static_cast<T>(1) + exp(-in[idx]));
    }
}

void sigmoid_hip_dispatch(const Tensor& input, Tensor& output) {
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    hipLaunchKernelGGL(sigmoid_kernel<float>, dim3(blocks), dim3(threads), 0, 0,
        input.data_ptr<const float>(), output.data_ptr<float>(), input.numel());
}

} // namespace vesper::nn::functional
```

### Step 3.2: Implement the `sigmoid` Function and its Backward Pass

Create `src/nn/functional.cpp`. The `backward_fn` for sigmoid is efficient because it only needs to capture the *output* of the function, not the input.

```cpp
// src/nn/functional.cpp
#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <vesper/autograd/guard.h>

namespace vesper::nn::functional {

Tensor sigmoid(const Tensor& input) {
    bool result_requires_grad = input.requires_grad() && autograd::grad_mode_enabled;
    Tensor result = empty(input.shape(), input.dtype(), input.device(), result_requires_grad);
    
    sigmoid_hip_dispatch(input, result);

    if (result_requires_grad) {
        result.grad_node = std::make_shared<autograd::Node>();
        result.grad_node->next_edges.push_back({input.grad_node});

        result.grad_node->backward_fn = [input, result]() mutable {
            // Sigmoid gradient: grad_output * (output * (1 - output))
            auto ones = vesper::ones(result.shape(), result.dtype(), result.device());
            auto term2 = ops::sub(ones, result);
            auto local_grad = ops::mul(result, term2);
            auto final_grad = ops::mul(result.grad(), local_grad);
            input.accumulate_grad(final_grad);
        };
    }
    return result;
}

} // namespace vesper::nn::functional
```

### Step 3.3: Create the `Sigmoid` Module Wrapper

Create `include/vesper/nn/activations.h`.
```cpp
// include/vesper/nn/activations.h
#pragma once
#include <vesper/nn/module.h>
#include <vesper/nn/functional.h>

namespace vesper::nn {

class Sigmoid : public Module {
public:
    Tensor forward(const Tensor& input) override {
        return functional::sigmoid(input);
    }
};

} // namespace vesper::nn
```

### Step 3.4: Update CMake
Add `src/nn/functional.cpp` and `src/ops/hip/activation.hip` to `src/CMakeLists.txt`.

## 4. Verification

The test will check the numerical correctness of both the forward and backward passes.

### Step 4.1: Create `tests/test_activations.cpp`
```cpp
// tests/test_activations.cpp
#include <vesper/nn/functional.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <cassert>
#include <cmath>

void test_sigmoid() {
    std::cout << "Testing sigmoid activation..." << std::endl;
    auto device = vesper::Device::CPU;

    auto input = vesper::empty({2}, vesper::DType::Float32, device, true);
    input.copy_from_host(std::vector<float>{0.0f, 2.0f}.data());

    // 1. Forward Pass Verification
    auto output = vesper::nn::functional::sigmoid(input);
    std::vector<float> out_data(2);
    output.copy_to_host(out_data.data());

    // y = 1 / (1 + exp(-x))
    assert(fabs(out_data[0] - 0.5f) < 1e-6); // sigmoid(0) = 0.5
    assert(fabs(out_data[1] - (1.0f / (1.0f + exp(-2.0f)))) < 1e-6);

    // 2. Backward Pass Verification
    auto loss = vesper::ops::sum(output);
    loss.backward();

    std::vector<float> in_grad(2);
    input.grad().copy_to_host(in_grad.data());

    // dy/dx = y * (1-y)
    // For x=0, y=0.5, grad=0.5*(1-0.5)=0.25
    // For x=2, y=0.88079, grad=0.88079*(1-0.88079)=0.10499
    assert(fabs(in_grad[0] - 0.25f) < 1e-6);
    assert(fabs(in_grad[1] - 0.10499f) < 1e-4);

    std::cout << "Sigmoid activation test passed!" << std::endl;
}

int main() {
    test_sigmoid();
    return 0;
}
```
### Step 4.2: Add to `tests/CMakeLists.txt`
```cmake
add_executable(activation_tests test_activations.cpp)
target_link_libraries(activation_tests PRIVATE vesper)
add_test(NAME ActivationTests COMMAND activation_tests)
```
A passing test confirms the Sigmoid activation and its gradient are correctly implemented.
