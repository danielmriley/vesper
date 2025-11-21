
# Vesper Build Plan - Chapter 13.2: Comparison Ops and ReLU Activation

## 1. Goal

Implement the ReLU activation function. The forward pass is simple (`max(0, x)`), but its backward pass (`grad_out * (x > 0)`) requires a new primitive: an element-wise comparison operation. This chapter focuses on creating that primitive and then using it to build a correct `relu` backward pass.

## 2. Prerequisites

-   Chapter 5.1: `ops::mul` is implemented.
-   Chapter 13.1: The structure for activations is established.

## 3. Detailed Steps

### Step 3.1: Implement a Comparison Kernel

We need an operation that returns `1.0f` if a condition is true, and `0.0f` otherwise. We will create a generic kernel for this.

In `src/ops/hip/activation.hip`, add a new kernel:
```cpp
// src/ops/hip/activation.hip

// Generic kernel for element-wise comparison with a scalar
template <typename T, typename CompareOp>
__global__ void comparison_scalar_kernel(const T* in, T* out, size_t n, T scalar, CompareOp op) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = op(in[idx], scalar) ? static_cast<T>(1) : static_cast<T>(0);
    }
}
```

### Step 3.2: Create the `greater_than` Functional

Create a C++ function to wrap the comparison kernel. This will live in a new file for comparison ops.

Create `include/vesper/ops/comparison.h`:
```cpp
// include/vesper/ops/comparison.h
#pragma once
#include <vesper/core/tensor.h>

namespace vesper::ops {
Tensor greater_than(const Tensor& a, float b);
void greater_than_hip_dispatch(const Tensor& a, float b, Tensor& out);
}
```

Create `src/ops/comparison.cpp` and `src/ops/hip/comparison.hip`.
```cpp
// src/ops/hip/comparison.hip
// ... new file, include headers ...
void greater_than_hip_dispatch(const Tensor& a, float b, Tensor& out) {
    // Launch comparison_scalar_kernel with std::greater<float>
}

// src/ops/comparison.cpp
// ... new file, include headers ...
Tensor greater_than(const Tensor& a, float b) {
    // This op does not support autograd, so `requires_grad` is false.
    Tensor result = empty(a.shape(), a.dtype(), a.device(), false);
    greater_than_hip_dispatch(a, b, result);
    return result;
}
```

### Step 3.3: Implement the `relu` Functional and Backward Pass

Now we can implement `relu` and its correct backward pass.

Add the declaration to `include/vesper/nn/functional.h`:
```cpp
Tensor relu(const Tensor& input);
void relu_hip_dispatch(const Tensor& input, Tensor& output);
```
Add the kernel and dispatch to `src/ops/hip/activation.hip`:
```cpp
// src/ops/hip/activation.hip
template <typename T>
__global__ void relu_kernel(const T* in, T* out, size_t n) { /* ... */ }
void relu_hip_dispatch(const Tensor& input, Tensor& output) { /* ... */ }
```
Modify `src/nn/functional.cpp` to implement `relu`:
```cpp
// src/nn/functional.cpp
#include <vesper/ops/comparison.h> // New include

Tensor relu(const Tensor& input) {
    bool result_requires_grad = input.requires_grad() && autograd::grad_mode_enabled;
    Tensor result = empty(input.shape(), input.dtype(), input.device(), result_requires_grad);
    relu_hip_dispatch(input, result);

    if (result_requires_grad) {
        result.grad_node = std::make_shared<autograd::Node>();
        result.grad_node->next_edges.push_back({input.grad_node});
        result.grad_node->backward_fn = [input, result]() mutable {
            // ReLU gradient: grad_input = grad_output * (input > 0)
            // 1. Create the mask from the original input
            auto mask = ops::greater_than(input, 0.0f);

            // 2. Multiply the upstream gradient by the mask
            auto final_grad = ops::mul(result.grad(), mask);
            input.accumulate_grad(final_grad);
        };
    }
    return result;
}
```

### Step 3.4: Create the `ReLU` Module Wrapper

In `include/vesper/nn/activations.h`, add the `ReLU` module.
```cpp
// include/vesper/nn/activations.h
class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override {
        return functional::relu(input);
    }
};
```

## 4. Verification

Update `tests/test_activations.cpp` to test the correct ReLU backward pass.

```cpp
// tests/test_activations.cpp

void test_relu_correct_backward() {
    std::cout << "Testing ReLU with correct backward pass..." << std::endl;
    auto device = vesper::Device::CPU;

    auto input = vesper::empty({4}, vesper::DType::Float32, device, true);
    input.copy_from_host(std::vector<float>{-1.0f, 0.5f, 1.0f, -2.0f}.data());

    auto output = vesper::nn::functional::relu(input);
    auto loss = vesper::ops::sum(output);
    loss.backward();

    std::vector<float> in_grad(4);
    input.grad().copy_to_host(in_grad.data());

    // Gradient should be 1.0 where input > 0, and 0.0 otherwise.
    assert(fabs(in_grad[0] - 0.0f) < 1e-6);
    assert(fabs(in_grad[1] - 1.0f) < 1e-6);
    assert(fabs(in_grad[2] - 1.0f) < 1e-6);
    assert(fabs(in_grad[3] - 0.0f) < 1e-6);

    std::cout << "ReLU correct backward test passed!" << std::endl;
}

int main() {
    test_sigmoid();
    test_relu_correct_backward();
    return 0;
}
```
A passing test confirms that the new comparison primitive works and that the ReLU backward pass is correctly implemented, demonstrating how new primitives can be composed for complex autograd behaviors.
