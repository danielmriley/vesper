
# Vesper Build Plan - Chapter 5.1: Expanding Element-wise Operations

## 1. Goal

Expand our element-wise operation capabilities beyond `add` to include `sub` (subtraction) and `mul` (multiplication), including their scalar variants and backward passes. These are fundamental building blocks required for nearly all advanced operations and loss functions.

## 2. Prerequisites

-   Chapter 5: A generic element-wise HIP kernel and the `ops::add` function.
-   Chapter 10: The autograd engine.

## 3. Detailed Steps

### Step 3.1: Implement `sub` and `mul` C++ Functions

We will reuse the `elementwise_binary_kernel` from Chapter 5 by passing it different C++ standard library functors (`std::minus`, `std::multiplies`).

First, add the declarations to `include/vesper/ops/elementwise.h`:
```cpp
// include/vesper/ops/elementwise.h
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, float b); // Scalar variant

// Backend dispatch declarations
void sub_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out);
void mul_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out);
```

Next, add the HIP dispatchers in `src/ops/hip/elementwise.hip`:
```cpp
// src/ops/hip/elementwise.hip
// ... after add_hip_dispatch

void sub_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    const int threads = 256;
    const int blocks = (a.numel() + threads - 1) / threads;
    hipLaunchKernelGGL(elementwise_binary_kernel<float, std::minus<float>>,
        dim3(blocks), dim3(threads), 0, 0,
        a.data_ptr<const float>(), b.data_ptr<const float>(), out.data_ptr<float>(),
        a.numel(), std::minus<float>());
}

void mul_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    const int threads = 256;
    const int blocks = (a.numel() + threads - 1) / threads;
    hipLaunchKernelGGL(elementwise_binary_kernel<float, std::multiplies<float>>,
        dim3(blocks), dim3(threads), 0, 0,
        a.data_ptr<const float>(), b.data_ptr<const float>(), out.data_ptr<float>(),
        a.numel(), std::multiplies<float>());
}
```

Finally, implement the C++ functions and their backward passes in `src/ops/elementwise.cpp`:
```cpp
// src/ops/elementwise.cpp

Tensor sub(const Tensor& a, const Tensor& b) {
    // ... pre-condition checks ...
    bool result_requires_grad = (a.requires_grad() || b.requires_grad()) && vesper::autograd::grad_mode_enabled;
    Tensor result = empty(a.shape(), a.dtype(), a.device(), result_requires_grad);
    sub_hip_dispatch(a, b, result); // Or other backend

    if (result_requires_grad) {
        result.grad_node = std::make_shared<autograd::Node>();
        // ... add edges ...
        result.grad_node->backward_fn = [a, b, result]() mutable {
            if (a.requires_grad()) a.accumulate_grad(result.grad());
            // For subtraction, the gradient to `b` is negated.
            if (b.requires_grad()) b.accumulate_grad(ops::mul(result.grad(), -1.0f));
        };
    }
    return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    // ... pre-condition checks ...
    bool result_requires_grad = (a.requires_grad() || b.requires_grad()) && vesper::autograd::grad_mode_enabled;
    Tensor result = empty(a.shape(), a.dtype(), a.device(), result_requires_grad);
    mul_hip_dispatch(a, b, result); // Or other backend

    if (result_requires_grad) {
        result.grad_node = std::make_shared<autograd::Node>();
        // ... add edges ...
        result.grad_node->backward_fn = [a, b, result]() mutable {
            if (a.requires_grad()) a.accumulate_grad(ops::mul(result.grad(), b));
            if (b.requires_grad()) b.accumulate_grad(ops::mul(result.grad(), a));
        };
    }
    return result;
}

// Scalar variant implementation
Tensor mul(const Tensor& a, float b) {
    // Create a tensor from the scalar to reuse the tensor-tensor mul op
    auto b_tensor = full(a.shape(), a.dtype(), a.device(), b);
    return mul(a, b_tensor);
}

// And the `full` factory in src/core/tensor.cpp (if not already present)
Tensor full(const std::vector<int64_t>& shape, DType dtype, Device device, float val, bool req_grad) {
    auto t = empty(shape, dtype, device, req_grad);
    std::vector<float> data(t.numel(), val);
    t.copy_from_host(data.data());
    return t;
}
```
*Note: The `full` factory and scalar `mul` are essential for optimizers and gradient calculations.*

## 4. Verification

A test should be created to verify the forward and backward passes for `mul`, as its backward pass is more complex than `add` or `sub`.

### Step 4.1: Update `tests/test_autograd_ops.cpp`
```cpp
// tests/test_autograd_ops.cpp

void test_mul_backward() {
    std::cout << "Testing mul backward pass..." << std::endl;
    auto device = vesper::Device::CPU;
    
    auto a = vesper::empty({2}, vesper::DType::Float32, device, true);
    auto b = vesper::empty({2}, vesper::DType::Float32, device, true);
    a.copy_from_host(std::vector<float>{6.0f, 7.0f}.data());
    b.copy_from_host(std::vector<float>{2.0f, 3.0f}.data());

    auto c = vesper::ops::mul(a, b); // c = {12, 21}
    auto loss = vesper::ops::sum(c);   // loss = 33

    loss.backward();

    // Verification
    // d(loss)/dc = {1, 1}
    // d(loss)/da = d(loss)/dc * b = {1, 1} * {2, 3} = {2, 3}
    // d(loss)/db = d(loss)/dc * a = {1, 1} * {6, 7} = {6, 7}
    std::vector<float> a_grad(2), b_grad(2);
    a.grad().copy_to_host(a_grad.data());
    b.grad().copy_to_host(b_grad.data());

    assert(fabs(a_grad[0] - 2.0f) < 1e-6);
    assert(fabs(a_grad[1] - 3.0f) < 1e-6);
    assert(fabs(b_grad[0] - 6.0f) < 1e-6);
    assert(fabs(b_grad[1] - 7.0f) < 1e-6);

    std::cout << "Mul backward pass test passed!" << std::endl;
}

int main() {
    test_matmul_backward();
    test_mul_backward();
    return 0;
}
```
This test confirms that `mul` and its backward pass are correctly implemented, paving the way for more complex functions like `MSELoss`.
