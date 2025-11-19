
# Vesper Build Plan - Chapter 12.3: Autograd for Matrix Multiplication

## 1. Goal

Implement the backward pass for the `ops::matmul` function. This is a critical step that enables any layers using `matmul` (like `nn::Linear`) to be trainable. The gradients for matrix multiplication are themselves matrix multiplications, beautifully demonstrating the compositional power of the autograd system.

## 2. Prerequisites

-   Chapter 7: The `ops::matmul` forward pass.
-   Chapter 10: The autograd engine and `Tensor::accumulate_grad`.
-   Chapter 12.1: The `Tensor::transpose` method.

## 3. The Mathematics of `matmul` Gradients

Given a matrix multiplication `C = A * B`, and the upstream gradient `d(Loss)/dC` (which is `C.grad`), the gradients with respect to the inputs `A` and `B` are calculated using the chain rule:

-   **Gradient with respect to `A`**: `d(Loss)/dA = d(Loss)/dC * B^T`
-   **Gradient with respect to `B`**: `d(Loss)/dB = A^T * d(Loss)/dC`

As you can see, the backward pass for `matmul` re-uses `matmul` itself.

## 4. Detailed Steps

### Step 4.1: Update `ops::matmul` with `backward_fn`

We will now modify the `matmul` function in `src/ops/gemm.cpp` to create and attach the correct `backward_fn` to the output tensor's `grad_node`.

Modify `src/ops/gemm.cpp`:
```cpp
// src/ops/gemm.cpp
#include <vesper/autograd/guard.h> // For no-grad context

Tensor matmul(const Tensor& a, const Tensor& b) {
    // --- Pre-condition checks from Chapter 7 ---
    if (a.device() != b.device()) // ...
    if (a.shape().size() != 2 || b.shape().size() != 2) // ...
    if (a.shape()[1] != b.shape()[0]) // ...

    // Check for autograd enablement
    bool result_requires_grad = (a.requires_grad() || b.requires_grad()) && vesper::autograd::grad_mode_enabled;

    // --- Handle non-contiguous inputs ---
    Tensor a_contig = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contig = b.is_contiguous() ? b : b.contiguous();

    // --- Prepare Output Tensor ---
    const int M = a_contig.shape()[0];
    const int N = b_contig.shape()[1];
    Tensor c = empty({M, N}, a.dtype(), a.device(), result_requires_grad);
    
    // --- Dispatch to Backend (CPU or HIP) ---
    switch (a.device()) {
        // ... cases for CPU, HIP ...
    }

    // --- Create the backward pass function ---
    if (result_requires_grad) {
        c.grad_node = std::make_shared<autograd::Node>();
        if (a.requires_grad()) c.grad_node->next_edges.push_back({a.grad_node});
        if (b.requires_grad()) c.grad_node->next_edges.push_back({b.grad_node});

        c.grad_node->backward_fn = [a_contig, b_contig, c]() mutable {
            // Use mutable lambda to modify captured tensors if needed
            if (a_contig.requires_grad()) {
                // grad_a = grad_c * b^T
                auto grad_a = ops::matmul(c.grad(), b_contig.transpose(0, 1));
                a_contig.accumulate_grad(grad_a);
            }
            if (b_contig.requires_grad()) {
                // grad_b = a^T * grad_c
                auto grad_b = ops::matmul(a_contig.transpose(0, 1), c.grad());
                b_contig.accumulate_grad(grad_b);
            }
        };
    }
    return c;
}
```
*Note: We capture `a_contig` and `b_contig` in the lambda. The `accumulate_grad` call on them will correctly propagate gradients back to the original `a` and `b` if `contiguous()` created a copy.*

## 5. Verification

The test will perform a forward and backward pass through a `matmul` operation and check the computed gradients against their known analytical values.

### Step 5.1: Create `tests/test_autograd_ops.cpp`
```cpp
// tests/test_autograd_ops.cpp
#include <vesper/ops/gemm.h>
#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>

void test_matmul_backward() {
    std::cout << "Testing matmul backward pass..." << std::endl;

    auto device = vesper::Device::CPU;
    
    // 1. Create input tensors
    auto a = vesper::empty({2, 3}, vesper::DType::Float32, device, true);
    auto b = vesper::empty({3, 4}, vesper::DType::Float32, device, true);
    a.copy_from_host(std::vector<float>{1, 2, 3, 4, 5, 6}.data());
    b.copy_from_host(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.data());

    // 2. Forward pass and a dummy loss
    auto c = vesper::ops::matmul(a, b);
    auto loss = vesper::ops::sum(c); // loss = sum(A * B)

    // 3. Backward pass
    loss.backward();

    // 4. Verification
    // d(loss)/d(C_ij) = 1 for all i,j because the loss is sum. So C.grad is all 1s.
    // d(loss)/dA = C.grad * B^T = ones(2,4) * B^T
    // The gradient for A should be the sum of the columns of B for each row of A's gradient.
    // For a_11, gradient is sum of first row of B^T, which is sum of first col of B.
    // So grad(A_ij) = sum(B_kj for all k).
    // Sum of B's columns: (1+5+9)=15, (2+6+10)=18, (3+7+11)=21, (4+8+12)=24
    
    std::vector<float> a_grad_data(a.numel());
    a.grad().copy_to_host(a_grad_data.data());

    // grad(A_i,j) = sum(B_k,j) over k
    // All rows in grad(A) should be identical: {15, 18, 21, ...} no, that's not right.
    // Grad(A) = Grad(C) * B^T. Shape is (2,4) * (4,3) -> (2,3)
    // grad(A)_ij = sum_k grad(C)_ik * B_jk
    // Since grad(C) is all ones, grad(A)_ij = sum_k B_jk. Sum of rows of B.
    // Sum of B rows: {1+2+3+4=10}, {5+6+7+8=26}, {9+10+11+12=42}
    // Each row of grad(A) should be the sum of the columns of B^T.
    // Each element grad(A)_ij is the sum of the j-th row of B.
    assert(fabs(a_grad_data[0] - 10) < 1e-4);
    assert(fabs(a_grad_data[1] - 26) < 1e-4);
    assert(fabs(a_grad_data[2] - 42) < 1e-4);
    assert(fabs(a_grad_data[3] - 10) < 1e-4);
    assert(fabs(a_grad_data[4] - 26) < 1e-4);
    assert(fabs(a_grad_data[5] - 42) < 1e-4);
    
    std::cout << "Matmul backward pass test passed!" << std::endl;
}

int main() {
    test_matmul_backward();
    return 0;
}
```
### Step 5.2: Add Test to `tests/CMakeLists.txt`
```cmake
add_executable(autograd_ops_tests test_autograd_ops.cpp)
target_link_libraries(autograd_ops_tests PRIVATE vesper)
add_test(NAME AutogradOpsTests COMMAND autograd_ops_tests)
```
With a passing test, you have a fully autograd-compatible matrix multiplication operation. This is arguably the most important single component for building modern deep learning models, and it makes your `nn::Linear` layer fully trainable.
