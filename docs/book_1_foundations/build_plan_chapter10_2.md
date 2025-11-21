
# Vesper Build Plan - Chapter 10: Autograd Engine: The `backward()` Pass

## 1. Goal

Implement the `backward()` method on `Tensor`. This is the core of the autograd engine, responsible for traversing the computational graph, executing the stored gradient functions, and accumulating gradients in the leaf tensors.

## 2. The Backward Pass Algorithm

The process, triggered by `tensor.backward()`, consists of two main phases:

1.  **Topological Sort**: The engine performs a traversal (Depth First Search) of the graph starting from the final node. This builds a list of all nodes in an order that guarantees that when we process a node, the gradients for all nodes that depend on it have already been computed.
2.  **Gradient Calculation**: The engine iterates through the sorted list in **reverse order**. For each node, it executes the `backward_fn` stored within it. This function calculates the local gradients and adds them to the `.grad` attribute of its input tensors.

## 3. Detailed Steps

### Step 3.1: Create the Autograd Engine

This component will encapsulate the graph traversal logic.

Create `include/vesper/autograd/engine.h`:
```cpp
// include/vesper/autograd/engine.h
#pragma once
#include <vesper/autograd/node.h>
#include <memory>

namespace vesper::autograd {

// Main entry point for the backward pass
void execute_backward_pass(const std::shared_ptr<Node>& root_node);

}
```

Create `src/autograd/engine.cpp`:
```cpp
// src/autograd/engine.cpp
#include <vesper/autograd/engine.h>
#include <vector>
#include <unordered_set>
#include <algorithm>

namespace vesper::autograd {

// Helper to build the topologically sorted list of nodes
void build_topo_sort(
    std::vector<std::shared_ptr<Node>>& sorted_nodes,
    std::unordered_set<Node*>& visited,
    const std::shared_ptr<Node>& node) {
    
    if (!node || visited.count(node.get())) {
        return;
    }
    visited.insert(node.get());

    for (const auto& edge : node->next_edges) {
        build_topo_sort(sorted_nodes, visited, edge.node);
    }
    sorted_nodes.push_back(node);
}

void execute_backward_pass(const std::shared_ptr<Node>& root_node) {
    std::vector<std::shared_ptr<Node>> sorted_nodes;
    std::unordered_set<Node*> visited;

    // 1. Build the topologically sorted graph
    build_topo_sort(sorted_nodes, visited, root_node);

    // 2. Iterate in reverse topological order and execute backward functions
    std::reverse(sorted_nodes.begin(), sorted_nodes.end());
    for (const auto& node : sorted_nodes) {
        if (node->backward_fn) {
            node->backward_fn();
        }
    }
}

} // namespace vesper::autograd
```

### Step 3.2: Implement `Tensor::backward()`

Add the `backward()` method to `Tensor` and a helper for accumulating gradients.

In `include/vesper/core/tensor.h`, add the declarations:
```cpp
// public section of Tensor
void backward();
void accumulate_grad(const Tensor& grad); // Helper
```

In `src/core/tensor.cpp`, add the implementations:
```cpp
// src/core/tensor.cpp
#include <vesper/autograd/engine.h> // New include
#include <vesper/ops/elementwise.h> // For accumulate_grad

void Tensor::backward() {
    if (!requires_grad_) {
        throw std::runtime_error("Cannot call backward on a tensor that does not require gradients.");
    }
    if (this->numel() != 1) {
        throw std::runtime_error("Backward can only be called on a scalar tensor.");
    }

    // 1. Initialize this tensor's gradient to 1.0
    this->grad() = ones({1}, this->dtype(), this->device());

    // 2. Start the backward pass from this tensor's node
    if (this->grad_node) {
        autograd::execute_backward_pass(this->grad_node);
    }
}

void Tensor::accumulate_grad(const Tensor& other) {
    if (!this->grad_) {
        this->grad() = other; // First time, just set it
    } else {
        *this->grad_ = ops::add(*this->grad_, other);
    }
}

// Also need to implement `ones` factory
Tensor ones(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad) {
    auto t = empty(shape, dtype, device, requires_grad);
    // In a real implementation, you'd launch a kernel to fill with 1s.
    // For autograd testing, we can do a blocking copy from host.
    if (dtype == DType::Float32) {
        std::vector<float> ones_vec(t.numel(), 1.0f);
        t.copy_from_host(ones_vec.data());
    }
    return t;
}
// Add to factories.h as well
```

### Step 3.3: Implement the `backward_fn` (The Cycle-Breaking Part)

Now we update our `add` operation to create a fully-functional `backward_fn`. This is where we use `weak_ptr` to avoid memory leaks.

Modify `src/ops/elementwise.cpp`:
```cpp
Tensor add(const Tensor& a, const Tensor& b) {
    // ... checks ...
    bool result_requires_grad = a.requires_grad() || b.requires_grad();
    // Use make_shared to get a shared_ptr to the result
    auto result_ptr = std::make_shared<Tensor>(empty(a.shape(), a.dtype(), a.device(), result_requires_grad));
    
    // ... dispatch ...
    // Note: The dispatch must now operate on the pointer's value
    add_hip_dispatch(a, b, *result_ptr);

    if (result_requires_grad) {
        result_ptr->grad_node = std::make_shared<autograd::Node>();
        // ... (add edges as before) ...

        // --- The backward function ---
        // Capture weak_ptrs to inputs and a raw pointer to result's grad
        auto a_weak = std::weak_ptr<Tensor>(std::const_pointer_cast<Tensor>(a.shared_from_this())); // This requires a change to how tensors are created.
        // Let's simplify and pass mutable shared_ptrs into ops for now.

        // A simpler way without shared_from_this for now:
        // Assume `a` and `b` are passed as shared_ptr
        // For now, let's just make the lambda. We will fix the pointers later.
        
        result_ptr->grad_node->backward_fn = [a, b, result_ptr]() {
            // Gradient of add is 1, so we just pass the upstream grad through
            if (a.requires_grad()) {
                a.accumulate_grad(result_ptr->grad());
            }
            if (b.requires_grad()) {
                b.accumulate_grad(result_ptr->grad());
            }
        };
    }
    return *result_ptr;
}

// This approach is simpler but has a high risk of memory cycles if the user isn't careful.
// A full framework would use weak_ptr, but this is a workable first implementation.
```
**Correction**: The above `add` implementation is complex. A much cleaner way is to make `Tensor` inherit from `std::enable_shared_from_this`. Let's do that.

**Revised `tensor.h`**:
```cpp
// include/vesper/core/tensor.h
class Tensor : public std::enable_shared_from_this<Tensor> { ... };
```
This requires that all Tensors are managed by `shared_ptr`. We must change our factories to return `std::shared_ptr<Tensor>`. This is a significant but necessary refactoring for safe autograd. For this chapter, we will stick to the simpler but less safe lambda capture to avoid a major refactor. The key is to demonstrate the backward pass.

### Step 3.4: Update CMake
Add `src/autograd/engine.cpp` to `src/CMakeLists.txt`.
```cmake
target_sources(vesper PRIVATE
    # ...
    ops/hip/reduction.hip
    autograd/engine.cpp  # Add this
)
```

## 4. Verification

This test will be the first end-to-end autograd validation.

### Step 4.1: Update `tests/test_autograd.cpp`

```cpp
// tests/test_autograd.cpp
#include <vesper/ops/reduction.h> // For sum

void test_backward_pass() {
    std::cout << "Testing autograd backward pass..." << std::endl;

    // Use CPU for simplicity and to avoid data transfer boilerplate
    auto a = vesper::empty({2, 2}, vesper::DType::Float32, vesper::Device::CPU, true);
    auto b = vesper::empty({2, 2}, vesper::DType::Float32, vesper::Device::CPU, true);

    std::vector<float> a_data = {1, 2, 3, 4};
    std::vector<float> b_data = {5, 6, 7, 8};
    a.copy_from_host(a_data.data());
    b.copy_from_host(b_data.data());

    // Forward pass
    auto c = vesper::ops::add(a, b);
    auto loss = vesper::ops::sum(c);

    // Backward pass
    loss.backward();

    // Verification
    // The gradient of sum() is a tensor of 1s.
    // The gradient of add() passes the upstream gradient through.
    // So, a.grad and b.grad should both be tensors of 1s.
    std::vector<float> grad_data(4);
    a.grad().copy_to_host(grad_data.data());

    for (int i = 0; i < 4; ++i) {
        assert(std::fabs(grad_data[i] - 1.0f) < 1e-6);
    }

    b.grad().copy_to_host(grad_data.data());
    for (int i = 0; i < 4; ++i) {
        assert(std::fabs(grad_data[i] - 1.0f) < 1e-6);
    }

    std::cout << "Backward pass test passed!" << std::endl;
}

int main() {
    test_graph_construction();
    test_backward_pass(); // Add this call
    return 0;
}
```

This requires updating `sum` to support autograd as well. The `backward_fn` for `sum` will create a tensor of ones shaped like the input (`broadcast_to`) and multiply it by the scalar output gradient. This is a broadcast operation we haven't written yet.

**Simplification for this test:** We will assume the gradient of `sum` is a tensor of ones. This is correct if the output gradient is `1.0`.

Update `src/ops/reduction.cpp` `sum()`:
```cpp
// Add to sum() function before returning result
if (result_requires_grad) {
    result_ptr->grad_node = std::make_shared<autograd::Node>();
    // ... add edge ...
    result_ptr->grad_node->backward_fn = [input_ptr, result_ptr]() {
        // d(sum)/d(input) is 1. We need to broadcast the scalar grad back.
        auto grad_of_ones = ones(input_ptr->shape(), ...);
        auto broadcasted_grad = ops::mul(grad_of_ones, result_ptr->grad()); // Needs mul op
        input_ptr->accumulate_grad(broadcasted_grad);
    };
}
```
This reveals we need more ops (`mul`) and concepts (`broadcast`) for a fully correct autograd. For Chapter 10, the test will focus *only* on the `add` gradient, and we will call `backward` on `c` directly after applying a sum on the CPU to get a scalar loss.

**Final Test Simplification:**
Instead of `loss = ops::sum(c)`, we will compute `c.backward()` but assume an incoming gradient of `1.0`.
In `Tensor::backward`, we will allow it to be called on non-scalars for now, and it will assume an incoming gradient of `1.0`s.

This is a pragmatic compromise to test the `add` backward function and the engine itself without implementing more primitives. The next chapters on NN modules will force the implementation of these other gradient functions.
