
# Vesper Build Plan - Chapter 9: Autograd: Computational Graph & Node Structure

## 1. Goal

Lay the structural foundation for Vesper's automatic differentiation (autograd) engine. This involves creating the `Node` class to represent operations in the computational graph and augmenting the `Tensor` class to track its history (`grad_node`) and whether it requires a gradient (`requires_grad`). This chapter focuses purely on graph *construction*, not execution.

## 2. Autograd Concepts

Automatic differentiation works by building a **Directed Acyclic Graph (DAG)** during the forward pass:
-   **Tensors** are the data flowing through the graph.
-   **Nodes** represent operations (like `add` or `matmul`) that create new tensors.
-   **Edges** connect operation nodes to the `grad_node` of their output tensors, representing the function that created them.

When `.backward()` is called on a final tensor, the autograd engine traverses this graph backward from that tensor, using the **chain rule** to compute gradients at each step.

## 3. Detailed Steps

### Step 3.1: Define the `Node` and `Edge` Structures

The `Node` is the core of the graph, storing the backward function and pointers to the previous operations.

Create `include/vesper/autograd/` directory and `node.h` inside it.
```sh
mkdir -p include/vesper/autograd
```
Create `include/vesper/autograd/node.h`:
```cpp
// include/vesper/autograd/node.h
#pragma once

#include <vector>
#include <functional>
#include <memory>

namespace vesper::autograd {

class Node; // Forward declaration

// An edge represents a dependency in the graph.
struct Edge {
    std::shared_ptr<Node> node;

    // The function that computes the gradient for this input.
    // It takes the upstream gradient and returns the downstream gradient.
    // For this chapter, we'll keep it simple.
};

// A Node represents an operation in the computational graph.
class Node {
public:
    // The function to execute when backpropagating.
    std::function<void()> backward_fn;
    
    // Pointers to the nodes that are inputs to this operation.
    std::vector<Edge> next_edges;
};

} // namespace vesper::autograd
```

### Step 3.2: Augment the `Tensor` Class for Autograd

The `Tensor` needs to become a part of the graph. We'll add a `requires_grad` flag, a pointer to its creator `Node`, and a placeholder for its gradient.

Modify `include/vesper/core/tensor.h`:
```cpp
// include/vesper/core/tensor.h
#pragma once // at top of file

// Add includes
#include <vesper/autograd/node.h> // New include
#include <atomic> // For requires_grad_

// ... inside the Tensor class definition ...
public:
    // --- Autograd Accessors ---
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

    // Access the gradient tensor
    Tensor& grad();

    // The node that created this tensor in the graph
    std::shared_ptr<autograd::Node> grad_node;

private:
    // --- Autograd Members ---
    bool requires_grad_ = false;
    std::unique_ptr<Tensor> grad_; // Lazily initialized gradient

    // Make factory functions friends
    friend Tensor empty(const std::vector<int64_t>&, DType, Device, bool);
```

Modify the `private` constructor in `tensor.h` and implement the `grad()` method in `src/core/tensor.cpp`:

```cpp
// In private section of Tensor in tensor.h
Tensor(std::shared_ptr<Storage> storage,
       DType dtype,
       std::vector<int64_t> shape,
       std::vector<int64_t> strides,
       size_t offset = 0,
       bool requires_grad = false);

// In src/core/tensor.cpp
#include <vesper/core/factories.h> // for empty()

// Add implementation for the new constructor
Tensor::Tensor(std::shared_ptr<Storage> storage, DType dtype,
               std::vector<int64_t> shape, std::vector<int64_t> strides,
               size_t offset, bool requires_grad)
    : storage_(std::move(storage)),
      dtype_(dtype),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset),
      requires_grad_(requires_grad) {} // Set the new member

// Add grad() implementation
Tensor& Tensor::grad() {
    if (!grad_) {
        // Lazily initialize gradient tensor as a tensor of zeros
        // with the same properties as this tensor.
        grad_ = std::make_unique<Tensor>(
            zeros(this->shape(), this->dtype(), this->device())
        );
    }
    return *grad_;
}
```

### Step 3.3: Update Factories and Operations to Build the Graph

The factory functions need to accept `requires_grad`. More importantly, operations like `add` must now create a `Node` if their inputs require gradients.

Update `include/vesper/core/factories.h`:
```cpp
// Modify the signature
Tensor empty(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad = false);
Tensor zeros(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad = false); // Add declaration
```

Update `src/core/tensor.cpp` (factory implementation):
```cpp
// Modify empty() signature and pass requires_grad to constructor
Tensor empty(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad) {
    // ... (logic is the same)
    return Tensor(std::move(storage), dtype, shape, strides, 0, requires_grad);
}

// Implement zeros factory
Tensor zeros(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad) {
    auto tensor = empty(shape, dtype, device, requires_grad);
    // In a future chapter, we'll add a kernel to fill with zeros.
    // For now, an empty tensor is sufficient for structure.
    return tensor;
}
```

Finally, update `src/ops/elementwise.cpp` to build the graph:
```cpp
// src/ops/elementwise.cpp
#include <vesper/autograd/node.h> // Add include

Tensor add(const Tensor& a, const Tensor& b) {
    // ... (pre-condition checks are the same)
    
    bool result_requires_grad = a.requires_grad() || b.requires_grad();
    Tensor result = empty(a.shape(), a.dtype(), a.device(), result_requires_grad);

    // ... (dispatch to backend is the same)

    // --- Autograd Graph Construction ---
    if (result_requires_grad) {
        result.grad_node = std::make_shared<autograd::Node>();
        if (a.requires_grad()) {
            // This is a simplification; we will add the backward_fn in the next chapter
            result.grad_node->next_edges.push_back({a.grad_node});
        }
        if (b.requires_grad()) {
            result.grad_node->next_edges.push_back({b.grad_node});
        }
    }

    return result;
}
```

## 4. Potential Pitfalls

-   **`unique_ptr` for Gradient**: `grad_` is a `unique_ptr<Tensor>`. This is tricky because `Tensor` is an incomplete type inside `tensor.h`. However, since `unique_ptr` only needs to know how to destroy the type, and the destructor will be defined in `tensor.cpp` where `Tensor` is a complete type, this works. It's a clean way to handle lazy initialization.
-   **Forgetting `requires_grad` Propagation**: An operation's output should require a gradient if *any* of its inputs require a gradient. Forgetting this breaks the chain of the graph.
-   **Graph Cycles**: As mentioned in the plan, capturing `shared_ptr`s to tensors inside a `backward_fn` will create a reference cycle. We have deferred implementing the `backward_fn` lambda to the next chapter to isolate this problem.

## 5. Integration and Verification

The test for this chapter doesn't compute any gradients. It simply verifies that the computational graph is being constructed correctly in the forward pass.

### Step 5.1: Create `tests/test_autograd.cpp`
```cpp
// tests/test_autograd.cpp
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <iostream>
#include <cassert>

void test_graph_construction() {
    std::cout << "Testing autograd graph construction..." << std::endl;

    // 1. Create two leaf tensors that require gradients
    auto a = vesper::zeros({2, 2}, vesper::DType::Float32, vesper::Device::CPU, true);
    auto b = vesper::zeros({2, 2}, vesper::DType::Float32, vesper::Device::CPU, true);

    // 2. Create a third tensor that does not
    auto d = vesper::zeros({2, 2}, vesper::DType::Float32, vesper::Device::CPU, false);

    // Verify leaf properties
    assert(a.requires_grad() && a.grad_node == nullptr);
    assert(b.requires_grad() && b.grad_node == nullptr);
    assert(!d.requires_grad());

    // 3. Perform an operation
    auto c = vesper::ops::add(a, b);

    // Check that the result requires grad and has a node
    assert(c.requires_grad());
    assert(c.grad_node != nullptr);
    
    // The new node should have two dependencies (for a and b)
    assert(c.grad_node->next_edges.size() == 2);
    // The edges should point to the parents' grad_nodes (which are null for leaves)
    assert(c.grad_node->next_edges[0].node == a.grad_node);
    assert(c.grad_node->next_edges[1].node == b.grad_node);

    // 4. Operation with one grad-requiring tensor
    auto e = vesper::ops::add(c, d);
    assert(e.requires_grad());
    assert(e.grad_node != nullptr);
    // Should only have one dependency, since `d` does not require grad
    assert(e.grad_node->next_edges.size() == 1);
    assert(e.grad_node->next_edges[0].node == c.grad_node);

    std::cout << "Graph construction test passed!" << std::endl;
}

int main() {
    test_graph_construction();
    return 0;
}
```
*Note: This test uses the CPU device for simplicity, as no actual computation is being tested.*

### Step 5.2: Update CMake
Add new files and the test to the `CMakeLists.txt` files.
```cmake
// src/CMakeLists.txt - add new cpp files if any (none in this case)

// tests/CMakeLists.txt
add_executable(autograd_tests test_autograd.cpp)
target_link_libraries(autograd_tests PRIVATE vesper)
add_test(NAME AutogradGraphTests COMMAND autograd_tests)
```
A passing test confirms that your tensors are now aware of their history, setting the stage for the backward pass.
