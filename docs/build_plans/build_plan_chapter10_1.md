
# Vesper Build Plan - Chapter 10.1: The "No-Grad" Context

## 1. Goal

Implement a "no-grad" context manager. This is a critical feature for an autograd library, allowing sections of code to be executed without building a computational graph. This is essential for the optimizer's update step and for speeding up inference or validation loops where gradients are not needed.

## 2. The "No-Grad" Concept

When an optimizer updates a parameter (e.g., `weight = weight - lr * grad`), this update itself is a series of mathematical operations. We do *not* want these operations to be tracked by autograd. If they were, the graph would grow infinitely with each training step, leading to a memory leak.

To solve this, we will introduce a global, thread-local flag that disables gradient tracking. An RAII-style guard class will make it easy and safe to toggle this flag within a specific scope.

## 3. Detailed Steps

### Step 3.1: Create the Guard and Global Flag

Create `include/vesper/autograd/guard.h`:
```cpp
// include/vesper/autograd/guard.h
#pragma once

namespace vesper::autograd {

// This thread_local flag ensures that grad mode is tracked separately for each thread.
extern thread_local bool grad_mode_enabled;

// An RAII-style guard to temporarily disable gradient tracking.
class NoGradGuard {
public:
    NoGradGuard() : prev_mode_(grad_mode_enabled) {
        grad_mode_enabled = false;
    }
    ~NoGradGuard() {
        grad_mode_enabled = prev_mode_;
    }

    // Disable copy and move to prevent misuse
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
    NoGradGuard(NoGradGuard&&) = delete;
    NoGradGuard& operator=(NoGradGuard&&) = delete;

private:
    bool prev_mode_;
};

} // namespace vesper::autograd
```
Create `src/autograd/guard.cpp` to provide the definition for the global variable:
```cpp
// src/autograd/guard.cpp
#include <vesper/autograd/guard.h>

namespace vesper::autograd {
// Initialize the thread-local flag to be true by default.
thread_local bool grad_mode_enabled = true;
}
```

### Step 3.2: Update Operations to Respect the Guard

Now, every operation that creates a graph node must be modified to check the `grad_mode_enabled` flag.

Modify `src/ops/elementwise.cpp` (and all other op files like `gemm.cpp`, `reduction.cpp`, etc.):
```cpp
// src/ops/elementwise.cpp
#include <vesper/autograd/guard.h> // New include

Tensor add(const Tensor& a, const Tensor& b) {
    // ... pre-condition checks ...
    
    // The key change: also check if grad mode is enabled globally
    bool result_requires_grad = (a.requires_grad() || b.requires_grad()) 
                              && vesper::autograd::grad_mode_enabled;

    Tensor result = empty(a.shape(), a.dtype(), a.device(), result_requires_grad);
    
    // ... dispatch to backend ...

    // The if-statement now correctly respects the no-grad context
    if (result_requires_grad) {
        // ... create backward_fn ...
    }
    
    return result;
}
// This change must be applied to `sub`, `mul`, `matmul`, `sum`, etc.
```

### Step 3.3: Update CMake

Add the new `guard.cpp` file to `src/CMakeLists.txt`:
```cmake
target_sources(vesper PRIVATE
    # ...
    autograd/engine.cpp
    autograd/guard.cpp   # Add this
    # ...
)
```

## 4. Verification

We will test the `NoGradGuard` by performing an operation inside its scope and asserting that no computational graph is created for the result.

### Step 4.1: Update `tests/test_autograd.cpp`
```cpp
// tests/test_autograd.cpp

void test_no_grad_guard() {
    std::cout << "Testing NoGradGuard..." << std::endl;

    auto a = vesper::zeros({2}, vesper::DType::Float32, vesper::Device::CPU, true);
    auto b = vesper::zeros({2}, vesper::DType::Float32, vesper::Device::CPU, true);
    
    Tensor c;
    {
        // 1. Enter the no-grad context
        vesper::autograd::NoGradGuard guard;

        // 2. Perform an operation
        c = vesper::ops::add(a, b);
    } // guard goes out of scope here, restoring grad mode

    // 3. Verification
    // Even though `a` and `b` require gradients, `c` should not, because
    // the operation happened inside the no-grad block.
    assert(!c.requires_grad());
    assert(c.grad_node == nullptr);

    // 4. Perform another operation outside the block to ensure grad mode is restored
    auto d = vesper::ops::add(a, c);
    assert(d.requires_grad());
    assert(d.grad_node != nullptr);

    std::cout << "NoGradGuard test passed!" << std::endl;
}

int main() {
    test_graph_construction();
    test_backward_pass();
    test_no_grad_guard(); // Add the new test
    return 0;
}
```
A passing test confirms that the `NoGradGuard` works as expected, preventing graph construction within its scope and restoring it upon exit. This is a critical safety feature for implementing optimizers.
