
# Vesper Build Plan - Chapter 15: Optimization Foundation: The `optim::Optimizer` Base Class

## 1. Goal

Define the abstract `vesper::optim::Optimizer` base class. This class establishes the contract for all optimization algorithms (like SGD and Adam), which are responsible for updating a model's parameters using their computed gradients. This chapter focuses on creating the structural interface, not a specific algorithm.

## 2. The Optimizer's Role

In a training loop, after `loss.backward()` has populated the `.grad` attribute of all parameters, the optimizer performs the update step. Its core jobs are:
1.  To hold a reference to all the parameters it needs to update.
2.  To provide a `step()` method that applies the specific optimization algorithm (e.g., `weight = weight - learning_rate * gradient`).
3.  To provide a `zero_grad()` method to clear old gradients before the next iteration.

## 3. Detailed Steps

### Step 3.1: Create `optimizer.h`

This file will define the abstract base class.

Create the `optim` directory and the header file:
```sh
mkdir -p include/vesper/optim
```
Create `include/vesper/optim/optimizer.h`:
```cpp
// include/vesper/optim/optimizer.h
#pragma once

#include <vesper/core/tensor.h>
#include <vector>

namespace vesper::optim {

class Optimizer {
public:
    // The constructor takes a vector of non-owning pointers to the parameters
    explicit Optimizer(std::vector<Tensor*> params);
    
    virtual ~Optimizer() = default;

    // `step` must be implemented by concrete optimizers (e.g., SGD, Adam)
    virtual void step() = 0;

    // A convenience method to zero the gradients of all managed parameters
    void zero_grad();

protected:
    std::vector<Tensor*> params_;
};

} // namespace vesper::optim
```

### Step 3.2: Create `optimizer.cpp`

This file will implement the non-pure-virtual methods of the `Optimizer` class (the constructor and `zero_grad`).

Create `src/optim/optimizer.cpp`:
```sh
mkdir -p src/optim
```
```cpp
// src/optim/optimizer.cpp
#include <vesper/optim/optimizer.h>
#include <vesper/core/factories.h> // For zeros()

namespace vesper::optim {

Optimizer::Optimizer(std::vector<Tensor*> params) : params_(std::move(params)) {}

void Optimizer::zero_grad() {
    for (auto* param : params_) {
        if (param && param->requires_grad()) {
            // This re-creates the gradient tensor, effectively zeroing it.
            // A more efficient `fill_(0)` method is a future optimization.
            param->grad() = zeros(param->shape(), param->dtype(), param->device());
        }
    }
}

} // namespace vesper::optim
```

### Step 3.3: Update CMake

Add the new source file to `src/CMakeLists.txt`:
```cmake
target_sources(vesper PRIVATE
    # ...
    nn/functional.cpp
    ops/hip/activation.hip
    optim/optimizer.cpp      # Add this
)
```

## 4. Code Structure Suggestions

-   **Abstract Interface**: By defining `step()` as a pure virtual function (`= 0`), we enforce that this class cannot be instantiated on its own. Users must choose a concrete algorithm like `SGD`. This is a classic application of polymorphism.
-   **Non-Owning Pointers**: The optimizer stores `Tensor*`. This is critical. The `nn::Module` that defines the parameters is their true owner. The optimizer merely observes them. This prevents ownership conflicts and memory management issues.
-   **Convenience of `zero_grad`**: While `module->zero_grad()` and `optimizer->zero_grad()` do the same thing, providing it on the optimizer is conventional and makes training loops cleaner: `optimizer.zero_grad(); loss.backward(); optimizer.step();`.

## 5. Potential Pitfalls

-   **Dangling Pointers**: Because the optimizer holds raw pointers, it's the user's responsibility to ensure the `Module` (and its parameters) outlives the `Optimizer`. If the model is destroyed but the optimizer is not, calling `step()` would result in undefined behavior. Using `std::vector<std::weak_ptr<Tensor>>` would be safer but adds complexity. For a controlled training loop, raw pointers are acceptable.
-   **Forgetting `virtual` Destructor**: A base class with virtual methods should always have a virtual destructor. This ensures that when a derived class pointer is `delete`d via a base class pointer, the correct destructor is called.

## 6. Verification

Since this is an abstract class, we can't test it directly. Instead, we'll create a minimal, concrete "dummy" optimizer to test the `zero_grad` functionality.

### Step 6.1: Create `tests/test_optimizer.cpp`
```cpp
// tests/test_optimizer.cpp
#include <vesper/optim/optimizer.h>
#include <vesper/nn/module.h> // To create a dummy module
#include <iostream>
#include <cassert>

// 1. A dummy module with one parameter
class SimpleModel : public vesper::nn::Module {
public:
    SimpleModel() {
        weight = vesper::ones({4}, vesper::DType::Float32, vesper::Device::CPU);
        register_parameter("weight", weight);
    }
    vesper::Tensor weight;
};

// 2. A dummy optimizer to make the base class concrete
class DummyOptimizer : public vesper::optim::Optimizer {
public:
    using vesper::optim::Optimizer::Optimizer; // Inherit constructor
    
    // Provide a dummy implementation for the pure virtual function
    void step() override {
        // Does nothing for this test
    }
};

void test_optimizer_base() {
    std::cout << "Testing Optimizer base..." << std::endl;

    // 3. Set up the model and optimizer
    auto model = SimpleModel();
    auto optimizer = DummyOptimizer(model.parameters());

    // 4. Manually give the parameter a gradient
    auto param = model.parameters()[0];
    param->grad() = vesper::ones(param->shape(), param->dtype(), param->device());
    
    // Check that the gradient is non-zero
    std::vector<float> grad_vec(param->numel());
    param->grad().copy_to_host(grad_vec.data());
    assert(grad_vec[0] == 1.0f);

    // 5. Call the optimizer's zero_grad method
    optimizer.zero_grad();

    // 6. Verify that the parameter's gradient is now zero
    param->grad().copy_to_host(grad_vec.data());
    assert(grad_vec[0] == 0.0f);

    std::cout << "Optimizer base test passed!" << std::endl;
}


int main() {
    test_optimizer_base();
    return 0;
}
```

### Step 6.2: Add to `tests/CMakeLists.txt`
```cmake
add_executable(optimizer_tests test_optimizer.cpp)
target_link_libraries(optimizer_tests PRIVATE vesper)
add_test(NAME OptimizerTests COMMAND optimizer_tests)
```
A passing test confirms that the optimizer's structure is sound, it correctly holds references to parameters, and its `zero_grad` utility functions as expected. We are now ready to implement our first real optimization algorithm.
