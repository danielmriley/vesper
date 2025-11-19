
# Vesper Build Plan - Chapter 16: First Optimizer: Implementing `optim::SGD`

## 1. Goal

Implement the Stochastic Gradient Descent (SGD) optimization algorithm. This will be the first concrete subclass of `optim::Optimizer` and will be responsible for applying gradient updates to a model's parameters.

## 2. Prerequisites

-   Chapter 10.1: The `NoGradGuard` is implemented and integrated into all ops.
-   Chapter 15: The `optim::Optimizer` base class exists.
-   Chapter 5.1: Scalar multiplication (`ops::mul(Tensor, float)`) is available.

## 3. The SGD Algorithm

The update rule for SGD is `parameter = parameter - learning_rate * gradient`. Our `step()` method will implement this rule for every parameter the optimizer manages. The entire update must happen within a `NoGradGuard` scope to prevent autograd from tracking it.

## 4. Detailed Steps

### Step 4.1: Create `sgd.h`

Create the header `include/vesper/optim/sgd.h`:
```cpp
// include/vesper/optim/sgd.h
#pragma once

#include <vesper/optim/optimizer.h>

namespace vesper::optim {

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor*> params, float lr = 0.01f);
    
    void step() override;

private:
    float lr_;
};

} // namespace vesper::optim
```

### Step 4.2: Implement `SGD::step()`

Create `src/optim/sgd.cpp`. The `step()` method iterates through the parameters and applies the update using Vesper's own ops.

```cpp
// src/optim/sgd.cpp
#include <vesper/optim/sgd.h>
#include <vesper/autograd/guard.h>
#include <vesper/ops/elementwise.h> // For sub and mul

namespace vesper::optim {

SGD::SGD(std::vector<Tensor*> params, float lr)
    : Optimizer(std::move(params)), lr_(lr) {}

void SGD::step() {
    // === Enter No-Grad context ===
    autograd::NoGradGuard guard;

    for (auto* param : params_) {
        if (param && param->requires_grad()) {
            auto& grad = param->grad();
            
            // 1. update = lr * grad
            auto update = ops::mul(grad, lr_); 

            // 2. param = param - update
            // This reassigns the tensor, it's not a true in-place update.
            // A future optimization is an in-place kernel.
            *param = ops::sub(*param, update);
        }
    }
}

} // namespace vesper::optim
```

### Step 4.3: Update CMake

Add the new `sgd.cpp` file to `src/CMakeLists.txt`:
```cmake
target_sources(vesper PRIVATE
    # ...
    autograd/guard.cpp
    optim/optimizer.cpp
    optim/sgd.cpp        # Add this
)
```

## 5. Verification

The test for `SGD` ensures that the `step()` method correctly applies the update rule to a parameter.

### Step 5.1: Update `tests/test_optimizer.cpp`

```cpp
// tests/test_optimizer.cpp
#include <vesper/optim/sgd.h>

void test_sgd_step() {
    std::cout << "Testing SGD step..." << std::endl;
    
    // 1. Create a parameter with an initial value and a gradient
    auto param = vesper::empty({1}, vesper::DType::Float32, vesper::Device::CPU, true);
    param.copy_from_host(std::vector<float>{10.0f}.data());
    param.grad() = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 2.0f);
    
    // 2. Create the optimizer
    std::vector<vesper::Tensor*> params = {&param};
    auto optimizer = vesper::optim::SGD(params, 0.1f); // lr = 0.1

    // 3. Perform the update
    optimizer.step();

    // 4. Verify the new value
    // Expected: 10.0 - (0.1 * 2.0) = 9.8
    std::vector<float> param_data(1);
    param.copy_to_host(param_data.data());
    assert(std::fabs(param_data[0] - 9.8f) < 1e-6);

    std::cout << "SGD step test passed!" << std::endl;
}

int main() {
    test_optimizer_base();
    test_sgd_step();
    return 0;
}
```

A passing test confirms that the `SGD` optimizer correctly modifies parameters. The library now has a complete pipeline: forward pass, loss calculation, backward pass, and parameter updates, ready for a full training loop.
