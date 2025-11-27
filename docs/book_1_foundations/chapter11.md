
# Vesper Build Plan - Chapter 11: Neural Network Foundation: The `nn::Module` Base Class

## 1. Goal

Define the `vesper::nn::Module` class, the cornerstone of all neural network models in Vesper. A `Module` is a reusable component that encapsulates a `forward` method, trainable parameters (Tensors), and potentially other sub-modules, allowing for complex models to be built by composition.

## 2. The Module Philosophy

Inspired by PyTorch, our `Module` class will serve as a base for all network layers (Linear, Conv2d) and containers (Sequential). Its primary responsibilities are:
-   **Parameter Registration**: To know which `Tensor`s are trainable parameters.
-   **Sub-module Tracking**: To compose complex models from simpler parts.
-   **A Universal Interface**: Providing `forward()`, `parameters()`, and `zero_grad()` methods.

## 3. Detailed Steps

### Step 3.1: Create `module.h` and `module.cpp`

Create the `nn` directory and the necessary files.
```sh
mkdir -p include/vesper/nn
mkdir -p src/nn
```
Create `include/vesper/nn/module.h`:
```cpp
// include/vesper/nn/module.h
#pragma once

#include <vesper/core/tensor.h>
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace vesper::nn {

class Module : public std::enable_shared_from_this<Module> {
public:
    virtual ~Module() = default;

    // The forward pass must be implemented by all subclasses
    virtual Tensor forward(const Tensor& input) {
        throw std::runtime_error("Forward not implemented.");
    };

    // Syntactic sugar to make modules callable: `model(input)`
    Tensor operator()(const Tensor& input) {
        return this->forward(input);
    }

    // Gathers all parameters from this module and its sub-modules
    std::vector<Tensor*> parameters();

    // Zeros the gradients of all parameters
    void zero_grad();

protected:
    // Methods for subclasses to register their components
    void register_parameter(const std::string& name, Tensor& param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);

private:
    std::map<std::string, Tensor*> _parameters;
    std::map<std::string, std::shared_ptr<Module>> _modules;
};

} // namespace vesper::nn
```
*Note: We store raw pointers `Tensor*` in `_parameters` because the Tensors will be owned by the subclass (e.g., `Linear` layer), not by the map.*

Create `src/nn/module.cpp`:
```cpp
// src/nn/module.cpp
#include <vesper/nn/module.h>

namespace vesper::nn {

void Module::register_parameter(const std::string& name, Tensor& param) {
    param.set_requires_grad(true); // All registered parameters are trainable
    _parameters[name] = &param;
}

void Module::register_module(const std::string& name, std::shared_ptr<Module> module) {
    _modules[name] = std::move(module);
}

std::vector<Tensor*> Module::parameters() {
    std::vector<Tensor*> params;
    // Add this module's own parameters
    for (auto const& [name, param] : _parameters) {
        params.push_back(param);
    }
    // Recursively add parameters from sub-modules
    for (auto const& [name, module] : _modules) {
        auto sub_params = module->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    return params;
}

void Module::zero_grad() {
    for (auto* param : this->parameters()) {
        if (param->requires_grad()) {
            // In C++17, we can't easily access the unique_ptr's underlying pointer
            // without a getter. For now, we'll just re-assign the gradient.
            // A better way would be a `grad()` method that returns a nullable pointer.
            // Or a fill_(0) method. Let's assume we create grad() lazily
            param->grad() = zeros(param->shape(), param->dtype(), param->device());
        }
    }
}

} // namespace vesper::nn
```

### Step 3.2: Update `tensor.h` for `zero_grad`
The `grad()` method in `Tensor` creates a zero-filled tensor. `zero_grad` can just call this and assign. To make this work robustly, `grad()` should only create it if it doesn't exist. Let's refine `grad()` and `zero_grad`.

In `src/core/tensor.cpp`, `Tensor::grad()` is already implemented to lazily create a zero tensor. Let's make `zero_grad()` use this.
In `src/nn/module.cpp`, the `zero_grad` implementation needs a slight correction. `param->grad()` returns a *reference* to the gradient tensor. We need to fill it with zeros.
```cpp
// src/nn/module.cpp -> corrected zero_grad
void Module::zero_grad() {
    for (auto* param : this->parameters()) {
        if (param->requires_grad()) {
            // Re-create the gradient tensor, effectively zeroing it.
            // A more efficient `fill_(0)` method is a future optimization.
            param->grad() = zeros(param->shape(), param->dtype(), param->device());
        }
    }
}
```

### Step 3.3: Update CMake
Add the new `module.cpp` to `src/CMakeLists.txt`.
```cmake
target_sources(vesper PRIVATE
    # ...
    autograd/engine.cpp
    nn/module.cpp      # Add this
)
```

## 4. Code Structure Suggestions

-   **Composition over Inheritance**: Models are built by composing modules. A `ResNet` model would contain several `ResBlock` modules, which in turn contain `Conv2d` and `BatchNorm` modules. The `parameters()` and `zero_grad()` calls cascade through this hierarchy automatically.
-   **Protected Registration**: Making `register_parameter` and `register_module` `protected` ensures that only the class itself (and its subclasses) can add to its state, providing good encapsulation.
-   **`enable_shared_from_this`**: This is crucial for allowing modules to be safely managed by `shared_ptr` and registered within other modules.

## 5. Potential Pitfalls

-   **Parameter Ownership**: We've decided that the `Module` subclass (e.g., `Linear`) owns its parameter Tensors as member variables, and the `_parameters` map just stores non-owning pointers (`Tensor*`). This avoids double ownership. The alternative is to have the `_parameters` map own them via `shared_ptr`, but this can be less intuitive for the subclass to access its own weights.
-   **Name Clashes**: Using a `std::map` for parameters and modules means names must be unique. Registering two parameters with the same name will silently overwrite the first one.

## 6. Verification

We will create a dummy `TestModule` with a sub-module to verify that parameter registration, collection, and gradient zeroing work recursively.

### Step 6.1: Create `tests/test_nn.cpp`
```cpp
// tests/test_nn.cpp
#include <vesper/nn/module.h>
#include <iostream>
#include <cassert>

// A dummy sub-module
class SubModule : public vesper::nn::Module {
public:
    SubModule() {
        bias = vesper::zeros({8}, vesper::DType::Float32, vesper::Device::CPU);
        register_parameter("bias", bias);
    }
    vesper::Tensor bias;
};

// A dummy top-level module
class TestModule : public vesper::nn::Module {
public:
    TestModule() {
        // Create and register a parameter
        weights = vesper::ones({16, 8}, vesper::DType::Float32, vesper::Device::CPU);
        register_parameter("weights", weights);

        // Create and register a sub-module
        sub = std::make_shared<SubModule>();
        register_module("sub", sub);
    }
    vesper::Tensor weights;
    std::shared_ptr<SubModule> sub;
};

void test_module_structure() {
    std::cout << "Testing nn::Module structure..." << std::endl;

    auto model = std::make_shared<TestModule>();

    // 1. Verify parameter collection
    auto params = model->parameters();
    assert(params.size() == 2); // Should find model.weights and model.sub.bias

    // 2. Verify zero_grad
    // Get a parameter and manually set its gradient to something non-zero
    Tensor* bias_param = model->sub->parameters()[0];
    assert(bias_param->requires_grad());
    
    // Manually set a non-zero gradient
    bias_param->grad() = vesper::ones(bias_param->shape(), bias_param->dtype(), bias_param->device());
    
    // Verify grad is not zero
    std::vector<float> grad_vec(bias_param->grad().numel());
    bias_param->grad().copy_to_host(grad_vec.data());
    assert(grad_vec[0] == 1.0f);

    // Now, zero all gradients
    model->zero_grad();

    // Verify grad is now zero
    bias_param->grad().copy_to_host(grad_vec.data());
    assert(grad_vec[0] == 0.0f);

    std::cout << "nn::Module structure test passed!" << std::endl;
}


int main() {
    test_module_structure();
    return 0;
}
```

### Step 6.2: Add to `tests/CMakeLists.txt`
```cmake
add_executable(nn_module_tests test_nn.cpp)
target_link_libraries(nn_module_tests PRIVATE vesper)
add_test(NAME NnModuleTests COMMAND nn_module_tests)
```

With a solid `Module` base class, you are now ready to implement your first real neural network layer in the next chapter.

---

## Addendum: Advanced Module Features

As Vesper matured, several advanced features were added to the `Module` class to support production-quality models:

### A.1 Member-Pointer Registration (Type-Safe)

The original `register_parameter(name, &tensor)` syntax using raw pointers works but doesn't verify type safety. The member-pointer syntax provides compile-time verification:

```cpp
class MyLayer : public Module {
public:
    MyLayer(int64_t in_features, int64_t out_features)
        : weight(randn({out_features, in_features}, DType::Float32, Device::CPU, true))
    {
        // Member-pointer syntax: type-safe, verified at compile time
        register_parameter<MyLayer>("weight", &MyLayer::weight);
    }
    
    Tensor weight;
};
```

**Benefits:**
- Compile-time verification that the member belongs to the class
- Clearer intent in the code
- Works with `to(device)` for proper device transfer

### A.2 ModuleList<T> Container

For models with variable numbers of identical layers (Transformers, ResNets), `ModuleList<T>` provides a type-safe container:

```cpp
#include <vesper/nn/module_list.h>

class TransformerStack : public Module {
public:
    TransformerStack(int num_layers, int d_model) {
        for (int i = 0; i < num_layers; ++i) {
            layers.emplace_back(d_model);  // Constructs TransformerBlock in-place
        }
        register_module_list("layers", &layers);
    }
    
    Tensor forward(const Tensor& x) override {
        Tensor out = x;
        for (auto& layer : layers) {
            out = layer.forward(out);
        }
        return out;
    }
    
    ModuleList<TransformerBlock> layers;
};
```

**Key Properties:**
- Uses `std::vector<std::unique_ptr<T>>` internally for stable addresses
- Automatically handles device transfer via `to(device)`
- Parameters collected recursively for optimizers
- State dict serialization with indexed keys (`layers.0.weight`, `layers.1.weight`, etc.)
- Full iterator support for range-based for loops

**API:**
| Method | Description |
|--------|-------------|
| `emplace_back(args...)` | Construct module in-place |
| `append(module)` | Add by copy/move |
| `operator[](i)` | Access module at index |
| `size()`, `empty()` | Capacity queries |
| `begin()`, `end()` | Iterator support |
| `parameters()` | Collect all parameters |
| `to(device)` | Move all modules to device |

### A.3 Pointer-Returning Methods

For advanced use cases (custom optimizers, gradient manipulation), pointer-returning methods are available:

```cpp
// Get raw pointers to parameters
std::vector<Tensor*> params = model.parameters_ptrs();

// Get named parameters as pointers
std::map<std::string, Tensor*> named = model.named_parameters_ptrs();

// Useful for custom gradient updates
for (Tensor* p : params) {
    p->grad() *= 0.5f;  // Gradient scaling
}
```

### A.4 In-Place Device Transfer

The `to(device)` method now properly moves parameters in-place:

```cpp
MyModel model(512, 1024);
model.to(Device::HIP(0));  // All parameters moved to GPU

// For individual tensors, to_() provides explicit in-place transfer
Tensor x = randn({100, 100}, DType::Float32, Device::CPU);
x.to_(Device::HIP(0));  // x is now on GPU, no copy returned
```

### A.5 Additional Tensor In-Place Operations

```cpp
// Copy data from another tensor (must have same shape)
Tensor dst = zeros({10, 10});
Tensor src = randn({10, 10});
dst.copy_(src);  // dst now contains src's data

// Zero out a tensor
Tensor weights = randn({100, 100});
weights.zero_();  // All elements set to 0
```
