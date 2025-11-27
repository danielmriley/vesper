# Vesper Usage Guide

Welcome to Vesper! This guide will help you integrate the Vesper deep learning library into your own C++ projects. Vesper is designed to be a lightweight, dependency-free (aside from GPU drivers) tensor library inspired by PyTorch.

## 1. Integration with CMake

Vesper is built as a static library. To use it in your project, you can include it as a subdirectory in your CMake project.

### Directory Structure
```
my_project/
├── CMakeLists.txt
├── main.cpp
└── extern/
    └── vesper/  (Clone the vesper repository here)
```

### CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyNeuralNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Configure Vesper options before adding subdirectory
option(USE_CUDA "Enable CUDA backend" ON) # or USE_HIP / USE_CPU
add_subdirectory(extern/vesper)

add_executable(my_net main.cpp)

# Link against Vesper
target_link_libraries(my_net PRIVATE vesper)

# If using CUDA, you might need to link CUDA runtime as well
if(USE_CUDA)
    target_link_libraries(my_net PRIVATE CUDA::cudart)
endif()
```

## 2. Core Concepts

### Tensors
The core data structure is the `Tensor`. It holds data, shape, and device information.

```cpp
#include <vesper/core/factories.h>
#include <vesper/core/tensor.h>

using namespace vesper;

// Create a 2x3 tensor of zeros on the GPU
Tensor a = zeros({2, 3}, DType::Float32, Device::CPU);

// Create a tensor from a vector
Tensor b = full({2, 3}, DType::Float32, Device::CPU, 1.0f);

// Move data between devices (returns new tensor)
Tensor c = b.to(Device::HIP(0));

// In-place device transfer (modifies tensor directly)
b.to_(Device::HIP(0));  // b is now on GPU
```

### In-Place Tensor Operations

Vesper provides several in-place tensor operations (denoted by trailing underscore):

```cpp
// In-place device transfer
Tensor x = randn({100, 100}, DType::Float32, Device::CPU);
x.to_(Device::HIP(0));  // Move to GPU without creating a copy

// Copy data from another tensor (same shape required)
Tensor src = randn({10, 10}, DType::Float32, Device::CPU);
Tensor dst = zeros({10, 10}, DType::Float32, Device::CPU);
dst.copy_(src);  // dst now contains src's data

// Zero out a tensor
Tensor weights = randn({100, 100}, DType::Float32, Device::CPU);
weights.zero_();  // All elements set to 0
```

### Autograd
Vesper supports automatic differentiation. Set `requires_grad=true` when creating tensors.

```cpp
Tensor x = full({2, 2}, DType::Float32, Device::CUDA, 1.0f, true); // requires_grad=true
Tensor y = ops::add(x, x);
Tensor z = ops::mul(y, y);

z.backward(); // Computes gradients

// Access gradients
// x.grad() will contain the gradient d(z)/d(x)
```

## 3. Building Neural Networks

Use the `vesper::nn` namespace to build layers and modules.

### Basic Module Pattern

```cpp
#include <vesper/nn/linear.h>
#include <vesper/nn/functional.h>

using namespace vesper;

class SimpleMLP : public nn::Module {
public:
    SimpleMLP(int in_features, int hidden_features, int out_features) 
        : fc1(in_features, hidden_features),
          fc2(hidden_features, out_features)
    {
        // Register submodules using member-pointer syntax
        register_module<SimpleMLP>("fc1", &SimpleMLP::fc1);
        register_module<SimpleMLP>("fc2", &SimpleMLP::fc2);
    }

    Tensor forward(const Tensor& x) override {
        Tensor out = fc1.forward(x);
        out = nn::functional::relu(out);
        out = fc2.forward(out);
        return out;
    }

    nn::Linear fc1;
    nn::Linear fc2;
};
```

### Using ModuleList for Dynamic Layers

For models with variable numbers of layers (like Transformers), use `ModuleList<T>`:

```cpp
#include <vesper/nn/module_list.h>
#include <vesper/nn/linear.h>

using namespace vesper;
using namespace vesper::nn;

class DeepMLP : public Module {
public:
    DeepMLP(int input_dim, int hidden_dim, int num_layers) {
        // Add layers dynamically
        layers.emplace_back(input_dim, hidden_dim);
        for (int i = 1; i < num_layers - 1; ++i) {
            layers.emplace_back(hidden_dim, hidden_dim);
        }
        layers.emplace_back(hidden_dim, input_dim);
        
        // Register the ModuleList - all layers automatically tracked
        register_module_list("layers", &layers);
    }

    Tensor forward(const Tensor& x) override {
        Tensor out = x;
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            out = nn::functional::relu(layers[i].forward(out));
        }
        return layers.back().forward(out);  // No activation on last layer
    }

    ModuleList<Linear> layers;
};
```

**ModuleList Features:**
- Type-safe container that maintains stable addresses (safe with `to()`)
- Automatic parameter collection from all contained modules
- Supports iteration with range-based for loops
- Methods: `emplace_back()`, `append()`, `operator[]`, `size()`, `empty()`

### Moving Models to GPU

```cpp
DeepMLP model(784, 256, 4);

// Move entire model to GPU (all parameters, including those in ModuleList)
model.to(Device::HIP(0));

// Or use in-place variant on individual tensors
// (Module::to() already handles this internally)
```

## 4. Training Loop Example

Here is a complete example of how to train a simple network.

```cpp
#include <vesper/core/factories.h>
#include <vesper/nn/loss.h>
#include <vesper/optim/sgd.h>
#include <iostream>

int main() {
    // 1. Setup Device
    vesper::Device device = vesper::Device::HIP(0);

    // 2. Create Model
    SimpleMLP model(784, 128, 10);
    model.to(device); // Move all parameters to GPU

    // 3. Create Optimizer
    // SGD with learning rate 0.01
    vesper::optim::SGD optimizer(model.parameters(), 0.01);

    // 4. Create Loss Function
    vesper::nn::MSELoss criterion;

    // 5. Dummy Data (Batch size 64)
    auto input = vesper::randn({64, 784}, vesper::DType::Float32, device, true);
    auto target = vesper::randn({64, 10}, vesper::DType::Float32, device);

    // 6. Training Loop
    for (int epoch = 0; epoch < 10; ++epoch) {
        // Zero gradients
        optimizer.zero_grad();

        // Forward pass
        auto output = model.forward(input);

        // Compute loss
        auto loss = criterion.forward(output, target);

        // Backward pass
        loss.backward();

        // Update weights
        optimizer.step();

        std::cout << "Epoch " << epoch << " finished." << std::endl;
    }

    return 0;
}
```

## 5. API Quick Reference

### Tensor In-Place Operations
| Method | Description |
|--------|-------------|
| `to_(Device, bool non_blocking=false)` | Move tensor to device in-place |
| `copy_(const Tensor& src)` | Copy data from another tensor |
| `zero_()` | Set all elements to zero |

### Module Registration Methods
| Method | Description |
|--------|-------------|
| `register_parameter<T>(name, &T::member)` | Register parameter using member-pointer |
| `register_parameter(name, Tensor*)` | Register parameter using raw pointer |
| `register_module<T>(name, &T::member)` | Register submodule using member-pointer |
| `register_module_list(name, ModuleList<T>*)` | Register a ModuleList |

### Module Utility Methods
| Method | Description |
|--------|-------------|
| `parameters()` | Get all parameters as `std::vector<Tensor>` |
| `parameters_ptrs()` | Get all parameters as `std::vector<Tensor*>` |
| `named_parameters_ptrs()` | Get parameters as `std::map<std::string, Tensor*>` |
| `to(Device, bool non_blocking=false)` | Move all parameters to device |
| `train(bool mode=true)` | Set training mode |
| `eval()` | Set evaluation mode |
| `zero_grad()` | Zero all parameter gradients |

## 6. Current Status

Vesper is production-ready for LLM research and experimentation:

**Fully Implemented:**
- Tensors with autograd support
- Device management (CPU, HIP/ROCm)
- Core layers: Linear, Embedding, LayerNorm, RMSNorm, Dropout
- Transformer components: MultiHeadAttention, RoPE, KV-Cache, SwiGLU
- Loss functions: MSELoss, CrossEntropyLoss
- Optimizers: SGD, Adam, AdamW
- Learning rate schedulers
- Model serialization (state_dict save/load)
- ModuleList for dynamic model architectures

**Backend:**
- HIP/ROCm: Primary supported backend with optimized kernels
- CPU: Full support for development and testing

This library is suitable for implementing and training LLM models like GPT, Llama, and Mistral architectures.
