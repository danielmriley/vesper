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
Tensor a = zeros({2, 3}, DType::Float32, Device::CUDA);

// Create a tensor from a vector
Tensor b = full({2, 3}, DType::Float32, Device::CPU, 1.0f);

// Move data between devices
Tensor c = b.to(Device::CUDA); // Not yet implemented directly, use copy helpers
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

```cpp
#include <vesper/nn/linear.h>
#include <vesper/nn/functional.h>

using namespace vesper;

class SimpleMLP : public nn::Module {
public:
    SimpleMLP(int in_features, int hidden_features, int out_features) {
        // Initialize layers
        fc1 = std::make_shared<nn::Linear>(in_features, hidden_features);
        fc2 = std::make_shared<nn::Linear>(hidden_features, out_features);
        
        // Register parameters so the optimizer can find them
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    Tensor forward(const Tensor& x) override {
        Tensor out = fc1->forward(x);
        out = nn::functional::relu(out);
        out = fc2->forward(out);
        return out;
    }

private:
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::Linear> fc2;
};
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
    vesper::Device device = vesper::Device::CUDA; // or Device::CPU

    // 2. Create Model
    SimpleMLP model(784, 128, 10);
    model.to(device); // Move parameters to GPU (if implemented, otherwise init on device)

    // 3. Create Optimizer
    // SGD with learning rate 0.01
    vesper::optim::SGD optimizer(model.parameters(), 0.01);

    // 4. Create Loss Function
    vesper::nn::MSELoss criterion;

    // 5. Dummy Data (Batch size 64)
    auto input = vesper::empty({64, 784}, vesper::DType::Float32, device);
    auto target = vesper::empty({64, 10}, vesper::DType::Float32, device);
    
    // Initialize with random data (using uniform_ for now)
    vesper::ops::uniform_(input, 0.0f, 1.0f);
    vesper::ops::uniform_(target, 0.0f, 1.0f);

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

        // Print loss (requires copying to CPU to print)
        // float loss_val = loss.item<float>(); 
        std::cout << "Epoch " << epoch << " finished." << std::endl;
    }

    return 0;
}
```

## 5. Current Limitations (as of Chapter 21)

*   **Batch GEMM**: `matmul` only supports 2D tensors. For 3D/4D tensors (needed for Attention), you must loop or reshape manually.
*   **Loss Functions**: Only `MSELoss` is currently implemented. For classification, you may need to implement `CrossEntropyLoss` or `Softmax`.
*   **Optimizers**: Only `SGD` is available.
*   **Layers**: `Linear` is the primary layer. `Conv2d`, `Embedding`, `LayerNorm` are not yet implemented.

This library is currently in an **educational/experimental** state. It is excellent for learning how deep learning frameworks work under the hood but is not yet a drop-in replacement for PyTorch for complex production models like LLMs.
