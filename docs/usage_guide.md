# Vesper Usage Guide

Welcome to Vesper! This guide will help you integrate the Vesper deep learning library into your own C++ projects. Vesper is designed to be a lightweight, dependency-free (aside from GPU drivers) tensor library inspired by PyTorch.

## Table of Contents

1. [Installation](#1-installation)
2. [Integration Methods](#2-integration-methods)
3. [Core Concepts](#3-core-concepts)
4. [Building Neural Networks](#4-building-neural-networks)
5. [Training Example](#5-training-example)
6. [Complete Project Example](#6-complete-project-example)
7. [API Quick Reference](#7-api-quick-reference)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| C++ Compiler | GCC 9+, Clang 10+, or MSVC 2019+ |
| CMake | 3.15+ |
| ROCm (AMD GPUs) | 5.0+ with hipcc |
| CUDA (NVIDIA GPUs) | 11.0+ |

### Building Vesper

#### For AMD GPUs (HIP/ROCm)

```bash
git clone https://github.com/danielmriley/vesper.git
cd vesper
mkdir build && cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX=$HOME/.local \
    -DUSE_HIP=ON \
    -DUSE_CUDA=OFF \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
    -DVESPER_BUILD_TESTS=OFF

make -j$(nproc)
make install
```

#### For NVIDIA GPUs (CUDA)

```bash
git clone https://github.com/danielmriley/vesper.git
cd vesper
mkdir build && cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX=$HOME/.local \
    -DUSE_CUDA=ON \
    -DUSE_HIP=OFF \
    -DVESPER_BUILD_TESTS=OFF

make -j$(nproc)
make install
```

### Environment Setup

Add to your shell configuration (`~/.bashrc` or `~/.zshrc`):

```bash
export CMAKE_PREFIX_PATH=$HOME/.local:$CMAKE_PREFIX_PATH
```

---

## 2. Integration Methods

### Method 1: find_package (Recommended)

After installing Vesper, create your project:

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(vesper 1.0 REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp vesper::vesper)
```

**Build:**
```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$HOME/.local
make
```

### Method 2: Subdirectory

Include Vesper directly in your project:

```bash
git submodule add https://github.com/danielmriley/vesper.git extern/vesper
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Configure Vesper before adding subdirectory
set(USE_HIP ON CACHE BOOL "Enable HIP backend")
set(USE_CUDA OFF CACHE BOOL "Disable CUDA backend")
set(VESPER_BUILD_TESTS OFF CACHE BOOL "Skip tests")

add_subdirectory(extern/vesper)

add_executable(myapp main.cpp)
target_link_libraries(myapp vesper)
```

---

## 3. Core Concepts

### Include Everything

Use the convenience header for all functionality:

```cpp
#include <vesper/vesper.h>

using namespace vesper;
```

### Tensors

```cpp
// Create tensors
Tensor a = zeros({2, 3}, DType::Float32, Device::CPU);
Tensor b = ones({2, 3}, DType::Float32, Device::HIP);
Tensor c = randn({4, 4}, DType::Float32, Device::HIP, true);  // requires_grad=true

// Factory functions
Tensor d = full({3, 3}, DType::Float32, Device::HIP, 2.5f);
Tensor e = empty({100, 100}, DType::Float32, Device::HIP, true);
```

### Device Selection

```cpp
// Runtime device selection based on compile configuration
#if defined(USE_HIP_BACKEND)
    Device device = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    Device device = Device::CUDA;
#else
    Device device = Device::CPU;
#endif

Tensor x = randn({100, 100}, DType::Float32, device, true);
```

### Moving Data

```cpp
// Copy to a new tensor on different device
Tensor gpu_tensor = cpu_tensor.to(Device::HIP);

// In-place move (modifies original tensor)
cpu_tensor.to_(Device::HIP);

// Copy data to/from host
std::vector<float> host_data(100);
tensor.copy_to_host(host_data.data());
tensor.copy_from_host(host_data.data());
```

### Autograd

```cpp
// Create tensor with gradient tracking
Tensor x = randn({3, 3}, DType::Float32, Device::HIP, true);  // requires_grad=true

// Forward operations
Tensor y = x * x;
Tensor z = y.sum();

// Backward pass
z.backward();

// Access gradients
Tensor grad = x.grad();  // Contains dz/dx

// Disable gradient computation for inference
{
    autograd::NoGradGuard no_grad;
    Tensor pred = model.forward(input);  // No gradients tracked
}
```

---

## 4. Building Neural Networks

### Basic Module

```cpp
#include <vesper/vesper.h>

using namespace vesper;

class SimpleMLP : public nn::Module {
public:
    nn::Linear fc1{784, 128};
    nn::Linear fc2{128, 10};
    
    SimpleMLP(Device device) {
        fc1.to(device);
        fc2.to(device);
        register_module("fc1", &fc1);
        register_module("fc2", &fc2);
    }
    
    Tensor forward(const Tensor& x) {
        auto h = nn::functional::relu(fc1(x));
        return fc2(h);
    }
};
```

### ModuleList for Dynamic Architectures

```cpp
class DeepMLP : public nn::Module {
public:
    nn::ModuleList<nn::Linear> layers;
    
    DeepMLP(int input_dim, int hidden_dim, int num_layers, Device device) {
        layers.emplace_back(input_dim, hidden_dim);
        for (int i = 1; i < num_layers - 1; ++i) {
            layers.emplace_back(hidden_dim, hidden_dim);
        }
        layers.emplace_back(hidden_dim, 10);
        
        register_module_list("layers", &layers);
        to(device);
    }
    
    Tensor forward(const Tensor& x) {
        Tensor out = x;
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            out = nn::functional::relu(layers[i](out));
        }
        return layers.back()(out);
    }
};
```

### Available Layers

| Layer | Description |
|-------|-------------|
| `nn::Linear` | Fully connected layer |
| `nn::Embedding` | Lookup table embeddings |
| `nn::Conv2d` | 2D convolution |
| `nn::LayerNorm` | Layer normalization |
| `nn::RMSNorm` | RMS normalization |
| `nn::Dropout` | Dropout regularization |
| `nn::MultiHeadAttention` | Multi-head attention |
| `nn::RoPE` | Rotary position embeddings |
| `nn::SwiGLU` | Gated activation |

### Activation Functions

```cpp
Tensor y = nn::functional::relu(x);
Tensor y = nn::functional::gelu(x);
Tensor y = nn::functional::sigmoid(x);
Tensor y = nn::functional::tanh(x);
Tensor y = nn::functional::softmax(x, /*dim=*/-1);
```

---

## 5. Training Example

```cpp
#include <vesper/vesper.h>
#include <iostream>

using namespace vesper;

int main() {
    // Select device
#if defined(USE_HIP_BACKEND)
    Device device = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    Device device = Device::CUDA;
#else
    Device device = Device::CPU;
#endif

    // Create model
    SimpleMLP model(device);
    
    // Create optimizer
    optim::Adam optimizer(model.parameters(), /*lr=*/0.001f);
    
    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        // Create batch (normally from DataLoader)
        Tensor x = randn({32, 784}, DType::Float32, device, true);
        Tensor y = randn({32, 10}, DType::Float32, device);
        
        // Forward pass
        Tensor pred = model.forward(x);
        Tensor loss = nn::functional::mse_loss(pred, y);
        
        // Backward pass
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        // Print progress
        if (epoch % 10 == 0) {
            float loss_val;
            loss.copy_to_host(&loss_val);
            std::cout << "Epoch " << epoch << " Loss: " << loss_val << std::endl;
        }
    }
    
    return 0;
}
```

---

## 6. Complete Project Example

Here's a complete example project structure:

```
my_project/
├── CMakeLists.txt
├── src/
│   └── main.cpp
└── build/
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(XORDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(vesper 1.0 REQUIRED)

add_executable(xor_demo src/main.cpp)
target_link_libraries(xor_demo vesper::vesper)
```

**src/main.cpp:**
```cpp
#include <vesper/vesper.h>
#include <iostream>
#include <iomanip>

using namespace vesper;

// Neural network for XOR
class XORNet : public nn::Module {
public:
    nn::Linear fc1{2, 8};
    nn::Linear fc2{8, 1};
    
    XORNet(Device device) {
        fc1.to(device);
        fc2.to(device);
        register_module("fc1", &fc1);
        register_module("fc2", &fc2);
    }
    
    Tensor forward(const Tensor& x) {
        auto h = nn::functional::relu(fc1(x));
        return nn::functional::sigmoid(fc2(h));
    }
};

int main() {
    // Select device
#if defined(USE_HIP_BACKEND)
    Device device = Device::HIP;
    std::cout << "Using HIP backend" << std::endl;
#elif defined(USE_CUDA_BACKEND)
    Device device = Device::CUDA;
    std::cout << "Using CUDA backend" << std::endl;
#else
    Device device = Device::CPU;
    std::cout << "Using CPU backend" << std::endl;
#endif

    // XOR training data
    std::vector<float> x_data = {0, 0, 0, 1, 1, 0, 1, 1};
    std::vector<float> y_data = {0, 1, 1, 0};
    
    Tensor x = empty({4, 2}, DType::Float32, device, true);
    Tensor y = empty({4, 1}, DType::Float32, device, false);
    x.copy_from_host(x_data.data());
    y.copy_from_host(y_data.data());
    
    // Create model and optimizer
    XORNet model(device);
    optim::Adam optimizer(model.parameters(), 0.1f);
    
    // Training
    std::cout << "\nTraining XOR network...\n";
    for (int epoch = 0; epoch < 1000; ++epoch) {
        Tensor pred = model.forward(x);
        Tensor diff = pred - y;
        Tensor loss = (diff * diff).mean();
        
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        if (epoch % 100 == 0) {
            float loss_val;
            loss.copy_to_host(&loss_val);
            std::cout << "Epoch " << std::setw(4) << epoch 
                      << " Loss: " << std::fixed << std::setprecision(6) 
                      << loss_val << std::endl;
        }
    }
    
    // Test
    std::cout << "\nResults:\n";
    {
        autograd::NoGradGuard no_grad;
        Tensor pred = model.forward(x);
        std::vector<float> results(4);
        pred.copy_to_host(results.data());
        
        std::cout << "0 XOR 0 = " << results[0] << " (expected 0)\n";
        std::cout << "0 XOR 1 = " << results[1] << " (expected 1)\n";
        std::cout << "1 XOR 0 = " << results[2] << " (expected 1)\n";
        std::cout << "1 XOR 1 = " << results[3] << " (expected 0)\n";
    }
    
    return 0;
}
```

**Build and Run:**
```bash
cd my_project
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$HOME/.local
make
./xor_demo
```

---

## 7. API Quick Reference

### Tensor Operations

| Operation | Example |
|-----------|---------|
| Addition | `c = a + b` or `c = ops::add(a, b)` |
| Subtraction | `c = a - b` or `c = ops::sub(a, b)` |
| Multiplication | `c = a * b` or `c = ops::mul(a, b)` |
| Division | `c = a / b` or `c = ops::div(a, b)` |
| Matrix multiply | `c = ops::matmul(a, b)` |
| Sum | `c = a.sum()` or `ops::sum(a)` |
| Mean | `c = a.mean()` or `ops::mean(a)` |

### In-Place Operations

| Method | Description |
|--------|-------------|
| `to_(Device)` | Move tensor to device |
| `copy_(Tensor)` | Copy data from another tensor |
| `zero_()` | Set all elements to zero |

### Module Methods

| Method | Description |
|--------|-------------|
| `forward(x)` | Forward pass |
| `parameters()` | Get all parameters |
| `to(Device)` | Move all parameters to device |
| `train(bool)` | Set training mode |
| `eval()` | Set evaluation mode |
| `zero_grad()` | Zero all gradients |

### Optimizers

| Optimizer | Constructor |
|-----------|-------------|
| SGD | `optim::SGD(params, lr, momentum, weight_decay)` |
| Adam | `optim::Adam(params, lr, beta1, beta2, eps, weight_decay)` |
| AdamW | `optim::AdamW(params, lr, beta1, beta2, eps, weight_decay)` |
| Lion | `optim::Lion(params, lr, beta1, beta2, weight_decay)` |

---

## 8. Troubleshooting

### HIP/ROCm Issues

**Error: hipcc not found**
```bash
export PATH=/opt/rocm/bin:$PATH
```

**Error: GPU device not found**
```bash
rocminfo  # Check if ROCm detects your GPU
```

### CUDA Issues

**Error: CUDA_HOME not set**
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### CMake Issues

**Error: vesper package not found**
```bash
# Ensure CMAKE_PREFIX_PATH includes install location
cmake .. -DCMAKE_PREFIX_PATH=$HOME/.local
```

**Error: HIP link flags in non-HIP project**

This is fixed in Vesper 1.0.0+. The library now exports only the necessary runtime libraries without HIP compiler flags.

### Build Issues

**Error: C++17 features not supported**
```bash
# Ensure you have a recent compiler
g++ --version  # Should be 9.0 or later
```

---

## Need Help?

- [GitHub Issues](https://github.com/danielmriley/vesper/issues)
- [Documentation](https://github.com/danielmriley/vesper/tree/main/docs)
