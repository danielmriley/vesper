# Vesper

A lightweight, pure C++ deep learning library inspired by PyTorch, designed for high-performance LLM workloads.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Features

- **Pure C++17** - No Python dependencies, fully native C++
- **Zero External Dependencies** - No BLAS, Eigen, or other math libraries; all kernels implemented from scratch
- **HIP/ROCm First** - Primary support for AMD GPUs with optimized kernels
- **CUDA Support** - Full NVIDIA GPU support (stubs ready, kernels implemented)
- **PyTorch-like API** - Familiar tensor operations, autograd, and neural network modules
- **LLM-Ready** - Transformer components, RoPE, KV-Cache, SwiGLU, and more

## Quick Start

### Prerequisites

- **C++17 compatible compiler** (GCC 9+, Clang 10+)
- **CMake 3.15+**
- **For HIP/AMD GPUs**: ROCm 5.0+ with hipcc
- **For CUDA/NVIDIA GPUs**: CUDA Toolkit 11.0+

### Installation

#### Option 1: Install System-Wide

```bash
git clone https://github.com/danielmriley/vesper.git
cd vesper

# For AMD GPUs (HIP/ROCm)
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
         -DUSE_HIP=ON \
         -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
make -j$(nproc)
sudo make install

# For NVIDIA GPUs (CUDA)
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
         -DUSE_CUDA=ON \
         -DUSE_HIP=OFF
make -j$(nproc)
sudo make install
```

#### Option 2: Install to User Directory

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local \
         -DUSE_HIP=ON \
         -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
make -j$(nproc)
make install

# Add to your shell profile:
export CMAKE_PREFIX_PATH=$HOME/.local:$CMAKE_PREFIX_PATH
```

#### Option 3: Use as Subdirectory

```bash
# In your project
git submodule add https://github.com/danielmriley/vesper.git extern/vesper
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_HIP` | `ON` | Enable HIP (AMD GPU) backend |
| `USE_CUDA` | `OFF` | Enable CUDA (NVIDIA GPU) backend |
| `USE_CPU` | `OFF` | Enable CPU backend |
| `VESPER_BUILD_TESTS` | `ON` | Build unit tests |

## Usage

### Using find_package (Recommended)

After installing Vesper, use it in your project:

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(vesper 1.0 REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp vesper::vesper)
```

### Using as Subdirectory

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Configure Vesper
set(USE_HIP ON CACHE BOOL "")
set(VESPER_BUILD_TESTS OFF CACHE BOOL "")
add_subdirectory(extern/vesper)

add_executable(myapp main.cpp)
target_link_libraries(myapp vesper)
```

### Example: Training a Neural Network

```cpp
#include <vesper/vesper.h>
#include <iostream>

using namespace vesper;

// Define a simple MLP
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

int main() {
    // Select device
#if defined(USE_HIP_BACKEND)
    Device device = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    Device device = Device::CUDA;
#else
    Device device = Device::CPU;
#endif

    // Create model and optimizer
    SimpleMLP model(device);
    optim::Adam optimizer(model.parameters(), 0.001f);
    
    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        // Create batch
        Tensor x = randn({32, 784}, DType::Float32, device, true);
        Tensor y = randn({32, 10}, DType::Float32, device);
        
        // Forward pass
        Tensor pred = model.forward(x);
        Tensor loss = nn::functional::mse_loss(pred, y);
        
        // Backward pass
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        if (epoch % 10 == 0) {
            float loss_val;
            loss.copy_to_host(&loss_val);
            std::cout << "Epoch " << epoch << " Loss: " << loss_val << std::endl;
        }
    }
    
    return 0;
}
```

## Components

### Core
- **Tensor** - N-dimensional array with autograd support
- **Device** - CPU, HIP (AMD), CUDA (NVIDIA)
- **DType** - Float32, Float64, Int32, Int64

### Neural Network Modules
- **Linear** - Fully connected layer
- **Embedding** - Lookup table embeddings
- **Conv2d** - 2D convolution
- **LayerNorm / RMSNorm** - Normalization layers
- **Dropout** - Regularization

### Transformer Components
- **MultiHeadAttention / GQAAttention** - Attention mechanisms
- **RoPE** - Rotary Position Embeddings
- **SwiGLU** - Gated activation
- **KVCache** - Efficient key-value caching for inference
- **TransformerBlock / Transformer** - Full transformer architecture

### Optimizers
- **SGD** - Stochastic Gradient Descent with momentum
- **Adam / AdamW** - Adaptive learning rate
- **Lion** - Memory-efficient optimizer

### I/O
- **SafeTensors** - Load/save model weights
- **StateDict** - PyTorch-compatible serialization

## Documentation

- [Usage Guide](docs/usage_guide.md) - Detailed integration instructions
- [API Reference](docs/book_1_foundations/) - Core concepts and tutorials
- [Build Plans](docs/blueprint/) - Architecture and design decisions

## Testing

```bash
cd build
cmake .. -DUSE_HIP=ON -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
make -j$(nproc)
ctest --output-on-failure
```

## Project Structure

```
vesper/
├── include/vesper/     # Public headers
│   ├── core/           # Tensor, Device, DType
│   ├── autograd/       # Automatic differentiation
│   ├── nn/             # Neural network modules
│   ├── optim/          # Optimizers
│   ├── ops/            # Low-level operations
│   ├── models/         # Pre-built model architectures
│   ├── generation/     # Text generation utilities
│   └── io/             # Serialization
├── src/                # Implementation
│   ├── ops/hip/        # HIP GPU kernels
│   ├── ops/cuda/       # CUDA GPU kernels
│   └── ops/cpu/        # CPU implementations
├── tests/              # Unit tests
└── docs/               # Documentation
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) before submitting a PR.

## Acknowledgments

Inspired by PyTorch, designed for the AMD GPU ecosystem.
