### Project Overview: Vesper Library

Vesper is a lightweight, pure C++ deep learning library inspired by PyTorch/LibTorch, designed for seamless integration into C++ projects. The primary goals are:
- **Independence**: No Python dependencies; fully native C++ for compilation and linking.
- **Modularity**: Each component (e.g., tensor operations, autograd, neural network modules) is encapsulated in dedicated files/classes to promote readability, maintainability, and reusability.
- **Comprehensive Coverage**: Aim to replicate core PyTorch functionalities, starting with essentials (tensors, autograd, basic NN modules) and expanding to advanced features (optimizers, data loaders, distributed training) in phases. Full coverage of PyTorch is ambitious and may take iterations, so prioritize LLM-relevant features like tensor ops, linear layers, and GPU acceleration.
- **GPU Compatibility**: Support for HIP (AMD ROCm) first, with stubs for future CUDA (NVIDIA) and CPU implementations. This allows building for specific backends without runtime overhead. All performance-critical operations, like GEMM (General Matrix Multiply), will be implemented from scratch with industry-grade algorithms tailored for each backend—no reliance on external BLAS or similar libraries.
- **Minimalism**: Rely solely on standard C++ (C++17 or later) for core logic. No external libraries whatsoever (e.g., no Boost, GMP, Eigen, or BLAS). All matrix/tensor math, including GEMM, will be implemented manually using std::vector, std::array, raw pointers, and custom optimized algorithms for efficiency. For GEMM specifically: Develop your own high-performance versions using techniques like loop tiling, cache optimization, and parallelization (std::thread for CPU), with custom kernels for HIP/CUDA.
- **Use Case Focus**: Easy linking to your LLM project—e.g., compile Vesper as a static/shared library and include headers/link in your build.
- **Rationale for Custom Implementations**: Implementing everything yourself (e.g., GEMM, optimizers) ensures full control, portability, and educational value. It avoids dependency bloat and allows optimization for your specific LLM workloads. While this increases initial effort (e.g., writing efficient GEMM from scratch can be complex), it aligns with your goal of a self-contained library. Start simple (naive implementations) and iterate to industry-grade (e.g., Strassen's algorithm variants, fused operations).

Estimated Scope: PyTorch has ~100k LoC; Vesper could start at 5-10k LoC for MVP (Minimum Viable Product) and grow. This is a significant project, so the plan emphasizes iterative development, beginning with HIP for GPU acceleration.

### Dependencies and Requirements
- **C++ Standard**: Use C++17 (for std::variant, structured bindings) or C++20 (for concepts, ranges) to enable modern, expressive code without extras.
- **Standard Libraries Only**: Core: `<vector>`, `<array>`, `<memory>`, `<functional>`, `<algorithm>`, `<thread>`, `<mutex>`, `<random>`, `<cmath>`, `<iostream>`. For parallelism: `<execution>` (C++17 parallel algorithms). No I/O beyond basics unless needed for data loading. All custom algorithms (e.g., GEMM) will use these for optimizations like multi-threading.
- **External Libraries**: None. All implementations are pure C++.
- **GPU SDKs**: 
  - AMD: ROCm (for hipcc compiler) as the starting point.
  - NVIDIA: CUDA Toolkit (for nvcc compiler)—stubs only initially.
  - No runtime dependencies; users must have the SDK installed for GPU builds.
- **Build Tools**: Use CMake for cross-platform building, with options to enable HIP (primary), CUDA (future), or CPU (future). No dependencies like Conan or vcpkg.

### High-Level Architecture
Vesper will mirror PyTorch's structure for familiarity but simplify for C++:
- **Namespace**: All in `namespace vesper { ... }` to avoid conflicts.
- **Core Abstractions**:
  - **Device Agnosticism**: Use an enum `Device { CPU, CUDA, HIP }` and template specializations or runtime dispatch (via std::variant or polymorphism) for operations. Preprocessor directives (`#ifdef __HIP__` for HIP-specific code initially; similar for others later).
  - **Tensor**: Central data structure, similar to `torch::Tensor`. Backed by a raw buffer (void* or typed) with shape, dtype, device.
  - **Modularity**: One header/source pair per major component (e.g., `tensor.h/cpp`, `autograd.h/cpp`). Use forward declarations to minimize includes.
- **Backend Handling**:
  - HIP: Full custom implementations first, including GEMM kernels using HIP syntax and optimizations (e.g., shared memory, coalesced access).
  - CUDA/CPU: Initial stubs (e.g., throw exceptions or fallback to naive serial loops) to allow compilation; expand later with equivalent custom GEMM (e.g., CUDA kernels via cuBLAS-like manual coding, CPU with SIMD via intrinsics if standard allows, or threaded loops).
  - Custom GEMM: Dedicated files like `gemm_hip.hip` for HIP kernel, with interfaces in `ops/gemm.h/cpp`. Aim for industry-grade: Block tiling, register blocking, double-buffering for HIP; similar patterns for others.
  - Build Variants: CMake targets like `vesper_hip` (primary), `vesper_cuda_stub`, `vesper_cpu_stub`.

Key Modules (inspired by PyTorch packages):
1. **Core (vesper/core)**: Tensors, dtypes, devices.
2. **Autograd (vesper/autograd)**: Automatic differentiation.
3. **NN (vesper/nn)**: Modules and functionals.
4. **Optim (vesper/optim)**: Optimizers (custom implementations, e.g., fused updates for efficiency).
5. **Data (vesper/data)**: Datasets and loaders.
6. **Utils (vesper/utils)**: Helpers like initializers, serialization.
7. **Distributed (vesper/distributed)**: Optional, for multi-GPU (advanced phase).
8. **Ops (vesper/ops)**: Low-level operations like GEMM, elementwise—where custom algorithms live.

### Detailed Component Breakdown
Organize into subdirectories with one class/file per functionality. Each file should be <500 LoC for manageability. Start all GPU ops with HIP kernels; add stubs for CUDA/CPU.

- **vesper/core/tensor.h/cpp**: `class Tensor` – Handles creation (e.g., `Tensor::zeros(shape, dtype, device)`), ops (add, mul, matmul via custom GEMM), indexing, reshaping. Dtypes: float32/64, int32/64 (use std::variant for storage).
- **vesper/core/device.h/cpp**: `enum Device`, context managers (e.g., `DeviceGuard` RAII class).
- **vesper/core/dtype.h/cpp**: Type traits and conversions.
- **vesper/core/storage.h/cpp**: Underlying buffer management (aligned malloc for CPU stubs, hipMalloc for HIP).
- **vesper/autograd/autograd.h/cpp**: `class Variable : public Tensor` (with gradient hook), `backward()` function. Implement graph via shared_ptr to nodes (e.g., `struct Node { std::function<void()> backward_fn; }`).
- **vesper/autograd/functions.h/cpp**: Specific backward funcs (e.g., AddBackward, MatmulBackward) as lambdas or classes.
- **vesper/nn/module.h/cpp**: Base `class Module` with `forward()`, parameter registration (std::map<std::string, Tensor>).
- **vesper/nn/linear.h/cpp**: `class Linear : public Module` – Weights, bias, forward (matmul via custom GEMM).
- **vesper/nn/conv.h/cpp**: `class Conv2d : public Module` – Convolution ops (implement im2col or direct with custom GEMM; HIP kernels).
- **vesper/nn/activation.h/cpp**: ReLU, Sigmoid, etc., as functionals (stateless) and modules.
- **vesper/nn/loss.h/cpp**: MSE, CrossEntropy (with softmax via custom ops).
- **vesper/optim/optimizer.h/cpp**: Base `class Optimizer` with `step()`—custom industry-grade impls (e.g., fused Adam with custom loops).
- **vesper/optim/sgd.h/cpp**: SGD with momentum (manual updates).
- **vesper/optim/adam.h/cpp**: Adam (track moments via Tensors; optimize with custom fused kernels).
- **vesper/data/dataset.h/cpp**: Abstract `class Dataset` with `get_item(idx)`.
- **vesper/data/dataloader.h/cpp**: `class DataLoader` – Batch sampling, multi-threaded (std::thread).
- **vesper/utils/initializers.h/cpp**: Kaiming, Xavier (using std::random).
- **vesper/utils/serialize.h/cpp**: Save/load models (binary streams via <fstream>).
- **vesper/ops/gemm.h/cpp**: Interface for custom GEMM; dispatches to backend-specific impls.
- **vesper/ops/gemm_hip.hip**: HIP kernel for GEMM (industry-grade: tiled, optimized for AMD GPUs).
- **vesper/ops/gemm_cuda.cu**: Stub initially (e.g., #error "CUDA not implemented yet"); later, custom CUDA kernel.
- **vesper/ops/gemm_cpu.cpp**: Stub initially; later, custom CPU GEMM (e.g., tiled loops with std::thread parallelization).
- **vesper/gpu/hip_kernels.hip**: Other HIP-specific kernels (e.g., elementwise, reductions). Stubs in cuda_kernels.cu and cpu equivalents.

For full PyTorch coverage: Start with above (covers ~70% for basic ML). Later add: RNN/LSTM, vision (transforms), audio, etc. Prioritize LLM needs: Attention, Embedding, Transformer modules in nn, all using custom ops.

### Implementation Phases
Break into iterative milestones, starting with HIP:

1. **Phase 1: Core Tensor with HIP (1-2 weeks)**  
   - Implement Tensor with basic ops (arithmetic, reductions). Focus on HIP for matmul via custom GEMM kernel.  
   - Stubs for CUDA/CPU: Throw or naive serial fallback.  
   - Test: Unit tests with assertions (simple main() harness) on AMD hardware.

2. **Phase 2: Autograd with HIP (2-3 weeks)**  
   - Add Variable, backward for add/mul/matmul using HIP kernels.  
   - Example: Train simple linear regression on HIP.

3. **Phase 3: NN Modules with HIP (2-4 weeks)**  
   - Build Linear, Conv, activations, losses—all leveraging custom HIP GEMM.  
   - Implement forward/backward for each.

4. **Phase 4: Custom Optimizers and Training Loop (1-2 weeks)**  
   - SGD/Adam with manual, fused implementations (no external deps).  
   - Utils for training (e.g., zero_grad()).

5. **Phase 5: Expand to CUDA and CPU (3-4 weeks)**  
   - Flesh out stubs: Custom GEMM kernels for CUDA (port from HIP), optimized CPU (tiled/threaded).  
   - Test cross-backend: Ensure Tensor.to(Device::CUDA) works once implemented.

6. **Phase 6: Data Handling and Extras (2-3 weeks)**  
   - Datasets (e.g., from CSV via <fstream>), loaders.  
   - Serialization, initializers.

7. **Phase 7: Expansion and Polish (Ongoing)**  
   - Add more modules (e.g., Transformer for LLMs).  
   - Performance tuning: Profile with ROCm tools (rocprof), then NVIDIA/CUDA equivalents.  
   - Full PyTorch parity: Audit against torch docs, implement missing funcs iteratively.

Total Timeline: 2-4 months for HIP MVP, 6+ for multi-backend comprehensive.

### Build and Integration
- **CMake Setup**: Root CMakeLists.txt with subdirs. Options: `-DUSE_HIP=ON` (default), `-DUSE_CUDA=OFF` (stub), `-DUSE_CPU=OFF` (stub). Link to hip libs.
- **Linking to Your LLM Project**: Build Vesper as libvesper.a/so. In your project: `#include <vesper/tensor.h>`, link `-lvesper`.
- **Testing**: Write a tests/ dir with mains (e.g., test_tensor.cpp). Later, add benchmarks comparing to PyTorch.
- **Documentation**: Use Doxygen for API docs. README with build instructions, examples (e.g., MNIST trainer on HIP).

### Potential Challenges and Mitigations
- **Performance**: Custom GEMM requires expertise—start naive, optimize iteratively (e.g., study open-source like CUTLASS for inspiration, but implement yourself).
- **Complexity**: Autograd with custom kernels—ensure graph handles device transfers.
- **Debugging HIP**: Use hip-memcheck/rocgdb.
- **Porting**: HIP to CUDA is similar; use macros for shared code.
- **Scope Creep**: Stick to phases; use Git for versioning.

