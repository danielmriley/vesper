
# Vesper Build Plan - Chapter 21: Preparing for the CUDA Backend

## 1. Goal

Create the structural foundation for a future CUDA backend. This involves updating the CMake build system to detect and use the CUDA toolkit, creating stub `.cu` files for our operations, and wiring them into the C++ dispatch functions. This makes adding CUDA kernel implementations a straightforward task without further changing the library's architecture.

## 2. The CUDA/HIP Portability Model

Many HIP APIs are intentionally named and designed to be direct analogues of CUDA APIs (e.g., `hipMalloc` vs. `cudaMalloc`, `hipLaunchKernelGGL` vs. `cudaLaunchKernelGGL`). This similarity means that porting the existing HIP kernels to CUDA will be a relatively direct translation process. Our goal in this chapter is to set up the file structure and build system to accommodate these new files.

## 3. Detailed Steps

### Step 3.1: Update CMake for CUDA

Modify the root `CMakeLists.txt` to find the CUDA toolkit and enable CUDA as a project language. This allows CMake to invoke `nvcc`, the CUDA compiler, for `.cu` files.

```cmake
# Vesper/CMakeLists.txt

# --- 2. Backend Options ---
# ...
option(USE_CUDA "Enable CUDA (NVIDIA GPU) backend" OFF)

# ... (after project(...) call)

if(USE_CUDA)
    # Find the CUDA toolkit provided by NVIDIA
    find_package(CUDA 11.0 REQUIRED)
    if(CUDA_FOUND)
        message(STATUS "Found CUDA toolkit. Enabling CUDA backend.")
        enable_language(CUDA)
        
        # Add a definition for preprocessor directives in the code
        add_compile_definitions(USE_CUDA_BACKEND)
    endif()
endif()

# --- 4. Add the Library Source Code ---
# ...
# This section conditionally adds CUDA source files to the build
if(USE_CUDA)
    file(GLOB CUDA_SOURCES "src/ops/cuda/*.cu")
    target_sources(vesper PRIVATE ${CUDA_SOURCES})
    # Link against the CUDA runtime library
    target_link_libraries(vesper PUBLIC CUDA::cudart)
endif()
```

### Step 3.2: Create CUDA Stub Files

For each of our existing `.hip` files, create a corresponding `.cu` file in `src/ops/cuda/`. Each file will contain a dispatch function with a "not implemented" error.

Create `src/ops/cuda/gemm.cu`:
```sh
mkdir -p src/ops/cuda
```
```cpp
// src/ops/cuda/gemm.cu
#include <vesper/ops/gemm.h>
#include <stdexcept>

namespace vesper::ops {

// The CUDA dispatch function stub
void gemm_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& c) {
    throw std::runtime_error("GEMM not implemented for CUDA backend yet.");
}

}
```
Create similar stubs for the other operations:
- `src/ops/cuda/elementwise.cu`
- `src/ops/cuda/reduction.cu`
- `src/ops/cuda/activation.cu`
- etc.

### Step 3.3: Update C++ Dispatch Functions

Finally, add the `case Device::CUDA:` to the `switch` statement in all public-facing C++ operation functions (`matmul`, `add`, `sum`, etc.).

Example in `src/ops/gemm.cpp`:
```cpp
// src/ops/gemm.cpp

// Add a forward declaration for the new CUDA dispatch function
void gemm_cuda_dispatch(const Tensor&, const Tensor&, Tensor&);

Tensor matmul(const Tensor& a, const Tensor& b) {
    // ...
    switch (a.device()) {
        case Device::HIP:
            // ...
            break;
        case Device::CPU:
            // ...
            break;
        case Device::CUDA:
#if defined(USE_CUDA_BACKEND)
            gemm_cuda_dispatch(a_contig, b_contig, c);
#else
            throw std::runtime_error("CUDA support was not compiled.");
#endif
            break;
    }
    // ...
}
```

## 4. Verification

There is no "pass/fail" test in the traditional sense. The verification is twofold:

1.  **Compile-Time Verification**: Configure the project with `cmake .. -DUSE_CUDA=ON -DUSE_HIP=OFF` (assuming you have the CUDA toolkit installed). The entire Vesper library should compile successfully. This proves that the stubs are correctly integrated into the build system.

2.  **Run-Time Verification**: Create a simple test that allocates a tensor on `Device::CUDA` and attempts to perform an operation (e.g., `matmul`). The test should fail and throw the `std::runtime_error` from our stub function. This confirms that the dispatch logic is correctly routing to the new CUDA path.

```cpp
// In a test file
void test_cuda_stub() {
    auto a = vesper::empty({2,2}, vesper::DType::Float32, vesper::Device::CUDA);
    auto b = vesper::empty({2,2}, vesper::DType::Float32, vesper::Device::CUDA);
    try {
        auto c = vesper::ops::matmul(a, b);
        assert(false); // Should not reach here
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        assert(msg.find("not implemented for CUDA") != std::string::npos);
    }
    std::cout << "CUDA stub test passed!" << std::endl;
}
```
This confirms that the entire architecture is ready for a developer to begin porting the HIP kernels to CUDA.
