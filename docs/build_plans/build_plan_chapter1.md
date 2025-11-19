
# Vesper Build Plan - Chapter 1: Project Setup & CMake Foundation

## 1. Goal

Establish the foundational directory structure and a robust CMake build system. This initial setup is crucial for managing the project's modularity, handling different backends (HIP, CUDA, CPU), and ensuring a smooth development workflow.

## 2. Prerequisites

- A C++ compiler (e.g., g++)
- CMake (version 3.15 or later)
- AMD ROCm Toolkit installed (for HIP support)

## 3. Detailed Steps

### Step 3.1: Create the Project Directory Structure

First, lay out the core directories. This structure separates the library's source code (`vesper`), external-facing headers (`include`), tests, and build outputs.

```sh
mkdir vesper
cd vesper
mkdir src
mkdir include
mkdir tests
mkdir examples
```

Your project root should now look like this:

```
vesper/
├── include/
├── src/
├── tests/
└── examples/
```

### Step 3.2: Create the Root `CMakeLists.txt`

This is the main control file for your build system. It will define the project, set the C++ standard, handle backend options, and include the subdirectories.

Create `CMakeLists.txt` in the project root:

```cmake
# Vesper/CMakeLists.txt

cmake_minimum_required(VERSION 3.15)
project(Vesper LANGUAGES CXX C)

# --- 1. Project-wide Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- 2. Backend Options ---
# Define options to control which backend to build. Default to HIP ON.
option(USE_HIP "Enable HIP (AMD GPU) backend" ON)
option(USE_CUDA "Enable CUDA (NVIDIA GPU) backend" OFF)
option(USE_CPU "Enable CPU backend" OFF)

# --- 3. Public Include Directory ---
# This allows users linking against Vesper to find headers.
include_directories(include)

# --- 4. Add the Library Source Code ---
# The core library logic will be defined in the src/ directory.
add_subdirectory(src)

# --- 5. Add Tests and Examples (optional for now) ---
# enable_testing()
# add_subdirectory(tests)
# add_subdirectory(examples)

# --- 6. Output ---
# Inform the user about the chosen configuration.
message(STATUS "Vesper Build Configuration:")
message(STATUS "  - C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  - USE_HIP: ${USE_HIP}")
message(STATUS "  - USE_CUDA: ${USE_CUDA}")
message(STATUS "  - USE_CPU: ${USE_CPU}")

```

### Step 3.3: Create the `src` Directory `CMakeLists.txt`

This file will be responsible for compiling the library source files and linking against the necessary backend libraries (starting with HIP).

Create `src/CMakeLists.txt`:

```cmake
# Vesper/src/CMakeLists.txt

# --- 1. Define the library target ---
# We will build Vesper as a static library for simplicity.
add_library(vesper STATIC)

# --- 2. Find and Link Backend Libraries ---
if(USE_HIP)
    # Find the HIP package provided by ROCm.
    find_package(amdhip-runtime REQUIRED)
    if(amdhip-runtime_FOUND)
        message(STATUS "Found HIP runtime. Enabling HIP support.")
        # Add a definition to use in preprocessor directives.
        target_compile_definitions(vesper PUBLIC USE_HIP_BACKEND)
        # Link the library to the HIP runtime.
        target_link_libraries(vesper PUBLIC amdhip::amdhip-runtime)
    endif()
endif()

if(USE_CUDA)
    # Stub for CUDA
    message(WARNING "CUDA backend is not yet implemented.")
    target_compile_definitions(vesper PUBLIC USE_CUDA_BACKEND)
    # In the future, you would find and link CUDA here.
    # find_package(CUDA REQUIRED)
    # target_link_libraries(vesper PUBLIC CUDA::cudart)
endif()

if(USE_CPU)
    # Stub for CPU
    message(STATUS "CPU backend stub enabled.")
    target_compile_definitions(vesper PUBLIC USE_CPU_BACKEND)
endif()

# --- 3. Add Source Files ---
# As we create files, we will add them here.
# For now, we can create a placeholder.
# target_sources(vesper PRIVATE
#     core/device.cpp
# )

# --- 4. Expose Public Headers ---
# This makes headers in `include/vesper` available to the `vesper` target
# and to any other target that links against it.
target_include_directories(vesper
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
)
```

## 4. Code Structure Suggestions

-   **Public Headers**: All headers that should be accessible to a user of your library must go into `include/vesper/`. For example: `include/vesper/tensor.h`.
-   **Private Headers/Source Files**: All implementation files (`.cpp`, `.hip`, `.cu`) and internal headers should go into `src/`. We will create subdirectories here that mirror the public ones (e.g., `src/core`, `src/ops`).

## 5. Potential Pitfalls

-   **ROCm Not Found**: If CMake reports that `amdhip-runtime` is not found, it means the ROCm installation path is not in CMake's search path. You may need to set the `CMAKE_PREFIX_PATH` environment variable:
    ```sh
    export CMAKE_PREFIX_PATH=/opt/rocm
    ```
-   **CMake Version**: Using an older CMake version might cause issues with `find_package` or other modern commands. Ensure you have at least version 3.15.
-   **Static vs. Shared Library**: We start with a `STATIC` library for simplicity. Changing to `SHARED` later might require handling symbol visibility (e.g., `__declspec(dllexport)` on Windows).

## 6. Integration and Verification

This chapter stands alone but is the foundation for all subsequent chapters.

**Milestone:** Your project should configure successfully without any source files.

1.  Navigate to your project root.
2.  Create a build directory.
3.  Run CMake.

```sh
cd /path/to/vesper
mkdir build
cd build
cmake ..
```

**Expected Output:**

You should see the status messages from your `CMakeLists.txt`, confirming that HIP is enabled and the stubs for CPU/CUDA are noted.

```
-- Vesper Build Configuration:
--   - C++ Standard: 17
--   - USE_HIP: ON
--   - USE_CUDA: OFF
--   - USE_CPU: OFF
-- Found HIP runtime. Enabling HIP support.
-- CPU backend stub enabled.
-- Configuring done
-- Generating done
-- Build files have been written to: /path/to/vesper/build
```

If you see this, your build system is correctly set up and ready for the first code implementation in Chapter 2.
