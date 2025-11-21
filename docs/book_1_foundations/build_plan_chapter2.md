
# Vesper Build Plan - Chapter 2: Core Utilities: Device and DType Enums

## 1. Goal

Define the fundamental enumerations for compute devices (`Device`) and data types (`DType`). These enums are the bedrock for device-agnostic operations and strongly-typed tensor data, and they need to be accessible throughout the library.

## 2. Prerequisites

- Chapter 1: Project setup is complete. The `CMakeLists.txt` and directory structure are in place.

## 3. Detailed Steps

### Step 3.1: Create the Core Header Directory

All public core headers will reside in `include/vesper/core/`.

```sh
mkdir -p include/vesper/core
```

### Step 3.2: Implement `device.h`

This header will define the `vesper::Device` enum to identify the computational backend. For now, it will distinguish between `CPU`, `CUDA`, and `HIP`.

Create `include/vesper/core/device.h`:

```cpp
// include/vesper/core/device.h
#pragma once

#include <iostream>

namespace vesper {

enum class Device {
    CPU,
    CUDA,
    HIP
};

// Utility for printing the device type
inline std::ostream& operator<<(std::ostream& os, const Device& device) {
    switch (device) {
        case Device::CPU:
            os << "CPU";
            break;
        case Device::CUDA:
            os << "CUDA";
            break;
        case Device::HIP:
            os << "HIP";
            break;
    }
    return os;
}

} // namespace vesper
```

### Step 3.3: Implement `dtype.h`

This header defines the `vesper::DType` enum, which specifies the data type of the elements within a Tensor. We will start with the most common types.

Create `include/vesper/core/dtype.h`:

```cpp
// include/vesper/core/dtype.h
#pragma once

#include <cstdint>
#include <iostream>

namespace vesper {

enum class DType {
    Float32,
    Float64,
    Float16,  // Added for LLM support
    BFloat16, // Added for LLM support
    Int32,
    Int64
};

// Utility to get the size of a DType in bytes
inline size_t GetDTypeSize(DType dtype) {
    switch (dtype) {
        case DType::Float32: return sizeof(float);
        case DType::Float64: return sizeof(double);
        case DType::Float16: return 2; // 16-bit
        case DType::BFloat16: return 2; // 16-bit
        case DType::Int32:   return sizeof(int32_t);
        case DType::Int64:   return sizeof(int64_t);
    }
    // This part should ideally be unreachable
    throw std::runtime_error("Unknown DType");
}

// Utility for printing the DType
inline std::ostream& operator<<(std::ostream& os, const DType& dtype) {
    switch (dtype) {
        case DType::Float32: os << "Float32"; break;
        case DType::Float64: os << "Float64"; break;
        case DType::Float16: os << "Float16"; break;
        case DType::BFloat16: os << "BFloat16"; break;
        case DType::Int32:   os << "Int32"; break;
        case DType::Int64:   os << "Int64"; break;
    }
    return os;
}

} // namespace vesper
```

### Step 3.4: Implement `macros.h`

This header will define essential macros for error handling and debugging, such as `VESPER_CHECK`.

Create `include/vesper/core/macros.h`:

```cpp
// include/vesper/core/macros.h
#pragma once

#include <stdexcept>
#include <string>

#define VESPER_CHECK(condition, message) \
    if (!(condition)) { \
        throw std::runtime_error(std::string("VESPER_CHECK failed: ") + (message)); \
    }

```

## 4. Code Structure Suggestions

-   **`#pragma once`**: Use this for include guards as it's simpler and more modern than traditional `#ifndef` guards.
-   **Namespace**: All code is encapsulated within the `vesper` namespace.
-   **Header-Only**: For simple types like these, a header-only implementation is perfect. No `.cpp` files are needed yet. Because no source files are added, you don't need to modify `src/CMakeLists.txt` in this chapter.
-   **Stream Operators**: Providing `operator<<` for your enums is invaluable for debugging and logging purposes.

## 5. Potential Pitfalls

-   **Forgetting the Namespace**: Failing to put these enums in the `vesper` namespace can lead to name clashes with other libraries or the user's own code.
-   **Incomplete Enums**: While we start with a few types, a real-world library would have more (`float16`, `bool`, etc.). This design is extensible, so we can add more later.

## 6. Integration and Verification

To ensure the new headers are correctly set up and accessible, we will create our first test.

### Step 6.1: Enable Testing in CMake

Modify the root `CMakeLists.txt` to include the `tests` directory. Uncomment the following lines:

```cmake
# Vesper/CMakeLists.txt

# ... (after add_subdirectory(src))

# --- 5. Add Tests and Examples ---
enable_testing()
add_subdirectory(tests)
# add_subdirectory(examples)

# ...
```

### Step 6.2: Create a `CMakeLists.txt` for Tests

This file will define a test executable that links against our `vesper` library.

Create `tests/CMakeLists.txt`:

```cmake
# Vesper/tests/CMakeLists.txt

# Define the test executable
add_executable(core_tests test_core.cpp)

# Link the test against our library
# This automatically handles include directories and library linking
target_link_libraries(core_tests PRIVATE vesper)

# Add the test to CTest
add_test(NAME CoreEnumTests COMMAND core_tests)
```

### Step 6.3: Create the Test Source File

Create a simple C++ file to include the headers and print some values. This confirms they are part of the build and compile correctly.

Create `tests/test_core.cpp`:

```cpp
// tests/test_core.cpp
#include <vesper/core/device.h>
#include <vesper/core/dtype.h>
#include <vesper/core/macros.h>
#include <iostream>
#include <cassert>

int main() {
    vesper::Device my_device = vesper::Device::HIP;
    vesper::DType my_dtype = vesper::DType::Float32;

    std::cout << "Testing Core Utilities..." << std::endl;
    std::cout << "Default Device: " << my_device << std::endl;
    std::cout << "Default DType: " << my_dtype << std::endl;
    std::cout << "Size of Float32: " << vesper::GetDTypeSize(my_dtype) << " bytes" << std::endl;
    std::cout << "Size of Float16: " << vesper::GetDTypeSize(vesper::DType::Float16) << " bytes" << std::endl;

    assert(vesper::GetDTypeSize(vesper::DType::Float32) == 4);
    assert(vesper::GetDTypeSize(vesper::DType::Int64) == 8);
    assert(vesper::GetDTypeSize(vesper::DType::Float16) == 2);
    assert(vesper::GetDTypeSize(vesper::DType::BFloat16) == 2);

    // Test VESPER_CHECK
    try {
        VESPER_CHECK(true, "This should not fail");
        // VESPER_CHECK(false, "This should fail"); // Uncomment to test failure
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Core Utilities Test Passed!" << std::endl;

    return 0;
}
```

### Step 6.4: Build and Run the Test

Navigate to your build directory, re-run CMake to detect the new files, build, and run the tests using `ctest`.

```sh
cd /path/to/vesper/build
cmake ..
make -j
ctest --verbose
```

**Expected Output:**

The CMake output should show that it found the test. The `make` command will compile the `core_tests` executable. Finally, `ctest` will run it and report a pass.

```
...
Running tests...
Test project /path/to/vesper/build
    Start 1: CoreEnumTests
1/1 Test #1: CoreEnumTests ....................   Passed    0.00 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   0.01 sec
```
The test output should include:
```
Testing Core Utilities...
Default Device: HIP
Default DType: Float32
Size of Float32: 4 bytes
Core Utilities Test Passed!
```

If the test passes, your core enums are correctly integrated and you are ready to build upon them in Chapter 3.
