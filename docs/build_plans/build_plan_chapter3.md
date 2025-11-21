
# Vesper Build Plan - Chapter 3: Abstracted Memory: The Storage Class

## 1. Goal

Implement the `vesper::Storage` class, a low-level container that abstracts the allocation, deallocation, and management of raw memory buffers across different compute devices (HIP, CPU, etc.). This class is the foundation upon which the `Tensor` will be built.

## 2. Prerequisites

- Chapter 1: Project and CMake setup is complete.
- Chapter 2: `Device` and `DType` enums are defined.

## 3. Detailed Steps

### Step 3.1: Create `storage.h`

This public header defines the interface for the `Storage` class. It will manage a raw pointer but use RAII principles to ensure memory is automatically released. We will make it movable but not copyable to prevent accidental and expensive deep copies.

Create `include/vesper/core/storage.h`:

```cpp
// include/vesper/core/storage.h
#pragma once

#include <vesper/core/device.h>
#include <memory>

namespace vesper {

class Storage {
public:
    // Constructor: Allocates a buffer of a given size on a specific device.
    Storage(Device device, size_t size_bytes);

    // Destructor: Frees the allocated memory.
    ~Storage();

    // --- Rule of 5 ---
    // Disable copying
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    // Enable moving
    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept;

    // --- Accessors ---
    void* data() { return data_ptr; }
    const void* data() const { return data_ptr; }
    Device device() const { return device_; }
    size_t size() const { return size_bytes_; }

private:
    void* data_ptr = nullptr;
    size_t size_bytes_ = 0;
    Device device_;
};

} // namespace vesper
```

### Step 3.2: Create `storage.cpp`

This is the implementation file where the actual memory allocation/deallocation logic lives. We will use preprocessor directives (`#if USE_HIP_BACKEND`) to conditionally compile backend-specific code.

Create `src/core/` directory first, and then the file inside.
```sh
mkdir -p src/core
```
Create `src/core/storage.cpp`:

```cpp
// src/core/storage.cpp
#include <vesper/core/storage.h>
#include <stdexcept>
#include <utility>

#if USE_HIP_BACKEND
#include <hip/hip_runtime.h>
#endif

namespace vesper {

Storage::Storage(Device device, size_t size_bytes)
    : device_(device), size_bytes_(size_bytes), data_ptr(nullptr) {
    if (size_bytes_ == 0) {
        return;
    }

    switch (device_) {
        case Device::HIP: {
#if USE_HIP_BACKEND
            hipError_t status = hipMalloc(&data_ptr, size_bytes_);
            if (status != hipSuccess) {
                throw std::runtime_error("Failed to allocate memory on HIP device.");
            }
#else
            throw std::runtime_error("HIP backend not enabled during build.");
#endif
            break;
        }
        case Device::CPU: {
            // Use aligned allocation for CPU for potential SIMD optimizations later
            data_ptr = new char[size_bytes_];
            if (!data_ptr) {
                throw std::runtime_error("Failed to allocate memory on CPU.");
            }
            break;
        }
        case Device::CUDA: {
            throw std::runtime_error("CUDA backend not yet implemented.");
        }
    }
}

Storage::~Storage() {
    if (data_ptr == nullptr) {
        return;
    }

    switch (device_) {
        case Device::HIP: {
#if USE_HIP_BACKEND
            hipFree(data_ptr);
#endif
            break;
        }
        case Device::CPU: {
            delete[] static_cast<char*>(data_ptr);
            break;
        }
        case Device::CUDA: {
            // No-op for now
            break;
        }
    }
}

Storage::Storage(Storage&& other) noexcept
    : data_ptr(other.data_ptr),
      size_bytes_(other.size_bytes_),
      device_(other.device_) {
    // Leave the moved-from object in a valid, destructible state
    other.data_ptr = nullptr;
    other.size_bytes_ = 0;
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        // Free our own resource first
        this->~Storage();

        // Pilfer the other's resources
        data_ptr = other.data_ptr;
        size_bytes_ = other.size_bytes_;
        device_ = other.device_;

        // Reset the other object
        other.data_ptr = nullptr;
        other.size_bytes_ = 0;
    }
    return *this;
}

} // namespace vesper
```

### Step 3.3: Update `src/CMakeLists.txt`

Add the new `storage.cpp` file to the `vesper` library target.

```cmake
# Vesper/src/CMakeLists.txt
# ... (after add_library)

# --- 3. Add Source Files ---
# As we create files, we will add them here.
target_sources(vesper PRIVATE
    core/storage.cpp
)

# ... (rest of the file)
```

## 4. Code Structure Suggestions

-   **RAII (Resource Acquisition Is Initialization)**: The constructor acquires the memory (`hipMalloc`/`new`) and the destructor releases it (`hipFree`/`delete[]`). This is fundamental to avoiding memory leaks in C++.
-   **Rule of 5**: By deleting copy semantics and implementing move semantics, we prevent expensive copies and clearly define ownership transfer. Moving a `Storage` object is a cheap operation that just transfers the pointer.
-   **Preprocessor Guards**: The `#if USE_HIP_BACKEND` guard ensures that the code requiring the HIP API is only compiled when HIP is enabled via the CMake option. This prevents build errors for users without the ROCm toolkit.

## 5. Potential Pitfalls

-   **Memory Allocation Overhead**: Currently, `Storage` calls `hipMalloc`/`free` directly. In high-performance deep learning, this is too slow. In future chapters, we should implement a **Caching Allocator** (memory pool) to reuse allocations. The `Storage` interface is designed to allow this swap later without changing the public API.
-   **Incorrect `delete`**: Using `delete` instead of `delete[]` for memory allocated with `new char[]` is undefined behavior.
-   **Move Assignment Self-Assignment**: The `if (this != &other)` check in the move assignment operator is crucial to prevent an object from destructing its own data before moving it.
-   **Error Handling**: The `hipMalloc` call can fail. We throw a `std::runtime_error`, which is a reasonable default. More advanced error handling could be added later.

## 6. Integration and Verification

We will add a new test to verify that the `Storage` class works as expected for the HIP backend.

### Step 6.1: Create `tests/test_storage.cpp`

This test will:
1.  Attempt to allocate a small buffer on the HIP device.
2.  Check that the pointer is not null.
3.  Verify the `device()` and `size()` accessors.
4.  Test the move constructor to ensure ownership is transferred correctly.

Create `tests/test_storage.cpp`:
```cpp
// tests/test_storage.cpp
#include <vesper/core/storage.h>
#include <iostream>
#include <cassert>

void test_hip_allocation() {
#if USE_HIP_BACKEND
    std::cout << "Testing HIP Storage allocation..." << std::endl;
    const size_t bytes = 1024;
    vesper::Storage storage(vesper::Device::HIP, bytes);

    assert(storage.data() != nullptr);
    assert(storage.device() == vesper::Device::HIP);
    assert(storage.size() == bytes);

    // Test move semantics
    vesper::Storage moved_storage(std::move(storage));
    assert(storage.data() == nullptr); // Original is now empty
    assert(storage.size() == 0);
    assert(moved_storage.data() != nullptr);
    assert(moved_storage.device() == vesper::Device::HIP);
    assert(moved_storage.size() == bytes);

    std::cout << "HIP Storage Test Passed!" << std::endl;
#else
    std::cout << "Skipping HIP Storage test (backend disabled)." << std::endl;
#endif
}

int main() {
    test_hip_allocation();
    return 0;
}
```

### Step 6.2: Add the New Test to `tests/CMakeLists.txt`

```cmake
# Vesper/tests/CMakeLists.txt

# Previous test
add_executable(core_tests test_core.cpp)
target_link_libraries(core_tests PRIVATE vesper)
add_test(NAME CoreEnumTests COMMAND core_tests)

# New storage test
add_executable(storage_tests test_storage.cpp)
target_link_libraries(storage_tests PRIVATE vesper)
add_test(NAME StorageTests COMMAND storage_tests)
```

### Step 6.3: Build and Run

Re-run CMake, build, and run `ctest`.

```sh
cd /path/to/vesper/build
cmake ..
make -j
ctest --verbose
```

**Expected Output:**

CTest should now run two tests. The `StorageTests` should run and pass, indicating that HIP memory was successfully allocated and managed.

```
Running tests...
Test project /path/to/vesper/build
    Start 1: CoreEnumTests
1/2 Test #1: CoreEnumTests ....................   Passed    0.00 sec
    Start 2: StorageTests
2/2 Test #2: StorageTests .....................   Passed    0.01 sec

100% tests passed, 0 tests failed out of 2
```
The output of the test executable will include:
```
Testing HIP Storage allocation...
HIP Storage Test Passed!
```

With a working `Storage` class, you are now ready to build the main `Tensor` data structure on top of it.
