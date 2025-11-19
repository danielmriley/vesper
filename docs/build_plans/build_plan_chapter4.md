
# Vesper Build Plan - Chapter 4: The Tensor: Core Data Structure

## 1. Goal

Implement the `vesper::Tensor` class, the central user-facing data structure in the library. A Tensor combines a `Storage` backend with metadata (shape, strides, data type) to represent a multi-dimensional array. This chapter focuses on the structure and basic properties of the Tensor, not its mathematical operations.

## 2. Prerequisites

- Chapter 2: `Device` and `DType` are defined.
- Chapter 3: The `Storage` class for memory management is implemented.

## 3. Detailed Steps

### Step 3.1: Define the `Tensor` Class Interface

The `Tensor` will hold a *shared pointer* to a `Storage` object. This is critical for enabling efficient views and slices later, as multiple `Tensor` objects can point to the same underlying memory block without copying it.

Create `include/vesper/core/tensor.h`:

```cpp
// include/vesper/core/tensor.h
#pragma once

#include <vesper/core/storage.h>
#include <vesper/core/dtype.h>
#include <vector>
#include <memory>
#include <numeric>

namespace vesper {

class Tensor {
public:
    // --- Accessors ---
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    Device device() const { return storage_->device(); }
    size_t offset() const { return offset_; }
    size_t numel() const;

    bool is_contiguous() const;

    // Returns a typed pointer to the start of the tensor's data (respecting offset)
    template <typename T>
    T* data_ptr() {
        return static_cast<T*>(storage_->data()) + offset_;
    }

    template <typename T>
    const T* data_ptr() const {
        return static_cast<const T*>(storage_->data()) + offset_;
    }

private:
    // Private constructor to be used by factory functions
    Tensor(std::shared_ptr<Storage> storage,
           DType dtype,
           std::vector<int64_t> shape,
           std::vector<int64_t> strides,
           size_t offset = 0);

    // Grant factory functions access to the private constructor
    friend Tensor empty(const std::vector<int64_t>& shape, DType dtype, Device device);

    std::shared_ptr<Storage> storage_;
    DType dtype_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    size_t offset_; // in number of elements, not bytes
};

} // namespace vesper
```

### Step 3.2: Implement Tensor Methods and Factory Functions

We will place the implementation of simple methods in the header for now, and create a new set of files for factory functions, which are responsible for constructing tensors.

First, implement `numel` and `is_contiguous` in `include/vesper/core/tensor.h`:

```cpp
// Add to public section of Tensor class in include/vesper/core/tensor.h

inline size_t Tensor::numel() const {
    if (shape_.empty()) {
        return 1; // Scalar tensor
    }
    return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<int64_t>());
}

inline bool Tensor::is_contiguous() const {
    // A tensor is contiguous if its strides match the standard layout
    int64_t current_stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (strides_[i] != current_stride) {
            return false;
        }
        current_stride *= shape_[i];
    }
    return true;
}
```

Next, create the public factory header `include/vesper/core/factories.h`:

```cpp
// include/vesper/core/factories.h
#pragma once

#include <vesper/core/tensor.h>

namespace vesper {

// Creates a tensor with uninitialized data
Tensor empty(const std::vector<int64_t>& shape, DType dtype, Device device);

// Creates a tensor filled with zeros (to be implemented later)
// Tensor zeros(const std::vector<int64_t>& shape, DType dtype, Device device);

} // namespace vesper
```

Now, create the implementation file `src/core/tensor.cpp`:

```cpp
// src/core/tensor.cpp
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>

namespace vesper {

// --- Private Constructor Implementation ---
Tensor::Tensor(std::shared_ptr<Storage> storage,
               DType dtype,
               std::vector<int64_t> shape,
               std::vector<int64_t> strides,
               size_t offset)
    : storage_(std::move(storage)),
      dtype_(dtype),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset) {}


// --- Factory Implementation ---
std::vector<int64_t> calculate_contiguous_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) {
        return strides;
    }
    int64_t current_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = current_stride;
        current_stride *= shape[i];
    }
    return strides;
}

Tensor empty(const std::vector<int64_t>& shape, DType dtype, Device device) {
    int64_t num_elements = 1;
    for(const auto& dim : shape) {
        num_elements *= dim;
    }

    size_t size_bytes = num_elements * GetDTypeSize(dtype);
    auto storage = std::make_shared<Storage>(device, size_bytes);
    auto strides = calculate_contiguous_strides(shape);

    return Tensor(std::move(storage), dtype, shape, strides, 0);
}

} // namespace vesper
```

### Step 3.3: Update `src/CMakeLists.txt`

Add `tensor.cpp` to your library's sources.

```cmake
# Vesper/src/CMakeLists.txt
# ...

target_sources(vesper PRIVATE
    core/storage.cpp
    core/tensor.cpp  # Add this line
)

# ...
```

## 4. Code Structure Suggestions

-   **`shared_ptr<Storage>`**: This is the most important design choice in this chapter. It allows Tensors to be cheap to copy and move, while also enabling multiple Tensors (e.g., views of a larger Tensor) to share the same underlying memory allocation safely.
-   **Private Constructor**: Making the constructor `private` and using friend factory functions (`empty`, `zeros`, etc.) is a powerful pattern. It guides users to create tensors in a safe, standard way and prevents inconsistent states.
-   **Contiguous Strides**: The concept of strides is key to a versatile tensor library. The `calculate_contiguous_strides` helper function establishes the default memory layout. A tensor where this layout holds true is "contiguous". Non-contiguous tensors (views, transpositions) will have different strides but can share the same storage.

## 5. Potential Pitfalls

-   **Integer Overflow**: When calculating `numel` or byte sizes, the result can exceed the capacity of a 32-bit integer. Using `int64_t` for dimensions and `size_t` for sizes is good practice.
-   **Shallow vs. Deep Copy**: A user might expect `Tensor b = a;` to create a deep copy. Our implementation makes this a shallow copy (both `a` and `b` point to the same data). This is the standard behavior in libraries like PyTorch, but it must be documented. We will add a `clone()` method later for explicit deep copies.
-   **Pointer Arithmetic**: The `data_ptr()` method relies on pointer arithmetic. Note that `static_cast<T*>(storage_->data()) + offset_` works correctly because `offset_` is measured in *number of elements*, not bytes.

## 6. Integration and Verification

We will write a test to validate the creation and properties of a `Tensor`.

### Step 6.1: Create `tests/test_tensor.cpp`

This test will create a tensor using `vesper::empty` and verify its metadata.

Create `tests/test_tensor.cpp`:
```cpp
// tests/test_tensor.cpp
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>

void test_tensor_creation() {
#if USE_HIP_BACKEND
    std::cout << "Testing Tensor creation..." << std::endl;
    
    const std::vector<int64_t> shape = {2, 3, 4};
    const vesper::DType dtype = vesper::DType::Float32;
    const vesper::Device device = vesper::Device::HIP;

    vesper::Tensor tensor = vesper::empty(shape, dtype, device);

    assert(tensor.shape() == shape);
    assert(tensor.dtype() == dtype);
    assert(tensor.device() == device);
    assert(tensor.numel() == 24);
    assert(tensor.offset() == 0);
    assert(tensor.is_contiguous());

    // Check strides: (3*4, 4, 1) -> (12, 4, 1)
    const auto& strides = tensor.strides();
    assert(strides[0] == 12);
    assert(strides[1] == 4);
    assert(strides[2] == 1);

    // Test data pointer (just check it's not null)
    float* data = tensor.data_ptr<float>();
    assert(data != nullptr);

    std::cout << "Tensor creation test passed!" << std::endl;
#else
    std::cout << "Skipping Tensor creation test (HIP backend disabled)." << std::endl;
#endif
}


int main() {
    test_tensor_creation();
    return 0;
}
```

### Step 6.2: Add the Test to `tests/CMakeLists.txt`

```cmake
# Vesper/tests/CMakeLists.txt
# ... (previous tests)

# New tensor test
add_executable(tensor_tests test_tensor.cpp)
target_link_libraries(tensor_tests PRIVATE vesper)
add_test(NAME TensorTests COMMAND tensor_tests)
```

### Step 6.3: Build and Run

```sh
cd /path/to/vesper/build
cmake ..
make -j
ctest --verbose
```

**Expected Output:**

CTest will now run three tests. `TensorTests` should pass, confirming that your `Tensor` class is correctly initialized with the right shape, strides, and other metadata.

```
Running tests...
Test project /path/to/vesper/build
    Start 1: CoreEnumTests
1/3 Test #1: CoreEnumTests ....................   Passed    0.00 sec
    Start 2: StorageTests
2/3 Test #2: StorageTests .....................   Passed    0.01 sec
    Start 3: TensorTests
3/3 Test #3: TensorTests ......................   Passed    0.01 sec

100% tests passed, 0 tests failed out of 3
```

With the `Tensor` class in place, you now have the core data structure ready. The next step is to start giving it functionality by implementing mathematical operations.
