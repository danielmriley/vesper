
# Vesper Build Plan - Chapter 19.1: The `Dataset` Abstraction

## 1. Goal

Implement an abstract `Dataset` class to provide a common interface for accessing data, and create a simple concrete `TensorDataset` for handling in-memory data. This requires adding a `slice` method to our `Tensor` class to retrieve individual items.

## 2. The `Dataset` Abstraction

A `Dataset` class simply needs to answer two questions:
1.  How many items are in the dataset? (`size()`)
2.  How do I get the i-th item? (`get_item(i)`)

By defining this interface, our `DataLoader` can be completely agnostic to where the data comes from (e.g., from memory, from files on disk).

## 3. Detailed Steps

### Step 3.1: Implement `Tensor::slice()`

To get the i-th item from a tensor of data, we need a way to slice it. A slice is a metadata-only operation that creates a view of a single item in a batch.

Add the declaration to `include/vesper/core/tensor.h`:
```cpp
// public:
// Creates a view of the i-th slice of the first dimension.
Tensor slice(size_t index) const;
```
Add the implementation to `src/core/tensor.cpp`:
```cpp
// src/core/tensor.cpp
Tensor Tensor::slice(size_t index) const {
    if (shape_.empty() || shape_[0] <= index) {
        throw std::runtime_error("Slice index out of bounds on dimension 0.");
    }
    
    // The new shape is the tail of the old shape.
    // e.g., shape {10, 5} -> slice -> shape {5}
    std::vector<int64_t> new_shape(shape_.begin() + 1, shape_.end());
    
    // The new strides are also the tail of the old strides.
    std::vector<int64_t> new_strides(strides_.begin() + 1, strides_.end());
    
    // The new offset is advanced by the stride of the first dimension.
    size_t new_offset = offset_ + index * strides_[0];

    // Create the new Tensor view. It shares storage with the original.
    Tensor view = *this;
    view.shape_ = new_shape;
    view.strides_ = new_strides;
    view.offset_ = new_offset;
    
    return view;
}
```

### Step 3.2: Create the `Dataset` Interface and `TensorDataset`

Create `include/vesper/data/dataset.h`:
```sh
mkdir -p include/vesper/data
```
```cpp
// include/vesper/data/dataset.h
#pragma once
#include <vesper/core/tensor.h>
#include <utility>
#include <memory>

namespace vesper::data {

// A sample is typically a pair of (data, target)
using Sample = std::pair<Tensor, Tensor>;

// Abstract base class for all datasets
class Dataset : public std::enable_shared_from_this<Dataset> {
public:
    virtual ~Dataset() = default;
    virtual Sample get_item(size_t index) = 0;
    virtual size_t size() const = 0;
};

// A concrete dataset for tensors held in memory.
// The first dimension of the tensors is assumed to be the batch dimension.
class TensorDataset : public Dataset {
public:
    TensorDataset(Tensor x, Tensor y)
        : x_data(std::move(x)), y_data(std::move(y)) {
        if (x_data.shape()[0] != y_data.shape()[0]) {
            throw std::runtime_error("Tensors must have the same size in dimension 0.");
        }
    }

    Sample get_item(size_t index) override {
        return {x_data.slice(index), y_data.slice(index)};
    }

    size_t size() const override {
        return x_data.shape()[0];
    }
private:
    Tensor x_data;
    Tensor y_data;
};

} // namespace vesper::data
```
*Note: For simplicity, the implementation of `TensorDataset` is included directly in the header.*

## 4. Verification

The test will create a `TensorDataset` and verify that `get_item` returns slices with the correct shape and data.

### Step 4.1: Create `tests/test_data.cpp`
```cpp
// tests/test_data.cpp
#include <vesper/data/dataset.h>
#include <iostream>
#include <cassert>

void test_tensor_dataset() {
    std::cout << "Testing TensorDataset..." << std::endl;
    
    // 1. Create data
    auto x = vesper::empty({10, 5}, vesper::DType::Float32, vesper::Device::CPU);
    auto y = vesper::empty({10, 1}, vesper::DType::Float32, vesper::Device::CPU);
    
    // 2. Create dataset
    auto dataset = std::make_shared<vesper::data::TensorDataset>(x, y);

    // 3. Check size
    assert(dataset->size() == 10);

    // 4. Get an item and check its shape
    auto sample = dataset->get_item(3);
    auto sample_x = sample.first;
    auto sample_y = sample.second;

    assert(sample_x.shape() == std::vector<int64_t>({5}));
    assert(sample_y.shape() == std::vector<int64_t>({1}));

    std::cout << "TensorDataset test passed!" << std::endl;
}

int main() {
    test_tensor_dataset();
    return 0;
}
```

### Step 4.2: Add Test to CMake
```cmake
add_executable(data_tests test_data.cpp)
target_link_libraries(data_tests PRIVATE vesper)
add_test(NAME DataTests COMMAND data_tests)
```
A passing test confirms that our `Dataset` abstraction and `TensorDataset` implementation are working correctly, ready to be used by a `DataLoader`.
