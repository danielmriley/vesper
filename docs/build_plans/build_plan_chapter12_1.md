
# Vesper Build Plan - Chapter 12.1: Tensor Views and Data Layout

## 1. Goal

Implement fundamental tensor manipulation methods required for advanced operations: `transpose()` and `contiguous()`. These methods are critical for managing data layouts efficiently without unnecessary copies, a core concept in high-performance tensor libraries.

## 2. Concepts

-   **`transpose()`**: This is a "metadata-only" operation. It does not move any data in memory. Instead, it returns a new `Tensor` *view* that points to the **same underlying storage** but has its shape and stride information reordered. This is extremely fast but results in a "non-contiguous" tensor.
-   **`contiguous()`**: This method is the bridge between non-contiguous views and operations that require a standard memory layout. It checks if a tensor's memory is contiguous. If it is, it returns the tensor itself (no cost). If not, it performs a deep copy of the data into a new, contiguous block of memory and returns a new tensor.

## 3. Detailed Steps

### Step 3.1: Implement `Tensor::transpose()`

This method creates a new `Tensor` view with swapped dimensions and strides.

Add the declaration to `include/vesper/core/tensor.h`:
```cpp
// In public section of Tensor class
Tensor transpose(int64_t dim0, int64_t dim1) const;
```

Add the implementation to `src/core/tensor.cpp`:
```cpp
// src/core/tensor.cpp
Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    // Note: This implementation is for a Tensor object that is copyable
    // and shares its storage via std::shared_ptr.

    if (dim0 < 0 || dim0 >= shape_.size() || dim1 < 0 || dim1 >= shape_.size()) {
        throw std::runtime_error("Transpose dimensions are out of bounds.");
    }
    
    auto new_shape = shape_;
    std::swap(new_shape[dim0], new_shape[dim1]);

    auto new_strides = strides_;
    std::swap(new_strides[dim0], new_strides[dim1]);

    // Create a new Tensor header pointing to the same storage, but with
    // updated metadata. This is a fast, shallow operation.
    Tensor transposed_view = *this;
    transposed_view.shape_ = new_shape;
    transposed_view.strides_ = new_strides;

    return transposed_view;
}
```

### Step 3.2: Implement `Tensor::contiguous()`

This method ensures a tensor has a standard memory layout, performing a copy if necessary.

Add the declaration to `include/vesper/core/tensor.h`:
```cpp
// In public section of Tensor class
Tensor contiguous() const;
```

Add the implementation to `src/core/tensor.cpp`:
```cpp
// src/core/tensor.cpp
Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }

    // If not contiguous, create a new tensor and perform a deep copy.
    Tensor contig_tensor = empty(shape_, dtype_, device(), requires_grad_);

    // This requires a copy kernel that can handle arbitrary strides.
    // For now, we implement a slow but correct version by copying via the host.
    // A future optimization is a dedicated `copy_kernel` on the GPU.
    std::vector<uint8_t> host_buffer(this->numel() * GetDTypeSize(dtype_));
    this->copy_to_host(host_buffer.data());
    contig_tensor.copy_from_host(host_buffer.data());

    // If the original tensor required a gradient, we need to link the new
    // contiguous tensor back to it in the autograd graph.
    if (this->requires_grad()) {
        contig_tensor.grad_node = std::make_shared<autograd::Node>();
        contig_tensor.grad_node->next_edges.push_back({this->grad_node});
        contig_tensor.grad_node->backward_fn = [self = *this, contig_tensor]() mutable {
            // The gradient for a copy operation is just to pass the upstream
            // gradient back to the original tensor.
            self.accumulate_grad(contig_tensor.grad());
        };
    }

    return contig_tensor;
}
```

## 4. Verification

Create a test to validate that `transpose` creates a correct view and that `contiguous` correctly copies the data.

### Step 4.1: Create `tests/test_tensor_views.cpp`

```cpp
// tests/test_tensor_views.cpp
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>
#include <vector>

void test_transpose_and_contiguous() {
    std::cout << "Testing transpose and contiguous..." << std::endl;
    
    auto device = vesper::Device::CPU;
    auto a = vesper::empty({2, 3}, vesper::DType::Float32, device);
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    a.copy_from_host(data.data());

    // Original strides should be {3, 1}
    assert(a.strides()[0] == 3 && a.strides()[1] == 1);
    assert(a.is_contiguous());

    // 1. Test transpose
    auto b = a.transpose(0, 1);
    assert(b.shape() == std::vector<int64_t>({3, 2}));
    // New strides should be {1, 3}
    assert(b.strides()[0] == 1 && b.strides()[1] == 3);
    assert(!b.is_contiguous()); // Transposed tensor is not contiguous

    // 2. Test contiguous
    auto c = b.contiguous();
    assert(c.is_contiguous());
    assert(c.shape() == std::vector<int64_t>({3, 2}));
    // New contiguous strides should be {2, 1}
    assert(c.strides()[0] == 2 && c.strides()[1] == 1);

    // 3. Verify data is preserved
    std::vector<float> c_data(c.numel());
    c.copy_to_host(c_data.data());
    
    // Expected data in c: {1, 4, 2, 5, 3, 6} (transposed layout)
    assert(fabs(c_data[0] - 1) < 1e-6); assert(fabs(c_data[1] - 4) < 1e-6);
    assert(fabs(c_data[2] - 2) < 1e-6); assert(fabs(c_data[3] - 5) < 1e-6);
    assert(fabs(c_data[4] - 3) < 1e-6); assert(fabs(c_data[5] - 6) < 1e-6);

    std::cout << "Transpose and contiguous test passed!" << std::endl;
}

int main() {
    test_transpose_and_contiguous();
    return 0;
}
```
### Step 4.2: Add Test to `tests/CMakeLists.txt`
```cmake
add_executable(tensor_view_tests test_tensor_views.cpp)
target_link_libraries(tensor_view_tests PRIVATE vesper)
add_test(NAME TensorViewTests COMMAND tensor_view_tests)
```
A passing test confirms that you can now create non-contiguous views of tensors and restore them to a contiguous layout, a prerequisite for implementing robust matrix multiplication.
