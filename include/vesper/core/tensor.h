#pragma once

#include <vesper/core/storage.h>
#include <vesper/core/dtype.h>
#include <vesper/autograd/node.h>
#include <vector>
#include <memory>
#include <numeric>
#include <functional>
#include <atomic>

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

    // --- Autograd Accessors ---
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

    // Access the gradient tensor
    Tensor& grad();

    // Accumulates a gradient update into the .grad() tensor
    void accumulate_grad(const Tensor& grad_update);

    // Computes the gradient of this tensor with respect to graph leaves
    void backward();

    // The node that created this tensor in the graph
    std::shared_ptr<autograd::Node> grad_node;

    // Copies data from a CPU buffer to the tensor's storage
    void copy_from_host(const void* host_ptr);
    // Copies data from the tensor's storage to a CPU buffer
    void copy_to_host(void* host_ptr) const;

    // Returns a typed pointer to the start of the tensor's data (respecting offset)
    template <typename T>
    T* data_ptr() {
        char* raw_ptr = static_cast<char*>(storage_->data());
        size_t byte_offset = offset_ * GetDTypeSize(dtype_);
        return static_cast<T*>(static_cast<void*>(raw_ptr + byte_offset));
    }

    template <typename T>
    const T* data_ptr() const {
        const char* raw_ptr = static_cast<const char*>(storage_->data());
        size_t byte_offset = offset_ * GetDTypeSize(dtype_);
        return static_cast<const T*>(static_cast<const void*>(raw_ptr + byte_offset));
    }

    // Returns a new tensor with the same data but with dimensions swapped.
    // This is a metadata-only operation (creates a view).
    Tensor transpose(int64_t dim0, int64_t dim1) const;

    // Returns a new tensor with the same data but different shape.
    // Only works if the total number of elements remains the same.
    // Currently only supports reshaping contiguous tensors.
    Tensor reshape(const std::vector<int64_t>& new_shape) const;

    // Returns a contiguous copy of the tensor.
    // If the tensor is already contiguous, returns *this.
    Tensor contiguous() const;

private:
    // Private constructor to be used by factory functions
    Tensor(std::shared_ptr<Storage> storage,
           DType dtype,
           std::vector<int64_t> shape,
           std::vector<int64_t> strides,
           size_t offset = 0,
           bool requires_grad = false);

    // Grant factory functions access to the private constructor
    friend Tensor empty(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad);

    std::shared_ptr<Storage> storage_;
    DType dtype_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    size_t offset_; // in number of elements, not bytes

    // --- Autograd Members ---
    bool requires_grad_ = false;
    // Indirect pointer to the gradient tensor to allow sharing between shallow copies
    std::shared_ptr<std::shared_ptr<Tensor>> grad_handle_; 
};

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

} // namespace vesper
