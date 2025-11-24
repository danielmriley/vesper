#pragma once

#include <vesper/core/storage.h>
#include <vesper/core/dtype.h>
#include <vesper/core/slice.h>
#include <vesper/autograd/node.h>
#include <vector>
#include <memory>
#include <numeric>
#include <functional>
#include <atomic>
#include <optional>

namespace vesper {

class Tensor;

namespace ops {
    Tensor view(const Tensor& input, const std::vector<int64_t>& shape);
    Tensor transpose(const Tensor& input, int64_t dim0, int64_t dim1);
    Tensor permute(const Tensor& input, const std::vector<int64_t>& dims);
    Tensor slice(const Tensor& input, size_t index); // Legacy
    Tensor index(const Tensor& input, const std::vector<IndexSelector>& selectors); // Advanced
}

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
    // If gradient is not provided, defaults to a tensor of ones (for scalar outputs)
    void backward();
    void backward(const Tensor& gradient);

    // The node that created this tensor in the graph
    std::shared_ptr<autograd::Node> grad_node;

    // Copies data from a CPU buffer to the tensor's storage
    void copy_from_host(const void* host_ptr);
    // Copies data from the tensor's storage to a CPU buffer
    void copy_to_host(void* host_ptr) const;
    // Copies data from another tensor. Supports Device-to-Device copy.
    void copy_from(const Tensor& other);

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
    // If the new shape is compatible with the current strides, returns a view.
    // Otherwise, performs a copy (via contiguous()).
    Tensor reshape(const std::vector<int64_t>& new_shape) const;

    // Returns a new tensor with the same data but different shape.
    // Throws if the new shape is not compatible with the current strides (i.e. cannot be a view).
    // Requires tensor to be contiguous for now (simplification).
    Tensor view(const std::vector<int64_t>& new_shape) const;

    // Permutes the dimensions of the tensor.
    Tensor permute(const std::vector<int64_t>& dims) const;

    // Returns a contiguous copy of the tensor.
    // If the tensor is already contiguous, returns *this.
    Tensor contiguous() const;

    // Creates a view of the i-th slice of the first dimension. (Legacy)
    Tensor slice(size_t index) const;

    // Advanced Indexing / Slicing
    // Supports list of indices or slices.
    // Equivalent to tensor[i, j:k, ::2]
    Tensor index(const std::vector<IndexSelector>& selectors) const;

    // Helper for Python-like slicing on dim 0: tensor[start:stop:step]
    Tensor slice(int64_t start, int64_t stop, int64_t step = 1) const;

    // Moves/Copies tensor to the specified device
    Tensor to(Device device) const;
    
    // Casts tensor to the specified dtype
    Tensor to(DType dtype) const;

    // In-place operations
    Tensor& add_(const Tensor& other);
    Tensor& add_(float value);
    Tensor& sub_(const Tensor& other);
    Tensor& sub_(float value);
    Tensor& mul_(float other);

    // Returns the value of a scalar tensor
    template <typename T>
    T item() const {
        if (numel() != 1) {
            throw std::runtime_error("item() only supported for scalar tensors");
        }
        T val;
        copy_to_host(&val);
        return val;
    }

    // Returns a deep copy of the tensor
    Tensor clone() const;

    // Debug helper to check reference count of the underlying storage
    long storage_use_count() const { return storage_.use_count(); }

    struct Weak {
        std::weak_ptr<Storage> storage;
        std::weak_ptr<std::shared_ptr<Tensor>> grad_handle;
        std::weak_ptr<autograd::Node> grad_node_weak;
        
        DType dtype;
        std::vector<int64_t> shape;
        std::vector<int64_t> strides;
        size_t offset;
        bool requires_grad;
        
        Weak(const Tensor& t) 
            : storage(t.storage_), 
              grad_handle(t.grad_handle_),
              grad_node_weak(t.grad_node),
              dtype(t.dtype_),
              shape(t.shape_),
              strides(t.strides_),
              offset(t.offset_),
              requires_grad(t.requires_grad_) {}
              
        std::optional<Tensor> lock() const {
            auto s = storage.lock();
            auto gh = grad_handle.lock();
            
            if (!s || !gh) return std::nullopt;
            
            Tensor t(s, dtype, shape, strides, offset, requires_grad);
            t.grad_handle_ = gh;
            t.grad_node = grad_node_weak.lock();
            
            return t;
        }
    };
    
    Weak weak() const { return Weak(*this); }

    // Default constructor
    Tensor() = default;

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
    
    friend Tensor ops::view(const Tensor&, const std::vector<int64_t>&);
    friend Tensor ops::transpose(const Tensor&, int64_t, int64_t);
    friend Tensor ops::permute(const Tensor&, const std::vector<int64_t>&);
    friend Tensor ops::slice(const Tensor&, size_t);
    friend Tensor ops::index(const Tensor&, const std::vector<IndexSelector>&);

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

// Operator Overloading
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator+(const Tensor& a, float b);
Tensor operator+(float a, const Tensor& b);

Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, float b);
Tensor operator-(float a, const Tensor& b);

Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, float b);
Tensor operator*(float a, const Tensor& b);

Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, float b);
Tensor operator/(float a, const Tensor& b);

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

