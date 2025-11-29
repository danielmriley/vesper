#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/engine.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/copy.h>
#include <vesper/core/stream.h>
#include <vesper/ops/view_ops.h>
#include <vesper/ops/cast.h>
#include <vesper/ops/random.h>
#include <cstring> // for std::memset
#include <functional> // for std::function

#if USE_HIP_BACKEND
#include <hip/hip_runtime.h>
#endif

#if USE_CUDA_BACKEND
#include <cuda_runtime.h>
#endif

namespace vesper {

Tensor Tensor::index(const std::vector<IndexSelector>& selectors) const {
    return ops::index(*this, selectors);
}

Tensor Tensor::slice(int64_t start, int64_t stop, int64_t step) const {
    return index({ Slice(start, stop, step) });
}

Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    // Normalize negative dimension
    int64_t ndim = static_cast<int64_t>(shape_.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("narrow: dimension out of range");
    }
    
    // Build selectors: Slice() for all dims before 'dim', then Slice(start, start+length) for 'dim'
    std::vector<IndexSelector> selectors;
    for (int64_t i = 0; i < dim; ++i) {
        selectors.push_back(Slice());  // Keep all elements in this dimension
    }
    selectors.push_back(Slice(start, start + length));  // Slice the target dimension
    // Remaining dimensions are automatically kept by index()
    
    return index(selectors);
}

// --- Private Constructor Implementation ---
Tensor::Tensor(std::shared_ptr<Storage> storage,
               DType dtype,
               std::vector<int64_t> shape,
               std::vector<int64_t> strides,
               size_t offset,
               bool requires_grad)
    : storage_(std::move(storage)),
      dtype_(dtype),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset),
      requires_grad_(requires_grad),
      grad_handle_(std::make_shared<std::shared_ptr<Tensor>>(nullptr)) {}

Tensor& Tensor::grad() {
    if (!*grad_handle_) {
        // Lazily initialize gradient tensor as a tensor of zeros
        // with the same properties as this tensor.
        *grad_handle_ = std::make_shared<Tensor>(
            zeros(this->shape(), this->dtype(), this->device(), false)
        );
    }
    return **grad_handle_;
}

void Tensor::accumulate_grad(const Tensor& grad_update) {
    if (!*grad_handle_) {
        *grad_handle_ = std::make_shared<Tensor>(grad_update);
    } else {
        **grad_handle_ = ops::add(**grad_handle_, grad_update);
    }
}

void Tensor::backward() {
    autograd::Engine::backward(*this);
}

void Tensor::backward(const Tensor& gradient) {
    autograd::Engine::backward(*this, gradient);
}

void Tensor::copy_from_host(const void* host_ptr) {
    const size_t size_bytes = this->numel() * GetDTypeSize(this->dtype_);
    switch (this->device()) {
        case Device::HIP:
#if USE_HIP_BACKEND
            {
                if (!is_contiguous()) {
                     throw std::runtime_error("copy_from_host not implemented for non-contiguous HIP tensors yet.");
                }
                hipError_t err = hipMemcpy(this->data_ptr<void>(), host_ptr, size_bytes, hipMemcpyHostToDevice);
                if (err != hipSuccess) {
                    throw std::runtime_error("hipMemcpy (HostToDevice) failed");
                }
            }
#else
            throw std::runtime_error("HIP backend not enabled.");
#endif
            break;
        case Device::CPU:
            if (is_contiguous()) {
                std::memcpy(this->data_ptr<void>(), host_ptr, size_bytes);
            } else {
                // Strided copy from contiguous host buffer to non-contiguous tensor
                const char* src_ptr = static_cast<const char*>(host_ptr);
                size_t elem_size = GetDTypeSize(dtype_);
                
                std::function<void(int, size_t)> copy_recursive = 
                    [&](int dim, size_t offset) {
                        if (dim == static_cast<int>(shape_.size())) {
                            char* dst = static_cast<char*>(storage_->data()) + (offset_ + offset) * elem_size;
                            std::memcpy(dst, src_ptr, elem_size);
                            src_ptr += elem_size;
                            return;
                        }
                        for (int64_t i = 0; i < shape_[dim]; ++i) {
                            copy_recursive(dim + 1, offset + i * strides_[dim]);
                        }
                    };
                copy_recursive(0, 0);
            }
            break;
        case Device::CUDA:
#if USE_CUDA_BACKEND
            {
                if (!is_contiguous()) {
                     throw std::runtime_error("copy_from_host not implemented for non-contiguous CUDA tensors yet.");
                }
                cudaError_t err = cudaMemcpy(this->data_ptr<void>(), host_ptr, size_bytes, cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    throw std::runtime_error("cudaMemcpy (HostToDevice) failed");
                }
            }
#else
            throw std::runtime_error("CUDA backend not enabled.");
#endif
            break;
        default:
            throw std::runtime_error("Device not supported for copy_from_host.");
    }
}

void Tensor::copy_to_host(void* host_ptr) const {
    const size_t size_bytes = this->numel() * GetDTypeSize(this->dtype_);
    
    if (!is_contiguous()) {
        // For non-contiguous tensors (e.g. permuted), we first create a contiguous copy
        // on the device using our optimized kernel, then copy that to host.
        Tensor contig = this->contiguous();
        contig.copy_to_host(host_ptr);
        return;
    }

    switch (this->device()) {
        case Device::HIP:
#if USE_HIP_BACKEND
            {
                hipError_t err = hipMemcpy(host_ptr, this->data_ptr<const void>(), size_bytes, hipMemcpyDeviceToHost);
                if (err != hipSuccess) {
                    throw std::runtime_error("hipMemcpy (DeviceToHost) failed");
                }
            }
#else
            throw std::runtime_error("HIP backend not enabled.");
#endif
            break;
        case Device::CPU:
            std::memcpy(host_ptr, this->data_ptr<const void>(), size_bytes);
            break;
        case Device::CUDA:
#if USE_CUDA_BACKEND
            {
                cudaError_t err = cudaMemcpy(host_ptr, this->data_ptr<const void>(), size_bytes, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    throw std::runtime_error("cudaMemcpy (DeviceToHost) failed");
                }
            }
#else
            throw std::runtime_error("CUDA backend not enabled.");
#endif
            break;
        default:
            throw std::runtime_error("Device not supported for copy_to_host.");
    }
}

void Tensor::copy_from(const Tensor& other) {
    if (this->numel() != other.numel()) {
        throw std::runtime_error("copy_from: element count mismatch.");
    }
    if (this->dtype() != other.dtype()) {
        throw std::runtime_error("copy_from: dtype mismatch.");
    }
    
    // For now, support only contiguous copy or basic checks
    if (!this->is_contiguous() || !other.is_contiguous()) {
        if (this->device() == Device::CPU && other.device() == Device::CPU) {
             ops::copy_strided_cpu_dispatch(other, *this);
             return;
        }
        if (this->device() == Device::CUDA && other.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
             ops::copy_strided_cuda_dispatch(other, *this);
             return;
#else
             throw std::runtime_error("CUDA backend not enabled.");
#endif
        }
        if (this->device() == Device::HIP && other.device() == Device::HIP) {
#if USE_HIP_BACKEND
             ops::copy_strided_hip_dispatch(other, *this);
             return;
#else
             throw std::runtime_error("HIP backend not enabled.");
#endif
        }
        throw std::runtime_error("copy_from currently only supports contiguous tensors.");
    }

    const size_t size_bytes = this->numel() * GetDTypeSize(this->dtype_);

    if (this->device() == Device::CPU && other.device() == Device::CPU) {
        std::memcpy(this->data_ptr<void>(), other.data_ptr<const void>(), size_bytes);
    } 
    else if (this->device() == Device::HIP && other.device() == Device::HIP) {
#if USE_HIP_BACKEND
        hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
        hipError_t err = hipMemcpyAsync(this->data_ptr<void>(), other.data_ptr<const void>(), size_bytes, hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess) {
            throw std::runtime_error("hipMemcpyAsync (DeviceToDevice) failed");
        }
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    }
    else if (this->device() == Device::CUDA && other.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
        cudaError_t err = cudaMemcpyAsync(this->data_ptr<void>(), other.data_ptr<const void>(), size_bytes, cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpyAsync (DeviceToDevice) failed");
        }
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    }
    else if (this->device() == Device::HIP && other.device() == Device::CPU) {
        this->copy_from_host(other.data_ptr<const void>());
    }
    else if (this->device() == Device::CPU && other.device() == Device::HIP) {
        other.copy_to_host(this->data_ptr<void>());
    }
    else if (this->device() == Device::CUDA && other.device() == Device::CPU) {
        this->copy_from_host(other.data_ptr<const void>());
    }
    else if (this->device() == Device::CPU && other.device() == Device::CUDA) {
        other.copy_to_host(this->data_ptr<void>());
    }
    else {
        throw std::runtime_error("copy_from: Cross-device copy between HIP/CUDA not supported directly.");
    }
}

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

Tensor empty(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad) {
    int64_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }

    size_t size_bytes = num_elements * GetDTypeSize(dtype);
    auto storage = std::make_shared<Storage>(device, size_bytes);
    auto strides = calculate_contiguous_strides(shape);

    return Tensor(std::move(storage), dtype, shape, strides, 0, requires_grad);
}

Tensor zeros(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad) {
    Tensor t = empty(shape, dtype, device, requires_grad);
    size_t bytes = t.numel() * GetDTypeSize(dtype);
    
    void* ptr = t.data_ptr<char>();

    if (device == Device::HIP) {
#if USE_HIP_BACKEND
        hipError_t err = hipMemset(ptr, 0, bytes);
        if (err != hipSuccess) {
            throw std::runtime_error("hipMemset failed in zeros()");
        }
#else
        throw std::runtime_error("HIP backend not enabled");
#endif
    } else if (device == Device::CPU) {
        std::memset(ptr, 0, bytes);
    } else if (device == Device::CUDA) {
#if USE_CUDA_BACKEND
        cudaError_t err = cudaMemset(ptr, 0, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed in zeros()");
        }
#else
        throw std::runtime_error("CUDA backend not enabled");
#endif
    } else {
        throw std::runtime_error("Backend not implemented for zeros");
    }
    return t;
}

Tensor ones(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad) {
    return full(shape, dtype, device, 1.0f, requires_grad);
}

Tensor randn(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad) {
    Tensor t = empty(shape, dtype, device, requires_grad);
    ops::normal_(t, 0.0f, 1.0f);
    return t;
}

Tensor full(const std::vector<int64_t>& shape, DType dtype, Device device, float val, bool requires_grad) {
    auto t = empty(shape, dtype, device, requires_grad);
    if (dtype != DType::Float32) {
        throw std::runtime_error("full factory currently only supports Float32");
    }
    std::vector<float> data(t.numel(), val);
    t.copy_from_host(data.data());
    return t;
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    return ops::transpose(*this, dim0, dim1);
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }

    Tensor contig_tensor = empty(shape_, dtype_, device(), requires_grad_);

    switch (device()) {
        case Device::HIP:
#if USE_HIP_BACKEND
            ops::copy_strided_hip_dispatch(*this, contig_tensor);
#else
            throw std::runtime_error("HIP backend not enabled.");
#endif
            break;
        case Device::CPU:
            ops::copy_strided_cpu_dispatch(*this, contig_tensor);
            break;
        case Device::CUDA:
#if USE_CUDA_BACKEND
            ops::copy_strided_cuda_dispatch(*this, contig_tensor);
#else
            throw std::runtime_error("CUDA backend not enabled.");
#endif
            break;
        default:
            throw std::runtime_error("Device not supported for contiguous().");
    }

    if (this->requires_grad()) {
        contig_tensor.grad_node = std::make_shared<autograd::Node>();
        if (this->grad_node) {
            contig_tensor.grad_node->next_edges.push_back({this->grad_node});
        }
        contig_tensor.grad_node->backward_fn = [self = *this, weak_contig=contig_tensor.weak()]() mutable {
            auto contig_tensor = weak_contig.lock();
            if (!contig_tensor) return;
            self.accumulate_grad(contig_tensor->grad());
        };
    }

    return contig_tensor;
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    return ops::reshape(*this, new_shape);
}

Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
    return ops::view(*this, new_shape);
}

Tensor Tensor::permute(const std::vector<int64_t>& dims) const {
    return ops::permute(*this, dims);
}

Tensor Tensor::slice(size_t index) const {
    return ops::slice(*this, index);
}

Tensor Tensor::to(Device device, bool non_blocking) const {
    if (this->device() == device) {
        return *this;
    }
    
    // If this tensor is non-contiguous and we're crossing device boundaries,
    // copy_from will fail. Make a contiguous copy first on the source device.
    const Tensor* src = this;
    Tensor contiguous_copy;
    if (!this->is_contiguous()) {
        contiguous_copy = this->contiguous();
        src = &contiguous_copy;
    }
    
    // Create new tensor on target device
    Tensor result = empty(shape_, dtype_, device, requires_grad_);
    result.copy_from(*src);
    
    // TODO: If non_blocking is true and we have CUDA/HIP streams,
    // we could return immediately without synchronizing.
    // For now, copy_from is synchronous regardless of non_blocking.
    (void)non_blocking;  // Suppress unused warning
    
    if (requires_grad_) {
        result.grad_node = std::make_shared<autograd::Node>();
        if (grad_node) {
            result.grad_node->next_edges.push_back({grad_node});
        }
        
        // Capture source device for backward
        Device source_device = this->device();
        
        result.grad_node->backward_fn = [self=*this, weak_res=result.weak(), source_device]() mutable {
            auto result = weak_res.lock();
            if (!result) return;
            if (self.requires_grad()) {
                // Get gradient from result (on target device)
                Tensor grad_output = result->grad();
                
                // Move gradient back to source device
                // Note: This calls .to() recursively, but on the gradient tensor.
                // If higher-order gradients are needed, this is correct.
                Tensor grad_input = grad_output.to(source_device);
                
                self.accumulate_grad(grad_input);
            }
        };
    }
    
    return result;
}

Tensor& Tensor::to_(Device device, bool non_blocking) {
    if (this->device() == device) {
        return *this;
    }
    
    // Create new storage on target device and copy data
    // Note: to() always creates a contiguous tensor via empty() + copy_from()
    Tensor temp = this->to(device, non_blocking);
    
    // Swap the storage (in-place modification)
    storage_ = std::move(temp.storage_);
    
    // CRITICAL: Update strides and offset to match the new contiguous storage
    // The temp tensor is always contiguous (created by empty()), so we must
    // update our strides to be contiguous as well, otherwise we'll read
    // from wrong memory offsets if this tensor was non-contiguous (e.g., transposed)
    strides_ = temp.strides_;
    offset_ = temp.offset_;  // Should be 0 for a freshly created tensor
    
    // Note: shape_, dtype_ remain unchanged (same logical tensor shape)
    // Note: requires_grad_ stays the same
    // Note: We do NOT move grad_node - the autograd graph should remain intact
    //       since this is the same logical tensor, just on a different device
    
    return *this;
}

Tensor Tensor::to(DType target_dtype) const {
    if (dtype_ == target_dtype) {
        return *this;
    }
    return ops::cast(*this, target_dtype);
}

Tensor& Tensor::add_(const Tensor& other) {
    return ops::add_(*this, other);
}

Tensor& Tensor::add_(float value) {
    return ops::add_(*this, value);
}

Tensor& Tensor::sub_(const Tensor& other) {
    return ops::sub_(*this, other);
}

Tensor& Tensor::sub_(float value) {
    return ops::sub_(*this, value);
}

Tensor& Tensor::mul_(float other) {
    return ops::mul_(*this, other);
}

Tensor& Tensor::copy_(const Tensor& src) {
    if (shape_ != src.shape()) {
        throw std::runtime_error("copy_(): shape mismatch");
    }
    if (dtype_ != src.dtype()) {
        throw std::runtime_error("copy_(): dtype mismatch");
    }
    copy_from(src);
    return *this;
}

Tensor& Tensor::zero_() {
    // Fill storage with zeros
    size_t num_bytes = numel() * GetDTypeSize(dtype_);
    if (device() == Device::CPU) {
        std::memset(storage_->data(), 0, num_bytes);
    } else {
#if USE_HIP_BACKEND
        (void)hipMemset(storage_->data(), 0, num_bytes);
#elif USE_CUDA_BACKEND
        (void)cudaMemset(storage_->data(), 0, num_bytes);
#else
        throw std::runtime_error("zero_(): GPU backend not available");
#endif
    }
    return *this;
}

Tensor Tensor::clone() const {
    // Create a new tensor with the same properties
    // If contiguous, just copy. If not, contiguous() creates a copy anyway.
    // But clone() usually implies detaching from graph?
    // PyTorch clone() propagates gradients.
    // So we should use an op if we want autograd.
    // But we don't have a 'copy' op exposed to autograd yet.
    // Let's implement it as a deep copy that DOES NOT propagate gradients for now (standard copy constructor behavior in C++ usually deep copies value types, but Tensor is reference type).
    // Wait, PyTorch clone() is differentiable.
    // For MVP, we'll implement it as `contiguous()` which creates a new tensor and copy, and IS differentiable.
    // If it's already contiguous, `contiguous()` returns *this (view). We want a COPY.
    
    Tensor result = empty(shape_, dtype_, device(), requires_grad_);
    result.copy_from(*this);
    
    if (requires_grad_) {
        result.grad_node = std::make_shared<autograd::Node>();
        if (grad_node) {
            result.grad_node->next_edges.push_back({grad_node});
        }
        result.grad_node->backward_fn = [self=*this, weak_res=result.weak()]() mutable {
            auto result = weak_res.lock();
            if (!result) return;
            self.accumulate_grad(result->grad());
        };
    }
    return result;
}

// Operator Overloading Implementation
Tensor operator+(const Tensor& a, const Tensor& b) { return ops::add(a, b); }
Tensor operator+(const Tensor& a, float b) { return ops::add(a, b); }
Tensor operator+(float a, const Tensor& b) { return ops::add(b, a); } // Commutative

Tensor operator-(const Tensor& a, const Tensor& b) { return ops::sub(a, b); }
Tensor operator-(const Tensor& a, float b) { return ops::sub(a, b); }
Tensor operator-(float a, const Tensor& b) { 
    // a - b = -b + a
    return ops::add(ops::mul(b, -1.0f), a); 
}

Tensor operator*(const Tensor& a, const Tensor& b) { return ops::mul(a, b); }
Tensor operator*(const Tensor& a, float b) { return ops::mul(a, b); }
Tensor operator*(float a, const Tensor& b) { return ops::mul(b, a); } // Commutative

Tensor operator/(const Tensor& a, const Tensor& b) { return ops::div(a, b); }
Tensor operator/(const Tensor& a, float b) { return ops::div(a, b); }
Tensor operator/(float a, const Tensor& b) { 
    // a / b = a * (1/b) = a * b^-1
    // We don't have power/reciprocal op yet.
    // But we can use full() to make 'a' a tensor and divide.
    auto a_tensor = full(b.shape(), b.dtype(), b.device(), a);
    // Wait, broadcasting might be needed if b is large.
    // full(b.shape()...) creates tensor of b's shape.
    // This is elementwise a / b_i.
    // Correct.
    return ops::div(a_tensor, b);
}

} // namespace vesper