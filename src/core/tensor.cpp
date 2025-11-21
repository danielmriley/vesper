#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/engine.h>
#include <vesper/ops/elementwise.h>
#include <cstring> // for std::memset

#if USE_HIP_BACKEND
#include <hip/hip_runtime.h>
#endif

namespace vesper {

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
        // Note: The gradient itself does not require a gradient (usually).
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
                        if (dim == shape_.size()) {
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
        default:
            throw std::runtime_error("Device not supported for copy_from_host.");
    }
}

void Tensor::copy_to_host(void* host_ptr) const {
    const size_t size_bytes = this->numel() * GetDTypeSize(this->dtype_);
    switch (this->device()) {
        case Device::HIP:
#if USE_HIP_BACKEND
            {
                if (!is_contiguous()) {
                     throw std::runtime_error("copy_to_host not implemented for non-contiguous HIP tensors yet.");
                }
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
            if (is_contiguous()) {
                std::memcpy(host_ptr, this->data_ptr<const void>(), size_bytes);
            } else {
                // Strided copy from non-contiguous tensor to contiguous host buffer
                char* dst_ptr = static_cast<char*>(host_ptr);
                size_t elem_size = GetDTypeSize(dtype_);

                std::function<void(int, size_t)> copy_recursive = 
                    [&](int dim, size_t offset) {
                        if (dim == shape_.size()) {
                            const char* src = static_cast<const char*>(storage_->data()) + (offset_ + offset) * elem_size;
                            std::memcpy(dst_ptr, src, elem_size);
                            dst_ptr += elem_size;
                            return;
                        }
                        for (int64_t i = 0; i < shape_[dim]; ++i) {
                            copy_recursive(dim + 1, offset + i * strides_[dim]);
                        }
                    };
                copy_recursive(0, 0);
            }
            break;
        default:
            throw std::runtime_error("Device not supported for copy_to_host.");
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
    
    // We can access storage directly via data_ptr<void> if we had it, 
    // or just cast data_ptr<char>
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
    } else {
        throw std::runtime_error("Backend not implemented for zeros");
    }
    return t;
}

Tensor full(const std::vector<int64_t>& shape, DType dtype, Device device, float val, bool requires_grad) {
    auto t = empty(shape, dtype, device, requires_grad);
    // Currently assuming float32 for simplicity as per build plan
    if (dtype != DType::Float32) {
        throw std::runtime_error("full factory currently only supports Float32");
    }
    std::vector<float> data(t.numel(), val);
    t.copy_from_host(data.data());
    return t;
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    if (dim0 < 0 || dim0 >= static_cast<int64_t>(shape_.size()) || 
        dim1 < 0 || dim1 >= static_cast<int64_t>(shape_.size())) {
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
    
    // Views have their own gradient state
    transposed_view.grad_handle_ = std::make_shared<std::shared_ptr<Tensor>>(nullptr);
    transposed_view.grad_node = nullptr;

    return transposed_view;
}

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
        if (this->grad_node) {
            contig_tensor.grad_node->next_edges.push_back({this->grad_node});
        }
        contig_tensor.grad_node->backward_fn = [self = *this, contig_tensor]() mutable {
            // The gradient for a copy operation is just to pass the upstream
            // gradient back to the original tensor.
            self.accumulate_grad(contig_tensor.grad());
        };
    }

    return contig_tensor;
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("Reshape currently only supports contiguous tensors.");
    }
    
    int64_t new_numel = 1;
    for (auto dim : new_shape) {
        new_numel *= dim;
    }
    
    if (new_numel != numel()) {
        throw std::runtime_error("Reshape: total number of elements must match.");
    }
    
    Tensor reshaped_view = *this;
    reshaped_view.shape_ = new_shape;
    reshaped_view.strides_ = calculate_contiguous_strides(new_shape);
    
    // Views have their own gradient state
    reshaped_view.grad_handle_ = std::make_shared<std::shared_ptr<Tensor>>(nullptr);
    reshaped_view.grad_node = nullptr;
    
    return reshaped_view;
}

} // namespace vesper
