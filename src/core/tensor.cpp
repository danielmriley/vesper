#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/engine.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/copy.h>
#include <vesper/core/stream.h>
#include <cstring> // for std::memset
#include <functional> // for std::function

#if USE_HIP_BACKEND
#include <hip/hip_runtime.h>
#endif

#if USE_CUDA_BACKEND
#include <cuda_runtime.h>
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
    if (dim0 < 0 || dim0 >= static_cast<int64_t>(shape_.size()) || 
        dim1 < 0 || dim1 >= static_cast<int64_t>(shape_.size())) {
        throw std::runtime_error("Transpose dimensions are out of bounds.");
    }
    
    auto new_shape = shape_;
    std::swap(new_shape[dim0], new_shape[dim1]);

    auto new_strides = strides_;
    std::swap(new_strides[dim0], new_strides[dim1]);

    Tensor transposed_view = *this;
    transposed_view.shape_ = new_shape;
    transposed_view.strides_ = new_strides;
    
    transposed_view.grad_handle_ = std::make_shared<std::shared_ptr<Tensor>>(nullptr);
    transposed_view.grad_node = nullptr;

    return transposed_view;
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
        contig_tensor.grad_node->backward_fn = [self = *this, contig_tensor]() mutable {
            self.accumulate_grad(contig_tensor.grad());
        };
    }

    return contig_tensor;
}

// Helper to compute new strides for a view if possible
// Returns empty vector if not possible
std::vector<int64_t> compute_view_strides(const std::vector<int64_t>& old_shape, 
                                          const std::vector<int64_t>& old_strides, 
                                          const std::vector<int64_t>& new_shape) {
    // Basic check: total elements must match
    int64_t numel = 1;
    for(auto s : old_shape) numel *= s;
    int64_t new_numel = 1;
    for(auto s : new_shape) new_numel *= s;
    
    if (numel != new_numel) return {};

    // Prepare sparse old dims (ignore size 1 dims)
    std::vector<std::pair<int64_t, int64_t>> old_dims;
    for (size_t i = 0; i < old_shape.size(); ++i) {
        if (old_shape[i] != 1) {
            old_dims.push_back({old_shape[i], old_strides[i]});
        }
    }

    std::vector<int64_t> new_strides(new_shape.size());
    size_t old_idx = 0;

    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == 1) {
            new_strides[i] = 1; // Stride for size 1 dim is arbitrary, set to 1
            continue;
        }

        if (old_idx >= old_dims.size()) {
            // Should not happen if numel matches and we handle size 1 correctly
            return {}; 
        }

        int64_t d_size = old_dims[old_idx].first;
        int64_t d_stride = old_dims[old_idx].second;

        if (d_size == new_shape[i]) {
            // Exact match
            new_strides[i] = d_stride;
            old_idx++;
        } else if (d_size > new_shape[i]) {
            // Splitting a dimension (e.g. [4] -> [2, 2])
            // Requires divisibility
            if (d_size % new_shape[i] != 0) return {};
            
            // New outer dim gets stride = inner_stride * inner_size
            int64_t inner_size = d_size / new_shape[i];
            new_strides[i] = d_stride * inner_size;
            
            // Update the current old dim to represent the remainder
            old_dims[old_idx].first = inner_size;
            // stride stays same for the inner part
        } else {
            // Merging dimensions (e.g. [2, 2] -> [4])
            // We need to consume multiple old dimensions
            int64_t target = new_shape[i];
            int64_t current = d_size;
            int64_t current_stride = d_stride;
            
            // Check contiguity while merging
            // We need to merge old_dims[old_idx], old_dims[old_idx+1]...
            // Contiguity condition: outer_stride == inner_size * inner_stride
            
            // The stride for the merged dimension will be the stride of the LAST consumed dimension?
            // No, stride of the merged dim is the stride of the OUTER-most merged dim.
            // e.g. [2, 2] strides [2, 1]. Merge to [4]. Stride is 1? No, stride is 1.
            // Wait. Flat index: i * stride.
            // [2, 2] strides [2, 1].
            // (0,0) -> 0. (0,1) -> 1. (1,0) -> 2. (1,1) -> 3.
            // [4] stride 1.
            // 0->0, 1->1, 2->2, 3->3.
            // So merged stride is the stride of the LAST (inner-most) dimension consumed.
            
            // Let's verify: stride[k] == shape[k+1]*stride[k+1].
            
            new_strides[i] = d_stride; // Tentatively start with outer stride?
            // Wait, if [2, 2] -> [4], strides [2, 1].
            // merged stride should be 1 (the inner stride).
            // But `d_stride` is 2.
            // Ah, for merging, we require: stride[outer] == size[inner] * stride[inner].
            
            // Let's loop until we match size
            while (current < target) {
                old_idx++;
                if (old_idx >= old_dims.size()) return {}; // Out of dims
                
                int64_t next_size = old_dims[old_idx].first;
                int64_t next_stride = old_dims[old_idx].second;
                
                // Verify contiguity: current_stride must equal next_size * next_stride
                if (current_stride != next_size * next_stride) return {};
                
                current *= next_size;
                current_stride = next_stride; // Update to inner stride for next check?
                // Wait, the stride of the resulting merged dimension is the stride of the *last* dimension we merge.
                // e.g. [2, 2] -> [4]. stride is 1.
                // d_stride was 2. next_stride is 1. 2 == 2*1. OK.
                // final new_stride should be 1 (next_stride).
            }
            
            if (current != target) return {}; // Mismatch (e.g. trying to merge 2,3 into 5)
            
            new_strides[i] = old_dims[old_idx].second; // The stride of the last merged dim
            old_idx++;
        }
    }
    
    // Check if we consumed all old dims
    if (old_idx != old_dims.size()) return {};

    return new_strides;
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    // Try to create a view first
    try {
        return view(new_shape);
    } catch (...) {
        // Fallback: Copy to contiguous then view
        return contiguous().view(new_shape);
    }
}

Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
    int64_t new_numel = 1;
    int64_t inferred_dim_idx = -1;
    
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (inferred_dim_idx != -1) throw std::runtime_error("Only one dimension can be -1 (inferred)");
            inferred_dim_idx = i;
        } else {
            new_numel *= new_shape[i];
        }
    }
    
    std::vector<int64_t> resolved_shape = new_shape;
    if (inferred_dim_idx != -1) {
        if (numel() % new_numel != 0) throw std::runtime_error("Invalid shape for view (inferred dimension mismatch)");
        resolved_shape[inferred_dim_idx] = numel() / new_numel;
    } else {
        if (numel() != new_numel) throw std::runtime_error("View size mismatch");
    }

    // Try to compute compatible strides
    std::vector<int64_t> new_strides = compute_view_strides(shape_, strides_, resolved_shape);
    
    if (!new_strides.empty()) {
        Tensor view_t = *this;
        view_t.shape_ = resolved_shape;
        view_t.strides_ = new_strides;
        
        view_t.grad_handle_ = std::make_shared<std::shared_ptr<Tensor>>(nullptr);
        view_t.grad_node = nullptr;
        return view_t;
    }
    
    // Failed to view
    throw std::runtime_error("view() not possible with current strides (not contiguous or incompatible layout). Call contiguous() first or use reshape().");
}

Tensor Tensor::permute(const std::vector<int64_t>& dims) const {
    if (dims.size() != shape_.size()) {
        throw std::runtime_error("permute: number of dimensions must match");
    }
    
    std::vector<int64_t> new_shape(dims.size());
    std::vector<int64_t> new_strides(dims.size());
    std::vector<bool> seen(dims.size(), false);
    
    for (size_t i = 0; i < dims.size(); ++i) {
        int64_t d = dims[i];
        if (d < 0) d += shape_.size();
        
        if (d < 0 || d >= static_cast<int64_t>(shape_.size()) || seen[d]) {
            throw std::runtime_error("permute: invalid dimension index");
        }
        seen[d] = true;
        
        new_shape[i] = shape_[d];
        new_strides[i] = strides_[d];
    }
    
    Tensor permuted = *this;
    permuted.shape_ = new_shape;
    permuted.strides_ = new_strides;
    
    permuted.grad_handle_ = std::make_shared<std::shared_ptr<Tensor>>(nullptr);
    permuted.grad_node = nullptr;
    
    return permuted;
}

Tensor Tensor::slice(size_t index) const {
    if (shape_.empty() || static_cast<size_t>(shape_[0]) <= index) {
        throw std::runtime_error("Slice index out of bounds on dimension 0.");
    }
    
    std::vector<int64_t> new_shape(shape_.begin() + 1, shape_.end());
    std::vector<int64_t> new_strides(strides_.begin() + 1, strides_.end());
    size_t new_offset = offset_ + index * strides_[0];

    Tensor view = *this;
    view.shape_ = new_shape;
    view.strides_ = new_strides;
    view.offset_ = new_offset;
    
    view.grad_handle_ = std::make_shared<std::shared_ptr<Tensor>>(nullptr);
    view.grad_node = nullptr;
    
    return view;
}

} // namespace vesper