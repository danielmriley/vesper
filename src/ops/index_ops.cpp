#include <vesper/ops/index_ops.h>
#include <vesper/core/factories.h>
#include <vesper/core/stream.h>
#include <stdexcept>
#include <cstring>

namespace vesper::ops {

// ============================================================================
// Helper: Compute linear index for strided tensor
// ============================================================================
static inline int64_t compute_linear_index(
    const std::vector<int64_t>& indices,
    const std::vector<int64_t>& strides
) {
    int64_t idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        idx += indices[i] * strides[i];
    }
    return idx;
}

// ============================================================================
// CPU: gather
// ============================================================================
static void gather_cpu_dispatch(const Tensor& input, int64_t dim, const Tensor& index, Tensor& out) {
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    int64_t ndim = static_cast<int64_t>(index.shape().size());
    int64_t numel = index.numel();
    
    // Get shapes and strides
    const auto& in_shape = input.shape();
    const auto& idx_shape = index.shape();
    const auto& in_strides = input.strides();
    const auto& out_strides = out.strides();
    
    // Iterate over all elements in index tensor
    std::vector<int64_t> coord(ndim, 0);
    
    auto get_index_value = [&](int64_t linear_idx) -> int64_t {
        if (index.dtype() == DType::Int32) {
            return static_cast<int64_t>(index.data_ptr<int32_t>()[linear_idx]);
        } else if (index.dtype() == DType::Int64) {
            return index.data_ptr<int64_t>()[linear_idx];
        } else {
            return static_cast<int64_t>(index.data_ptr<float>()[linear_idx]);
        }
    };
    
    for (int64_t i = 0; i < numel; ++i) {
        // Compute multi-dimensional coordinate from linear index
        int64_t tmp = i;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            coord[d] = tmp % idx_shape[d];
            tmp /= idx_shape[d];
        }
        
        // Get the index value at this coordinate
        int64_t idx_val = get_index_value(i);
        
        // Bounds check
        if (idx_val < 0 || idx_val >= in_shape[dim]) {
            throw std::runtime_error("gather: index " + std::to_string(idx_val) + 
                " is out of bounds for dimension " + std::to_string(dim) + 
                " with size " + std::to_string(in_shape[dim]));
        }
        
        // Build input coordinate (replace coord[dim] with idx_val)
        std::vector<int64_t> in_coord = coord;
        in_coord[dim] = idx_val;
        
        // Compute linear indices
        int64_t in_linear = compute_linear_index(in_coord, in_strides);
        int64_t out_linear = compute_linear_index(coord, out_strides);
        
        out_ptr[out_linear] = in_ptr[in_linear];
    }
}

// ============================================================================
// CPU: scatter
// ============================================================================
static void scatter_cpu_dispatch(Tensor& out, int64_t dim, const Tensor& index, const Tensor& src) {
    float* out_ptr = out.data_ptr<float>();
    const float* src_ptr = src.data_ptr<float>();
    
    int64_t ndim = static_cast<int64_t>(index.shape().size());
    int64_t numel = index.numel();
    
    const auto& out_shape = out.shape();
    const auto& idx_shape = index.shape();
    const auto& out_strides = out.strides();
    const auto& src_strides = src.strides();
    
    std::vector<int64_t> coord(ndim, 0);
    
    auto get_index_value = [&](int64_t linear_idx) -> int64_t {
        if (index.dtype() == DType::Int32) {
            return static_cast<int64_t>(index.data_ptr<int32_t>()[linear_idx]);
        } else if (index.dtype() == DType::Int64) {
            return index.data_ptr<int64_t>()[linear_idx];
        } else {
            return static_cast<int64_t>(index.data_ptr<float>()[linear_idx]);
        }
    };
    
    for (int64_t i = 0; i < numel; ++i) {
        // Compute coordinate
        int64_t tmp = i;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            coord[d] = tmp % idx_shape[d];
            tmp /= idx_shape[d];
        }
        
        int64_t idx_val = get_index_value(i);
        
        if (idx_val < 0 || idx_val >= out_shape[dim]) {
            throw std::runtime_error("scatter: index out of bounds");
        }
        
        // Build output coordinate
        std::vector<int64_t> out_coord = coord;
        out_coord[dim] = idx_val;
        
        int64_t out_linear = compute_linear_index(out_coord, out_strides);
        int64_t src_linear = compute_linear_index(coord, src_strides);
        
        out_ptr[out_linear] = src_ptr[src_linear];
    }
}

static void scatter_scalar_cpu_dispatch(Tensor& out, int64_t dim, const Tensor& index, float value) {
    float* out_ptr = out.data_ptr<float>();
    
    int64_t ndim = static_cast<int64_t>(index.shape().size());
    int64_t numel = index.numel();
    
    const auto& out_shape = out.shape();
    const auto& idx_shape = index.shape();
    const auto& out_strides = out.strides();
    
    std::vector<int64_t> coord(ndim, 0);
    
    auto get_index_value = [&](int64_t linear_idx) -> int64_t {
        if (index.dtype() == DType::Int32) {
            return static_cast<int64_t>(index.data_ptr<int32_t>()[linear_idx]);
        } else if (index.dtype() == DType::Int64) {
            return index.data_ptr<int64_t>()[linear_idx];
        } else {
            return static_cast<int64_t>(index.data_ptr<float>()[linear_idx]);
        }
    };
    
    for (int64_t i = 0; i < numel; ++i) {
        int64_t tmp = i;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            coord[d] = tmp % idx_shape[d];
            tmp /= idx_shape[d];
        }
        
        int64_t idx_val = get_index_value(i);
        
        if (idx_val < 0 || idx_val >= out_shape[dim]) {
            throw std::runtime_error("scatter: index out of bounds");
        }
        
        std::vector<int64_t> out_coord = coord;
        out_coord[dim] = idx_val;
        
        int64_t out_linear = compute_linear_index(out_coord, out_strides);
        out_ptr[out_linear] = value;
    }
}

// ============================================================================
// CPU: where
// ============================================================================
static void where_cpu_dispatch(const Tensor& condition, const Tensor& x, const Tensor& y, Tensor& out) {
    const float* x_ptr = x.data_ptr<float>();
    const float* y_ptr = y.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    size_t n = out.numel();
    
    // Handle different condition dtypes
    if (condition.dtype() == DType::Int32) {
        const int32_t* cond_ptr = condition.data_ptr<int32_t>();
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = (cond_ptr[i] != 0) ? x_ptr[i] : y_ptr[i];
        }
    } else if (condition.dtype() == DType::Int64) {
        const int64_t* cond_ptr = condition.data_ptr<int64_t>();
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = (cond_ptr[i] != 0) ? x_ptr[i] : y_ptr[i];
        }
    } else {
        // Float - treat non-zero as true
        const float* cond_ptr = condition.data_ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = (cond_ptr[i] != 0.0f) ? x_ptr[i] : y_ptr[i];
        }
    }
}

// ============================================================================
// CPU: masked_fill
// ============================================================================
static void masked_fill_cpu_dispatch(Tensor& out, const Tensor& mask, float value) {
    float* out_ptr = out.data_ptr<float>();
    
    size_t n = out.numel();
    
    // Handle different mask dtypes
    if (mask.dtype() == DType::Int32) {
        const int32_t* mask_ptr = mask.data_ptr<int32_t>();
        for (size_t i = 0; i < n; ++i) {
            if (mask_ptr[i] != 0) {
                out_ptr[i] = value;
            }
        }
    } else if (mask.dtype() == DType::Int64) {
        const int64_t* mask_ptr = mask.data_ptr<int64_t>();
        for (size_t i = 0; i < n; ++i) {
            if (mask_ptr[i] != 0) {
                out_ptr[i] = value;
            }
        }
    } else {
        // Float - treat non-zero as true
        const float* mask_ptr = mask.data_ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            if (mask_ptr[i] != 0.0f) {
                out_ptr[i] = value;
            }
        }
    }
}

// ============================================================================
// Public API: gather
// ============================================================================
Tensor gather(const Tensor& input, int64_t dim, const Tensor& index) {
    int64_t ndim = static_cast<int64_t>(input.shape().size());
    
    // Normalize negative dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("gather: dimension out of range");
    }
    
    // Validate shapes
    if (index.shape().size() != input.shape().size()) {
        throw std::runtime_error("gather: index must have same number of dimensions as input");
    }
    
    // All dimensions except dim must match between index and input (or be broadcastable)
    // For simplicity, we require exact match except along dim
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim && index.shape()[d] > input.shape()[d]) {
            throw std::runtime_error("gather: index size along dimension " + 
                std::to_string(d) + " exceeds input size");
        }
    }
    
    // Output has same shape as index
    Tensor result = empty(index.shape(), input.dtype(), input.device(), 
                          input.requires_grad());
    
    if (input.device() == Device::CPU) {
        gather_cpu_dispatch(input, dim, index, result);
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        gather_hip_dispatch(input, dim, index, result);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        gather_cuda_dispatch(input, dim, index, result);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for gather.");
    }
    
    // Autograd support
    if (input.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        node->next_edges.push_back({input.grad_node});
        
        node->backward_fn = [input_copy=input, dim, index, result_weak=result.weak()]() mutable {
            auto result_ptr = result_weak.lock();
            if (!result_ptr) return;
            
            if (input_copy.requires_grad()) {
                // Backward of gather is scatter_add
                Tensor grad_input = zeros(input_copy.shape(), input_copy.dtype(), input_copy.device());
                Tensor grad_output = result_ptr->grad();
                
                // Scatter add: grad_input.scatter_add_(dim, index, grad_output)
                // For now, implement CPU fallback
                if (grad_input.device() == Device::CPU) {
                    float* gi_ptr = grad_input.data_ptr<float>();
                    const float* go_ptr = grad_output.data_ptr<float>();
                    
                    int64_t ndim = static_cast<int64_t>(index.shape().size());
                    int64_t numel = index.numel();
                    const auto& gi_shape = grad_input.shape();
                    const auto& idx_shape = index.shape();
                    const auto& gi_strides = grad_input.strides();
                    const auto& go_strides = grad_output.strides();
                    
                    std::vector<int64_t> coord(ndim, 0);
                    
                    auto get_idx = [&](int64_t li) -> int64_t {
                        if (index.dtype() == DType::Int32) {
                            return static_cast<int64_t>(index.data_ptr<int32_t>()[li]);
                        } else if (index.dtype() == DType::Int64) {
                            return index.data_ptr<int64_t>()[li];
                        }
                        return static_cast<int64_t>(index.data_ptr<float>()[li]);
                    };
                    
                    for (int64_t i = 0; i < numel; ++i) {
                        int64_t tmp = i;
                        for (int64_t d = ndim - 1; d >= 0; --d) {
                            coord[d] = tmp % idx_shape[d];
                            tmp /= idx_shape[d];
                        }
                        
                        int64_t idx_val = get_idx(i);
                        std::vector<int64_t> gi_coord = coord;
                        gi_coord[dim] = idx_val;
                        
                        int64_t gi_linear = 0;
                        for (size_t d = 0; d < coord.size(); ++d) {
                            gi_linear += gi_coord[d] * gi_strides[d];
                        }
                        int64_t go_linear = 0;
                        for (size_t d = 0; d < coord.size(); ++d) {
                            go_linear += coord[d] * go_strides[d];
                        }
                        
                        gi_ptr[gi_linear] += go_ptr[go_linear];
                    }
                }
                // TODO: GPU backward
                
                input_copy.accumulate_grad(grad_input);
            }
        };
        result.grad_node = node;
    }
    
    return result;
}

// ============================================================================
// Public API: scatter
// ============================================================================
Tensor scatter(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) {
    int64_t ndim = static_cast<int64_t>(input.shape().size());
    
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("scatter: dimension out of range");
    }
    
    // Clone input to create output
    Tensor result = input.clone();
    
    if (input.device() == Device::CPU) {
        scatter_cpu_dispatch(result, dim, index, src);
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        scatter_hip_dispatch(result, dim, index, src);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        scatter_cuda_dispatch(result, dim, index, src);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for scatter.");
    }
    
    return result;
}

// ============================================================================
// Public API: scatter_ (in-place)
// ============================================================================
Tensor& scatter_(Tensor& input, int64_t dim, const Tensor& index, float value) {
    int64_t ndim = static_cast<int64_t>(input.shape().size());
    
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("scatter_: dimension out of range");
    }
    
    if (input.device() == Device::CPU) {
        scatter_scalar_cpu_dispatch(input, dim, index, value);
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        scatter_scalar_hip_dispatch(input, dim, index, value);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        scatter_scalar_cuda_dispatch(input, dim, index, value);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for scatter_.");
    }
    
    return input;
}

// ============================================================================
// Public API: where
// ============================================================================
Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y) {
    // Validate shapes match (no broadcasting for now)
    if (condition.shape() != x.shape() || x.shape() != y.shape()) {
        throw std::runtime_error("where: all input tensors must have the same shape (broadcasting not yet supported)");
    }
    
    if (condition.device() != x.device() || x.device() != y.device()) {
        throw std::runtime_error("where: all input tensors must be on the same device");
    }
    
    Tensor result = empty(x.shape(), x.dtype(), x.device(), 
                          x.requires_grad() || y.requires_grad());
    
    if (x.device() == Device::CPU) {
        where_cpu_dispatch(condition, x, y, result);
    } else if (x.device() == Device::HIP) {
#if USE_HIP_BACKEND
        where_hip_dispatch(condition, x, y, result);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (x.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        where_cuda_dispatch(condition, x, y, result);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for where.");
    }
    
    // Autograd for where
    if (x.requires_grad() || y.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (x.requires_grad()) node->next_edges.push_back({x.grad_node});
        if (y.requires_grad()) node->next_edges.push_back({y.grad_node});
        
        node->backward_fn = [condition, x_copy=x, y_copy=y, result_weak=result.weak()]() mutable {
            auto result_ptr = result_weak.lock();
            if (!result_ptr) return;
            
            Tensor grad_out = result_ptr->grad();
            
            // grad_x = where(condition, grad_out, 0)
            // grad_y = where(condition, 0, grad_out)
            if (x_copy.requires_grad()) {
                Tensor grad_x = vesper::ops::where(condition, grad_out, 
                    zeros(grad_out.shape(), grad_out.dtype(), grad_out.device()));
                x_copy.accumulate_grad(grad_x);
            }
            if (y_copy.requires_grad()) {
                Tensor grad_y = vesper::ops::where(condition,
                    zeros(grad_out.shape(), grad_out.dtype(), grad_out.device()), grad_out);
                y_copy.accumulate_grad(grad_y);
            }
        };
        result.grad_node = node;
    }
    
    return result;
}

// ============================================================================
// Public API: masked_fill
// ============================================================================
Tensor masked_fill(const Tensor& input, const Tensor& mask, float value) {
    if (input.shape() != mask.shape()) {
        throw std::runtime_error("masked_fill: input and mask must have the same shape");
    }
    
    if (input.device() != mask.device()) {
        throw std::runtime_error("masked_fill: input and mask must be on the same device");
    }
    
    // Clone input
    Tensor result = input.clone();
    
    if (input.device() == Device::CPU) {
        masked_fill_cpu_dispatch(result, mask, value);
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        masked_fill_hip_dispatch(result, mask, value);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        masked_fill_cuda_dispatch(result, mask, value);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for masked_fill.");
    }
    
    // Autograd: gradient is zero where mask is true
    if (input.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        node->next_edges.push_back({input.grad_node});
        
        node->backward_fn = [input_copy=input, mask, result_weak=result.weak()]() mutable {
            auto result_ptr = result_weak.lock();
            if (!result_ptr) return;
            
            if (input_copy.requires_grad()) {
                Tensor grad_out = result_ptr->grad();
                // Zero out gradients where mask was true
                Tensor grad_input = grad_out.clone();
                
                if (grad_input.device() == Device::CPU) {
                    float* gi_ptr = grad_input.data_ptr<float>();
                    const float* mask_ptr = mask.data_ptr<float>();
                    for (size_t i = 0; i < grad_input.numel(); ++i) {
                        if (mask_ptr[i] != 0.0f) {
                            gi_ptr[i] = 0.0f;
                        }
                    }
                }
                // TODO: GPU backward
                
                input_copy.accumulate_grad(grad_input);
            }
        };
        result.grad_node = node;
    }
    
    return result;
}

// ============================================================================
// Public API: masked_fill_ (in-place)
// ============================================================================
Tensor& masked_fill_(Tensor& input, const Tensor& mask, float value) {
    if (input.shape() != mask.shape()) {
        throw std::runtime_error("masked_fill_: input and mask must have the same shape");
    }
    
    if (input.device() != mask.device()) {
        throw std::runtime_error("masked_fill_: input and mask must be on the same device");
    }
    
    if (input.device() == Device::CPU) {
        masked_fill_cpu_dispatch(input, mask, value);
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        masked_fill_hip_dispatch(input, mask, value);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        masked_fill_cuda_dispatch(input, mask, value);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for masked_fill_.");
    }
    
    return input;
}

} // namespace vesper::ops
