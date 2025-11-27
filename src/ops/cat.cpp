#include <vesper/ops/cat.h>
#include <vesper/core/factories.h>
#include <vesper/core/macros.h>

#include <algorithm>
#include <numeric>

namespace vesper::ops {

// CPU implementation of cat
static void cat_cpu(const std::vector<Tensor>& inputs, Tensor& output, int dim) {
    VESPER_CHECK(!inputs.empty(), "Cannot cat empty tensor list");
    
    int num_tensors = static_cast<int>(inputs.size());
    const auto& shape = inputs[0].shape();
    int ndim = static_cast<int>(shape.size());
    
    // Handle negative dimension
    if (dim < 0) dim += ndim;
    VESPER_CHECK(dim >= 0 && dim < ndim, "Invalid dimension for cat");
    
    // Calculate outer, inner sizes
    int64_t outer_size = 1;
    for (int d = 0; d < dim; ++d) {
        outer_size *= shape[d];
    }
    
    int64_t inner_size = 1;
    for (int d = dim + 1; d < ndim; ++d) {
        inner_size *= shape[d];
    }
    
    DType dtype = output.dtype();
    
    if (dtype == DType::Float32) {
        float* out_ptr = output.data_ptr<float>();
        int64_t total_cat_dim = output.shape()[dim];
        
        for (int64_t outer = 0; outer < outer_size; ++outer) {
            int64_t cat_offset = 0;
            for (int t = 0; t < num_tensors; ++t) {
                const float* in_ptr = inputs[t].data_ptr<float>();
                int64_t in_cat_size = inputs[t].shape()[dim];
                
                for (int64_t c = 0; c < in_cat_size; ++c) {
                    for (int64_t inner = 0; inner < inner_size; ++inner) {
                        int64_t out_idx = outer * total_cat_dim * inner_size + 
                                         (cat_offset + c) * inner_size + inner;
                        int64_t in_idx = outer * in_cat_size * inner_size + 
                                        c * inner_size + inner;
                        out_ptr[out_idx] = in_ptr[in_idx];
                    }
                }
                cat_offset += in_cat_size;
            }
        }
    } else if (dtype == DType::Int32) {
        int32_t* out_ptr = output.data_ptr<int32_t>();
        int64_t total_cat_dim = output.shape()[dim];
        
        for (int64_t outer = 0; outer < outer_size; ++outer) {
            int64_t cat_offset = 0;
            for (int t = 0; t < num_tensors; ++t) {
                const int32_t* in_ptr = inputs[t].data_ptr<int32_t>();
                int64_t in_cat_size = inputs[t].shape()[dim];
                
                for (int64_t c = 0; c < in_cat_size; ++c) {
                    for (int64_t inner = 0; inner < inner_size; ++inner) {
                        int64_t out_idx = outer * total_cat_dim * inner_size + 
                                         (cat_offset + c) * inner_size + inner;
                        int64_t in_idx = outer * in_cat_size * inner_size + 
                                        c * inner_size + inner;
                        out_ptr[out_idx] = in_ptr[in_idx];
                    }
                }
                cat_offset += in_cat_size;
            }
        }
    } else if (dtype == DType::Int64) {
        int64_t* out_ptr = output.data_ptr<int64_t>();
        int64_t total_cat_dim = output.shape()[dim];
        
        for (int64_t outer = 0; outer < outer_size; ++outer) {
            int64_t cat_offset = 0;
            for (int t = 0; t < num_tensors; ++t) {
                const int64_t* in_ptr = inputs[t].data_ptr<int64_t>();
                int64_t in_cat_size = inputs[t].shape()[dim];
                
                for (int64_t c = 0; c < in_cat_size; ++c) {
                    for (int64_t inner = 0; inner < inner_size; ++inner) {
                        int64_t out_idx = outer * total_cat_dim * inner_size + 
                                         (cat_offset + c) * inner_size + inner;
                        int64_t in_idx = outer * in_cat_size * inner_size + 
                                        c * inner_size + inner;
                        out_ptr[out_idx] = in_ptr[in_idx];
                    }
                }
                cat_offset += in_cat_size;
            }
        }
    } else {
        VESPER_CHECK(false, "Unsupported dtype for cat");
    }
}

Tensor cat(const std::vector<Tensor>& tensors, int dim) {
    VESPER_CHECK(!tensors.empty(), "Cannot cat empty tensor list");
    
    const auto& first_shape = tensors[0].shape();
    Device device = tensors[0].device();
    DType dtype = tensors[0].dtype();
    int ndim = static_cast<int>(first_shape.size());
    
    // Handle negative dimension
    if (dim < 0) dim += ndim;
    VESPER_CHECK(dim >= 0 && dim < ndim, "Invalid dimension " + std::to_string(dim) + 
                 " for tensor with " + std::to_string(ndim) + " dimensions");
    
    // Validate all tensors have compatible shapes
    int64_t total_dim_size = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        VESPER_CHECK(t.device() == device, "All tensors must be on same device");
        VESPER_CHECK(t.dtype() == dtype, "All tensors must have same dtype");
        VESPER_CHECK(static_cast<int>(t.shape().size()) == ndim, 
                     "All tensors must have same number of dimensions");
        
        // Check all dimensions except cat dimension match
        for (int d = 0; d < ndim; ++d) {
            if (d != dim) {
                VESPER_CHECK(t.shape()[d] == first_shape[d],
                    "Tensor " + std::to_string(i) + " has incompatible shape at dim " + 
                    std::to_string(d) + ": " + std::to_string(t.shape()[d]) + 
                    " vs " + std::to_string(first_shape[d]));
            }
        }
        
        total_dim_size += t.shape()[dim];
    }
    
    // Build output shape
    std::vector<int64_t> output_shape = first_shape;
    output_shape[dim] = total_dim_size;
    
    Tensor output = vesper::empty(output_shape, dtype, device);
    
    // Dispatch to appropriate backend
    if (device == Device::CPU) {
        cat_cpu(tensors, output, dim);
    }
#ifdef VESPER_HIP_ENABLED
    else if (device == Device::HIP) {
        cat_hip_dispatch(tensors, output, dim);
    }
#endif
#ifdef VESPER_CUDA_ENABLED
    else if (device == Device::CUDA) {
        cat_cuda_dispatch(tensors, output, dim);
    }
#endif
    else {
        VESPER_CHECK(false, "Unsupported device for cat operation");
    }
    
    return output;
}

} // namespace vesper::ops
