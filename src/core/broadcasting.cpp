#include <vesper/core/broadcasting.h>

namespace vesper {

std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape_a, const std::vector<int64_t>& shape_b) {
    std::vector<int64_t> out_shape;
    
    size_t dims_a = shape_a.size();
    size_t dims_b = shape_b.size();
    size_t max_dims = std::max(dims_a, dims_b);
    
    out_shape.resize(max_dims);
    
    // Align shapes to the right (i.e., match last dimensions first)
    for (size_t i = 0; i < max_dims; ++i) {
        int64_t dim_a = 1;
        int64_t dim_b = 1;
        
        if (i < dims_a) {
            dim_a = shape_a[dims_a - 1 - i];
        }
        if (i < dims_b) {
            dim_b = shape_b[dims_b - 1 - i];
        }
        
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::runtime_error("Shapes are not broadcastable: " + 
                                     std::to_string(dim_a) + " vs " + std::to_string(dim_b));
        }
        
        out_shape[max_dims - 1 - i] = std::max(dim_a, dim_b);
    }
    
    return out_shape;
}

std::vector<int64_t> compute_broadcast_strides(const std::vector<int64_t>& tensor_shape, 
                                               const std::vector<int64_t>& tensor_strides, 
                                               const std::vector<int64_t>& target_shape) {
    std::vector<int64_t> new_strides(target_shape.size());
    
    size_t tensor_dims = tensor_shape.size();
    size_t target_dims = target_shape.size();
    size_t dim_offset = target_dims - tensor_dims;
    
    for (size_t i = 0; i < target_dims; ++i) {
        if (i < dim_offset) {
            // This dimension is prepended (e.g., broadcasting [3] to [2, 3])
            // The stride is effectively 0 because iterating this new dimension 
            // doesn't move the pointer in the original tensor.
            new_strides[i] = 0; 
        } else {
            size_t tensor_idx = i - dim_offset;
            int64_t tensor_dim = tensor_shape[tensor_idx];
            int64_t target_dim = target_shape[i];
            
            if (tensor_dim == target_dim) {
                new_strides[i] = tensor_strides[tensor_idx];
            } else if (tensor_dim == 1) {
                // Dimension expanded from 1 to N
                // Stride is 0 because we repeat the same value
                new_strides[i] = 0;
            } else {
                // This should have been caught by broadcast_shapes
                throw std::runtime_error("Logic error in compute_broadcast_strides");
            }
        }
    }
    
    return new_strides;
}

} // namespace vesper
