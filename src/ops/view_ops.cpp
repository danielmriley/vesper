#include <vesper/ops/view_ops.h>
#include <vesper/autograd/node.h>
#include <vesper/core/factories.h> // for zeros
#include <vesper/ops/elementwise.h> // for add (accumulate grad)
#include <stdexcept>
#include <numeric>

namespace vesper::ops {

// Helper to compute new strides for a view if possible
std::vector<int64_t> compute_view_strides(const std::vector<int64_t>& old_shape, 
                                          const std::vector<int64_t>& old_strides, 
                                          const std::vector<int64_t>& new_shape) {
    int64_t numel = 1;
    for(auto s : old_shape) numel *= s;
    int64_t new_numel = 1;
    for(auto s : new_shape) new_numel *= s;
    
    if (numel != new_numel) return {};

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
            new_strides[i] = 1; 
            continue;
        }

        if (old_idx >= old_dims.size()) return {}; 

        int64_t d_size = old_dims[old_idx].first;
        int64_t d_stride = old_dims[old_idx].second;

        if (d_size == new_shape[i]) {
            new_strides[i] = d_stride;
            old_idx++;
        } else if (d_size > new_shape[i]) {
            if (d_size % new_shape[i] != 0) return {};
            int64_t inner_size = d_size / new_shape[i];
            new_strides[i] = d_stride * inner_size;
            old_dims[old_idx].first = inner_size;
        } else {
            int64_t target = new_shape[i];
            int64_t current = d_size;
            int64_t current_stride = d_stride;
            
            while (current < target) {
                old_idx++;
                if (old_idx >= old_dims.size()) return {};
                
                int64_t next_size = old_dims[old_idx].first;
                int64_t next_stride = old_dims[old_idx].second;
                
                if (current_stride != next_size * next_stride) return {};
                
                current *= next_size;
                current_stride = next_stride;
            }
            
            if (current != target) return {};
            new_strides[i] = old_dims[old_idx].second;
            old_idx++;
        }
    }
    
    if (old_idx != old_dims.size()) return {};

    return new_strides;
}

Tensor view(const Tensor& input, const std::vector<int64_t>& new_shape) {
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
        if (input.numel() % new_numel != 0) throw std::runtime_error("Invalid shape for view (inferred dimension mismatch)");
        resolved_shape[inferred_dim_idx] = input.numel() / new_numel;
    } else {
        if (input.numel() != new_numel) throw std::runtime_error("View size mismatch");
    }

    std::vector<int64_t> new_strides = compute_view_strides(input.shape(), input.strides(), resolved_shape);
    if (new_strides.empty()) {
        throw std::runtime_error("View cannot be created with given shape (not contiguous in memory). Use reshape() or contiguous() first.");
    }

    // Create the view tensor
    // We use the private constructor via friend access
    Tensor result(input.storage_, input.dtype_, resolved_shape, new_strides, input.offset_, input.requires_grad_);
    
    // Setup Autograd
    if (input.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (input.grad_node) node->next_edges.push_back({input.grad_node});
        
        // Backward: grad_input = grad_output.view(input.shape)
        node->backward_fn = [input_shape=input.shape(), input=input, result_weak=result.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            if (input.requires_grad()) {
                // We need to reshape the gradient of the result back to the input shape
                // result.grad() has shape 'resolved_shape'
                // We want to view it as 'input_shape'
                // Since view is just metadata, we can view the gradient tensor.
                
                // Note: We use ops::view (or Tensor::view) here.
                // But we must be careful about infinite recursion if we use ops::view inside backward?
                // No, because we are operating on gradients which are Tensors.
                // But we want to accumulate into input.grad().
                
                Tensor grad_view = result->grad().view(input_shape);
                input.accumulate_grad(grad_view);
            }
        };
        result.grad_node = node;
    }
    
    return result;
}

Tensor reshape(const Tensor& input, const std::vector<int64_t>& shape) {
    try {
        return view(input, shape);
    } catch (...) {
        return view(input.contiguous(), shape);
    }
}

Tensor transpose(const Tensor& input, int64_t dim0, int64_t dim1) {
    if (dim0 < 0 || dim0 >= static_cast<int64_t>(input.shape().size()) || 
        dim1 < 0 || dim1 >= static_cast<int64_t>(input.shape().size())) {
        throw std::runtime_error("Transpose dimensions are out of bounds.");
    }
    
    auto new_shape = input.shape();
    std::swap(new_shape[dim0], new_shape[dim1]);

    auto new_strides = input.strides();
    std::swap(new_strides[dim0], new_strides[dim1]);

    Tensor result(input.storage_, input.dtype_, new_shape, new_strides, input.offset_, input.requires_grad_);

    if (input.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (input.grad_node) node->next_edges.push_back({input.grad_node});
        
        node->backward_fn = [dim0, dim1, input=input, result_weak=result.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            if (input.requires_grad()) {
                // Backward: transpose back
                Tensor grad_transposed = result->grad().transpose(dim0, dim1);
                input.accumulate_grad(grad_transposed);
            }
        };
        result.grad_node = node;
    }

    return result;
}

Tensor permute(const Tensor& input, const std::vector<int64_t>& dims) {
    if (dims.size() != input.shape().size()) {
        throw std::runtime_error("Permute dims must match tensor rank");
    }
    
    std::vector<int64_t> new_shape(dims.size());
    std::vector<int64_t> new_strides(dims.size());
    
    for (size_t i = 0; i < dims.size(); ++i) {
        new_shape[i] = input.shape()[dims[i]];
        new_strides[i] = input.strides()[dims[i]];
    }
    
    Tensor result(input.storage_, input.dtype_, new_shape, new_strides, input.offset_, input.requires_grad_);
    
    if (input.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (input.grad_node) node->next_edges.push_back({input.grad_node});
        
        // Compute inverse permutation for backward
        std::vector<int64_t> inv_dims(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            inv_dims[dims[i]] = i;
        }
        
        node->backward_fn = [inv_dims, input=input, result_weak=result.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            if (input.requires_grad()) {
                Tensor grad_permuted = result->grad().permute(inv_dims);
                input.accumulate_grad(grad_permuted);
            }
        };
        result.grad_node = node;
    }
    
    return result;
}

Tensor slice(const Tensor& input, size_t index) {
    if (input.shape().empty()) throw std::runtime_error("Cannot slice a scalar");
    if (index >= input.shape()[0]) throw std::runtime_error("Slice index out of bounds");
    
    std::vector<int64_t> new_shape(input.shape().begin() + 1, input.shape().end());
    std::vector<int64_t> new_strides(input.strides().begin() + 1, input.strides().end());
    
    size_t new_offset = input.offset_ + index * input.strides()[0];
    
    Tensor result(input.storage_, input.dtype_, new_shape, new_strides, new_offset, input.requires_grad_);
    
    if (input.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (input.grad_node) node->next_edges.push_back({input.grad_node});
        
        node->backward_fn = [index, input=input, result_weak=result.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            if (input.requires_grad()) {
                // Backward for slice:
                // We need to create a zero tensor of input shape, and place result.grad() at index.
                // Since we don't have scatter/index_put yet, we can try to construct it.
                // Or we can use a hack: create zeros, slice it, add grad, then accumulate.
                // Wait, slicing a zero tensor creates a view. If we modify the view, we modify the zero tensor.
                // Yes!
                
                Tensor grad_input = zeros(input.shape(), input.dtype(), input.device(), false);
                Tensor grad_slice = grad_input.slice(index);
                
                // We need to copy result.grad() into grad_slice.
                // We don't have a copy_ operator exposed easily, but we can use add (since it's zeros).
                // grad_slice += result.grad()
                // But we need in-place add or copy.
                // Tensor::copy_from works!
                
                grad_slice.copy_from(result->grad());
                
                input.accumulate_grad(grad_input);
            }
        };
        result.grad_node = node;
    }
    
    return result;
}

} // namespace vesper::ops
