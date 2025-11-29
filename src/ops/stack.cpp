#include <vesper/ops/stack.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/node.h>
#include <vesper/autograd/guard.h>
#include <stdexcept>

namespace vesper::ops {

Tensor stack(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::runtime_error("Cannot stack an empty list of tensors.");
    }
    
    if (dim != 0) {
        throw std::runtime_error("stack currently only supports dim=0.");
    }

    // 1. All tensors must have the same shape and device
    const auto& first_shape = tensors[0].shape();
    const auto& device = tensors[0].device();
    const auto& dtype = tensors[0].dtype();
    
    bool any_requires_grad = false;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].shape() != first_shape || tensors[i].device() != device) {
            throw std::runtime_error("All tensors in stack must have the same shape and device.");
        }
        if (tensors[i].requires_grad()) {
            any_requires_grad = true;
        }
    }

    // 2. Determine the output shape
    auto output_shape = first_shape;
    output_shape.insert(output_shape.begin() + dim, tensors.size());
    
    bool requires_grad = any_requires_grad && autograd::grad_mode_enabled;
    Tensor output = empty(output_shape, dtype, device, requires_grad);

    // 3. Copy data
    for (size_t i = 0; i < tensors.size(); ++i) {
        output.slice(i).copy_from(tensors[i]);
    }
    
    // 4. Setup autograd
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        
        // Add edges to all input tensors that require grad
        for (const auto& t : tensors) {
            if (t.requires_grad() && t.grad_node) {
                node->next_edges.push_back({t.grad_node});
            }
        }
        
        // Clone tensors for backward to ensure autograd safety
        // This prevents issues if the original tensors are modified in-place
        std::vector<Tensor> tensors_saved;
        tensors_saved.reserve(tensors.size());
        for (const auto& t : tensors) {
            tensors_saved.push_back(t.clone());
        }
        
        // Keep original tensors for grad accumulation
        std::vector<Tensor> tensors_orig = tensors;
        
        node->backward_fn = [tensors_saved, tensors_orig, dim, result_weak=output.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            Tensor& grad_output = result->grad();
            
            // grad_output shape: [N, ...original_shape...]
            // For each input tensor i, its gradient is grad_output.slice(i)
            for (size_t i = 0; i < tensors_orig.size(); ++i) {
                if (tensors_orig[i].requires_grad()) {
                    Tensor grad_i = grad_output.slice(i).contiguous();
                    tensors_orig[i].accumulate_grad(grad_i);
                }
            }
        };
        
        output.grad_node = node;
    }
    
    return output;
}

} // namespace vesper::ops
