#include "vesper/ops/reduction.h"
#include "vesper/core/factories.h"
#include "vesper/autograd/node.h"
#include "vesper/ops/elementwise.h"
#include <stdexcept>

namespace vesper {
namespace ops {

Tensor sum(const Tensor& input) {
    // 1. Check constraints
    if (input.dtype() != DType::Float32) {
        throw std::runtime_error("sum only supports Float32 for now.");
    }
    if (!input.is_contiguous()) {
        throw std::runtime_error("sum only supports contiguous tensors for now.");
    }

    // 2. Create output tensor (scalar)
    // We use empty shape {} for scalar.
    bool requires_grad = input.requires_grad();
    Tensor output = vesper::empty({}, input.dtype(), input.device(), requires_grad);

    // 3. Dispatch
    if (input.device() == Device::CPU) {
        // Simple CPU implementation for testing/fallback
        const float* in_ptr = input.data_ptr<float>();
        float* out_ptr = output.data_ptr<float>();
        float sum_val = 0.0f;
        size_t n = input.numel();
        for (size_t i = 0; i < n; ++i) {
            sum_val += in_ptr[i];
        }
        *out_ptr = sum_val;
    } else if (input.device() == Device::HIP) {
        sum_hip_dispatch(input, output);
    } else {
        throw std::runtime_error("Unknown device type.");
    }

    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        if (input.requires_grad() && input.grad_node) {
            node->next_edges.push_back({input.grad_node});
        }
        
        node->backward_fn = [input=input, output=output]() mutable {
            if (input.requires_grad()) {
                Tensor& grad_output = output.grad();
                // We need to broadcast grad_output (scalar) to input.shape()
                // Since we don't have broadcast ops, we'll read the scalar value.
                // Note: This causes a sync!
                float grad_val = 0.0f;
                grad_output.copy_to_host(&grad_val);
                
                Tensor grad_input_contrib = full(input.shape(), input.dtype(), input.device(), grad_val);
                input.grad() = ops::add(input.grad(), grad_input_contrib);
            }
        };
        output.grad_node = node;
    }

    return output;
}

Tensor mean(const Tensor& input) {
    Tensor sum_val = sum(input);
    float n = static_cast<float>(input.numel());
    return div(sum_val, n);
}

} // namespace ops
} // namespace vesper
