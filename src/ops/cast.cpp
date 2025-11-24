#include <vesper/ops/cast.h>
#include <vesper/autograd/node.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <stdexcept>

namespace vesper {
namespace ops {

// Forward declaration of CPU dispatch
void cast_cpu_dispatch(const Tensor& input, Tensor& output);
void cast_cuda_dispatch(const Tensor& input, Tensor& output);
void cast_hip_dispatch(const Tensor& input, Tensor& output);

Tensor cast(const Tensor& input, DType target_dtype) {
    if (input.dtype() == target_dtype) {
        return input;
    }

    // Create output tensor
    // Note: empty() creates a contiguous tensor.
    Tensor result = empty(input.shape(), target_dtype, input.device(), input.requires_grad());

    // Dispatch
    if (input.device() == Device::CPU) {
        cast_cpu_dispatch(input, result);
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        cast_cuda_dispatch(input, result);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        cast_hip_dispatch(input, result);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Cast not implemented for this device yet.");
    }

    // Autograd
    if (input.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (input.grad_node) {
            node->next_edges.push_back({input.grad_node});
        }
        
        // Capture input dtype for backward cast
        DType input_dtype = input.dtype();
        
        node->backward_fn = [input=input, result_weak=result.weak(), input_dtype]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            
            // Gradient of cast is cast of gradient
            // dy/dx = 1 (conceptually)
            // dx = dy
            // But we need to cast dy back to input type.
            
            Tensor grad_output = result->grad();
            Tensor grad_input = cast(grad_output, input_dtype);
            
            if (input.requires_grad()) {
                // Accumulate gradient
                input.grad() = ops::add(input.grad(), grad_input);
            }
        };
        
        result.grad_node = node;
    }

    return result;
}

} // namespace ops
} // namespace vesper
