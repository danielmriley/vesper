#include <vesper/ops/pooling.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/node.h>
#include <stdexcept>

namespace vesper::ops {

std::pair<Tensor, Tensor> max_pool2d(const Tensor& input, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w) {
    if (input.shape().size() != 4) {
        throw std::runtime_error("max_pool2d input must be 4D [B, C, H, W]");
    }
    
    int64_t B = input.shape()[0];
    int64_t C = input.shape()[1];
    int64_t H = input.shape()[2];
    int64_t W = input.shape()[3];
    
    int64_t out_h = (H + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t out_w = (W + 2 * padding_w - kernel_w) / stride_w + 1;
    
    bool requires_grad = input.requires_grad();
    Tensor output = empty({B, C, out_h, out_w}, input.dtype(), input.device(), requires_grad);
    // Indices tensor doesn't need grad, usually Int64 or Int32. Using Float32 for simplicity in kernels?
    // Standard is Int64 (Long).
    Tensor indices = empty({B, C, out_h, out_w}, DType::Int64, input.device(), false);
    
    switch(input.device()) {
        case Device::CPU:
            max_pool2d_cpu_dispatch(input, output, indices, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
            break;
        case Device::CUDA:
#if USE_CUDA_BACKEND
            max_pool2d_cuda_dispatch(input, output, indices, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
#else
            throw std::runtime_error("CUDA backend not enabled");
#endif
            break;
        case Device::HIP:
#if USE_HIP_BACKEND
            max_pool2d_hip_dispatch(input, output, indices, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
#else
            throw std::runtime_error("HIP backend not enabled");
#endif
            break;
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        if (input.grad_node) node->next_edges.push_back({input.grad_node});
        
        std::vector<int64_t> input_shape = input.shape();
        
        // Explicit copy to ensure mutability in lambda
        Tensor input_copy = input;
        
        // Capture 'indices' by value (it's a tensor, so shared storage)
        node->backward_fn = [input_copy, weak_out=output.weak(), indices, input_shape]() mutable {
            auto output = weak_out.lock();
            if (!output) return;
            if (input_copy.requires_grad()) {
                Tensor grad_input = max_pool2d_backward(output->grad(), indices, input_shape);
                input_copy.accumulate_grad(grad_input);
            }
        };
        output.grad_node = node;
    }
    
    return {output, indices};
}

Tensor max_pool2d_backward(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) {
    Tensor grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());
    
    switch(grad_output.device()) {
        case Device::CPU:
            max_pool2d_backward_cpu_dispatch(grad_output, indices, grad_input);
            break;
        case Device::CUDA:
#if USE_CUDA_BACKEND
            max_pool2d_backward_cuda_dispatch(grad_output, indices, grad_input);
#else
            throw std::runtime_error("CUDA backend not enabled");
#endif
            break;
        case Device::HIP:
#if USE_HIP_BACKEND
            max_pool2d_backward_hip_dispatch(grad_output, indices, grad_input);
#else
            throw std::runtime_error("HIP backend not enabled");
#endif
            break;
    }
    return grad_input;
}

} // namespace vesper::ops
