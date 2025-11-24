#include <vesper/ops/im2col.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/node.h>
#include <stdexcept>

namespace vesper::ops {

Tensor im2col(const Tensor& input, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w) {
    if (input.shape().size() != 4) {
        throw std::runtime_error("im2col input must be 4D [B, C, H, W]");
    }
    
    int64_t B = input.shape()[0];
    int64_t C = input.shape()[1];
    int64_t H = input.shape()[2];
    int64_t W = input.shape()[3];
    
    int64_t out_h = (H + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t out_w = (W + 2 * padding_w - kernel_w) / stride_w + 1;
    
    // Output shape: [C * KH * KW, B * OH * OW]
    int64_t col_h = C * kernel_h * kernel_w;
    int64_t col_w = B * out_h * out_w;
    
    bool requires_grad = input.requires_grad();
    Tensor output = empty({col_h, col_w}, input.dtype(), input.device(), requires_grad);
    
    switch(input.device()) {
        case Device::CPU:
            im2col_cpu_dispatch(input, output, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
            break;
        case Device::CUDA:
#if USE_CUDA_BACKEND
            im2col_cuda_dispatch(input, output, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
#else
            throw std::runtime_error("CUDA backend not enabled");
#endif
            break;
        case Device::HIP:
#if USE_HIP_BACKEND
            im2col_hip_dispatch(input, output, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
#else
            throw std::runtime_error("HIP backend not enabled");
#endif
            break;
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        if (input.grad_node) {
            node->next_edges.push_back({input.grad_node});
        }
        
        // Capture shape and params
        std::vector<int64_t> input_shape = input.shape();
        
        // Explicit copy to ensure mutability in lambda
        Tensor input_copy = input;

        node->backward_fn = [input_copy, weak_out=output.weak(), input_shape, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w]() mutable {
            auto output = weak_out.lock();
            if (!output) return;
            
            if (input_copy.requires_grad()) {
                // output.grad() is [C*KH*KW, B*OH*OW]
                // col2im returns [B, C, H, W]
                Tensor grad_input = col2im(output->grad(), input_shape, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
                // input_copy is non-const here
                input_copy.accumulate_grad(grad_input);
            }
        };
        
        output.grad_node = node;
    }
    
    return output;
}

Tensor col2im(const Tensor& grad_col, const std::vector<int64_t>& input_shape, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w) {
    // grad_col is [C*KH*KW, B*OH*OW]
    // input_shape is [B, C, H, W]
    
    Tensor grad_img = zeros(input_shape, grad_col.dtype(), grad_col.device());
    
    switch(grad_col.device()) {
        case Device::CPU:
            col2im_cpu_dispatch(grad_col, grad_img, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
            break;
        case Device::CUDA:
#if USE_CUDA_BACKEND
            col2im_cuda_dispatch(grad_col, grad_img, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
#else
            throw std::runtime_error("CUDA backend not enabled");
#endif
            break;
        case Device::HIP:
#if USE_HIP_BACKEND
            col2im_hip_dispatch(grad_col, grad_img, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
#else
            throw std::runtime_error("HIP backend not enabled");
#endif
            break;
    }
    
    return grad_img;
}

} // namespace vesper::ops
