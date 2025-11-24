#include <vesper/nn/linear.h>
#include <vesper/core/factories.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/elementwise.h> // For bias addition
#include <vesper/nn/init.h> // New include
#include <cmath>

namespace vesper::nn {

// If the header has default arguments, we only implement the full constructor.
// Assuming header has: Linear(..., bool use_bias = true, Device device = Device::CPU);

Linear::Linear(int64_t in_features, int64_t out_features, bool use_bias, Device device)
    : use_bias_(use_bias) {
    
    // 1. Initialize weight tensor with Kaiming initialization
    weight = empty({out_features, in_features}, DType::Float32, device);
    init::kaiming_uniform_(weight, std::sqrt(5.0f)); // a=sqrt(5) is PyTorch default for Linear
    register_parameter("weight", weight); 

    if (use_bias_) {
        // 2. Initialize bias tensor to zeros
        // PyTorch initializes bias with uniform(-bound, bound) where bound = 1/sqrt(fan_in)
        bias = empty({out_features}, DType::Float32, device);
        
        // Fan-in calculation
        int64_t fan_in = in_features;
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        init::uniform_(bias, -bound, bound);
        
        register_parameter("bias", bias);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // ... (rest unchanged) ...
    auto output = ops::matmul(input, weight.transpose(0, 1));

    if (use_bias_) {
        output = ops::add(output, bias);
    }
    return output;
}

} // namespace vesper::nn
