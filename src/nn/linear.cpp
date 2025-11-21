#include <vesper/nn/linear.h>
#include <vesper/core/factories.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/elementwise.h> // For bias addition
#include <vesper/ops/random.h>
#include <cmath>

namespace vesper::nn {

// Helper function for Kaiming Uniform Initialization
void kaiming_uniform_init(Tensor& t, int64_t fan_in) {
    const float bound = std::sqrt(6.0f / fan_in);
    ops::uniform_(t, -bound, bound);
}

Linear::Linear(int64_t in_features, int64_t out_features, bool use_bias, Device device)
    : weight(empty({out_features, in_features}, DType::Float32, device)),
      bias(use_bias ? zeros({out_features}, DType::Float32, device) : zeros({0}, DType::Float32, device)),
      use_bias_(use_bias) {
    
    // 1. Initialize weight tensor with Kaiming initialization
    kaiming_uniform_init(weight, in_features);
    register_parameter("weight", weight); // Register it as a trainable parameter

    if (use_bias_) {
        register_parameter("bias", bias);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // 3. Compute the forward pass: y = x * W^T + b
    
    // Use gemm directly to avoid creating a transposed view that disconnects autograd
    // y = input * weight^T
    auto output = ops::gemm(input, weight, false, true);

    if (use_bias_) {
        // This requires broadcasting support for `ops::add`.
        // We will assume this is implemented for this chapter.
        // A simple implementation would expand `bias` to match `output`'s shape.
        output = ops::add(output, bias);
    }
    return output;
}

} // namespace vesper::nn
