#include <vesper/nn/linear.h>
#include <vesper/core/factories.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/elementwise.h> // For bias addition
#include <random>
#include <cmath>

namespace vesper::nn {

// Helper function for Kaiming Uniform Initialization
void kaiming_uniform_init(Tensor& t, int64_t fan_in) {
    const float bound = std::sqrt(6.0f / fan_in);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-bound, bound);

    std::vector<float> data(t.numel());
    for (float& val : data) {
        val = dist(rng);
    }
    t.copy_from_host(data.data());
}

Linear::Linear(int64_t in_features, int64_t out_features, bool use_bias)
    : weight(empty({out_features, in_features}, DType::Float32, Device::CPU)),
      bias(use_bias ? zeros({out_features}, DType::Float32, Device::CPU) : zeros({0}, DType::Float32, Device::CPU)),
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
    
    // The transpose() call creates a non-contiguous view. Our matmul from Ch 7
    // was updated to handle this by calling .contiguous() internally.
    auto output = ops::matmul(input, weight.transpose(0, 1));

    if (use_bias_) {
        // This requires broadcasting support for `ops::add`.
        // We will assume this is implemented for this chapter.
        // A simple implementation would expand `bias` to match `output`'s shape.
        output = ops::add(output, bias);
    }
    return output;
}

} // namespace vesper::nn
