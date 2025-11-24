#include <vesper/nn/normalization.h>
#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>

namespace vesper::nn {

LayerNorm::LayerNorm(std::vector<int64_t> normalized_shape, float eps, bool elementwise_affine)
    : normalized_shape(normalized_shape), eps(eps), elementwise_affine(elementwise_affine) {
    
    if (elementwise_affine) {
        weight = vesper::full(normalized_shape, DType::Float32, Device::CPU, 1.0f, true);
        bias = vesper::full(normalized_shape, DType::Float32, Device::CPU, 0.0f, true);
        register_parameter("weight", weight);
        register_parameter("bias", bias);
    }
}

Tensor LayerNorm::forward(const Tensor& input) {
    return functional::layer_norm(input, normalized_shape, weight, bias, eps);
}

RMSNorm::RMSNorm(std::vector<int64_t> normalized_shape, float eps, bool elementwise_affine)
    : normalized_shape(normalized_shape), eps(eps), elementwise_affine(elementwise_affine) {
    
    if (elementwise_affine) {
        weight = vesper::full(normalized_shape, DType::Float32, Device::CPU, 1.0f, true);
        register_parameter("weight", weight);
    }
}

Tensor RMSNorm::forward(const Tensor& input) {
    return functional::rms_norm(input, normalized_shape, weight, eps);
}

} // namespace vesper::nn
