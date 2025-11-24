#pragma once
#include <vesper/nn/module.h>
#include <vesper/core/tensor.h>
#include <vector>

namespace vesper::nn {

class LayerNorm : public Module {
public:
    LayerNorm(std::vector<int64_t> normalized_shape, float eps = 1e-5, bool elementwise_affine = true);
    Tensor forward(const Tensor& input) override;
    
    std::vector<int64_t> normalized_shape;
    float eps;
    bool elementwise_affine;
    Tensor weight;
    Tensor bias;
};

class RMSNorm : public Module {
public:
    RMSNorm(std::vector<int64_t> normalized_shape, float eps = 1e-5, bool elementwise_affine = true);
    Tensor forward(const Tensor& input) override;
    
    std::vector<int64_t> normalized_shape;
    float eps;
    bool elementwise_affine;
    Tensor weight;
};

} // namespace vesper::nn
