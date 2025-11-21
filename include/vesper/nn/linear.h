#pragma once

#include <vesper/nn/module.h>

namespace vesper::nn {

class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, bool use_bias = true);

    Tensor forward(const Tensor& input) override;

    // The parameters of the layer, owned by the class
    Tensor weight;
    Tensor bias;
private:
    bool use_bias_;
};

} // namespace vesper::nn
