#pragma once
#include <vesper/nn/module.h>
#include <vesper/nn/functional.h>

namespace vesper::nn {

class MSELoss : public Module {
public:
    MSELoss() = default;

    // The forward pass takes a prediction and a target
    Tensor forward(const Tensor& y_pred, const Tensor& y_true) {
        return functional::mse_loss(y_pred, y_true);
    }
    
    // Override the base forward to guide the user
    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("MSELoss requires two inputs: prediction and target.");
    }
};

} // namespace vesper::nn
