#pragma once
#include <vesper/nn/module.h>
#include <vesper/nn/functional.h>

namespace vesper::nn {

class Sigmoid : public Module {
public:
    Tensor forward(const Tensor& input) override {
        return functional::sigmoid(input);
    }
};

class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override {
        return functional::relu(input);
    }
};

} // namespace vesper::nn
