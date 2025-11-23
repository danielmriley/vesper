#pragma once

#include <vesper/core/tensor.h>

namespace vesper::autograd {

class Engine {
public:
    // Computes the gradient of the root tensor with respect to all leaf tensors
    // that have requires_grad=true.
    static void backward(Tensor& root, const Tensor& gradient = {});
};

} // namespace vesper::autograd
