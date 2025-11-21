#pragma once
#include <vesper/core/tensor.h>

namespace vesper::nn::functional {

Tensor sigmoid(const Tensor& input);
void sigmoid_hip_dispatch(const Tensor& input, Tensor& output);

Tensor relu(const Tensor& input);
void relu_hip_dispatch(const Tensor& input, Tensor& output);
// ... other activations will be added here ...

}
