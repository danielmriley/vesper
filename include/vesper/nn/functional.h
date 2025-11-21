#pragma once
#include <vesper/core/tensor.h>

namespace vesper::nn::functional {

Tensor sigmoid(const Tensor& input);
void sigmoid_hip_dispatch(const Tensor& input, Tensor& output);
void sigmoid_cuda_dispatch(const Tensor& input, Tensor& output);

Tensor relu(const Tensor& input);
void relu_hip_dispatch(const Tensor& input, Tensor& output);
void relu_cuda_dispatch(const Tensor& input, Tensor& output);

Tensor mse_loss(const Tensor& y_pred, const Tensor& y_true);

}
