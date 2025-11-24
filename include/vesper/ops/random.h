#pragma once

#include <vesper/core/tensor.h>

namespace vesper::ops {

// Fills the input tensor with random numbers from a uniform distribution [min, max]
// In-place operation (hence the underscore suffix convention)
void uniform_(Tensor& tensor, float min, float max);

// Fills the input tensor with random numbers from a normal distribution (mean, std)
void normal_(Tensor& tensor, float mean, float std);

// Backend dispatchers
void uniform_hip_dispatch(Tensor& tensor, float min, float max);
void uniform_cuda_dispatch(Tensor& tensor, float min, float max);

void normal_hip_dispatch(Tensor& tensor, float mean, float std);
void normal_cuda_dispatch(Tensor& tensor, float mean, float std);

} // namespace vesper::ops
