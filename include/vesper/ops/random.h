#pragma once

#include <vesper/core/tensor.h>

namespace vesper::ops {

// Fills the input tensor with random numbers from a uniform distribution [min, max]
// In-place operation (hence the underscore suffix convention)
void uniform_(Tensor& tensor, float min, float max);

// Backend dispatchers
void uniform_hip_dispatch(Tensor& tensor, float min, float max);

} // namespace vesper::ops
