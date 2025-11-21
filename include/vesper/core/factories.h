#pragma once

#include <vesper/core/tensor.h>

namespace vesper {

// Creates a tensor with uninitialized data
Tensor empty(const std::vector<int64_t>& shape, DType dtype, Device device);

// Creates a tensor filled with zeros
Tensor zeros(const std::vector<int64_t>& shape, DType dtype, Device device);

// Creates a tensor filled with a scalar value
Tensor full(const std::vector<int64_t>& shape, DType dtype, Device device, float val);

} // namespace vesper
