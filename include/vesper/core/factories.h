#pragma once

#include <vesper/core/tensor.h>

namespace vesper {

// Creates a tensor with uninitialized data
Tensor empty(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad = false);

// Creates a tensor filled with zeros
Tensor zeros(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad = false);

// Creates a tensor filled with a scalar value
Tensor full(const std::vector<int64_t>& shape, DType dtype, Device device, float val, bool requires_grad = false);

// Creates a tensor filled with ones
Tensor ones(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad = false);

// Creates a tensor filled with random numbers from a normal distribution (mean=0, std=1)
Tensor randn(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad = false);

} // namespace vesper
