#pragma once

#include <vesper/core/tensor.h>

namespace vesper {
namespace ops {

Tensor cast(const Tensor& input, DType target_dtype);

} // namespace ops
} // namespace vesper
