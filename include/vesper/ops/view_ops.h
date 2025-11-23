#pragma once
#include <vesper/core/tensor.h>
#include <vector>

namespace vesper::ops {

Tensor view(const Tensor& input, const std::vector<int64_t>& shape);
Tensor reshape(const Tensor& input, const std::vector<int64_t>& shape);
Tensor transpose(const Tensor& input, int64_t dim0, int64_t dim1);
Tensor permute(const Tensor& input, const std::vector<int64_t>& dims);
Tensor slice(const Tensor& input, size_t index);

} // namespace vesper::ops
