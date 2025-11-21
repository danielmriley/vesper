#pragma once

#include <vesper/core/tensor.h>

namespace vesper::reference {

// Element-wise operations
void add(const Tensor& a, const Tensor& b, Tensor& out);
void sub(const Tensor& a, const Tensor& b, Tensor& out);
void mul(const Tensor& a, const Tensor& b, Tensor& out);

// Reduction operations
void sum(const Tensor& input, Tensor& out);

// Matrix multiplication
void gemm(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB);

} // namespace vesper::reference
