#pragma once
#include <vesper/core/tensor.h>

namespace vesper::ops {
Tensor greater_than(const Tensor& a, float b);
Tensor equal(const Tensor& a, const Tensor& b);

// Dispatch functions
void greater_than_hip_dispatch(const Tensor& a, float b, Tensor& out);
void greater_than_cuda_dispatch(const Tensor& a, float b, Tensor& out);
void equal_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out);
void equal_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& out);
void equal_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out);
void equal_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out);
}
