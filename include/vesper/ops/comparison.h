#pragma once
#include <vesper/core/tensor.h>

namespace vesper::ops {
Tensor greater_than(const Tensor& a, float b);
void greater_than_hip_dispatch(const Tensor& a, float b, Tensor& out);
void greater_than_cuda_dispatch(const Tensor& a, float b, Tensor& out);
}
