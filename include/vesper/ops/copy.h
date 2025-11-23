#pragma once
#include <vesper/core/tensor.h>

namespace vesper::ops {

// Copies data from src to dst.
// dst must be contiguous. src can be strided.
// Both must be on the same device.
void copy_strided(const Tensor& src, Tensor& dst);

// Backend dispatchers
void copy_strided_hip_dispatch(const Tensor& src, Tensor& dst);
void copy_strided_cuda_dispatch(const Tensor& src, Tensor& dst);
void copy_strided_cpu_dispatch(const Tensor& src, Tensor& dst);

} // namespace vesper::ops
