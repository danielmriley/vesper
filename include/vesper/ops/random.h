#pragma once

#include <vesper/core/tensor.h>
#include <cstdint>

namespace vesper::ops {

// ============================================================================
// Random Number Generator Seeding
// ============================================================================

// Set the seed for the global random number generator for reproducibility.
// This affects all subsequent random operations (uniform_, normal_, etc.)
void manual_seed(uint64_t seed);

// ============================================================================
// Random Fill Operations
// ============================================================================

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
