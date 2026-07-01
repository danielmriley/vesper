#pragma once

#include <vesper/core/tensor.h>
#include <cstdint>
#include <random>
#include <mutex>

namespace vesper::ops {

// ============================================================================
// Random Number Generator Seeding
// ============================================================================

// Set the seed for the global random number generator for reproducibility.
// This affects all subsequent random operations (uniform_, normal_, etc.)
void manual_seed(uint64_t seed);

// Accessors for the process-wide RNG used by random fills, sampling, and text
// generation. manual_seed() controls this engine, so routing every random draw
// through it makes the whole pipeline reproducible.
//
// Thread-safety: callers MUST hold get_global_rng_mutex() (e.g. via a
// std::lock_guard) for as long as they use the returned engine reference.
std::mt19937& get_global_rng();
std::mutex& get_global_rng_mutex();

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
