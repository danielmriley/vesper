#pragma once

#if defined(__HIPCC__) || defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace vesper::core {

// Helper for vectorized loads (128-bit)
// Assumes ptr is aligned to 16 bytes (or at least `sizeof(float4)`)
template <typename T>
__device__ inline T load_vectorized(const T* ptr) {
    return *ptr; 
}

// Specialization for float4
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
__device__ inline float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ inline void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}
#endif

} // namespace vesper::core
