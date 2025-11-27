#pragma once
#include <vesper/core/tensor.h>
#include <vector>

namespace vesper::ops {

/**
 * Concatenate tensors along an existing dimension.
 * 
 * Unlike stack() which adds a new dimension, cat() joins tensors along
 * an existing dimension. All tensors must have the same shape except
 * in the concatenation dimension.
 * 
 * Example:
 *   Tensor a = zeros({2, 3}, Float32, device);  // [2, 3]
 *   Tensor b = zeros({2, 5}, Float32, device);  // [2, 5]
 *   Tensor c = cat({a, b}, 1);                  // [2, 8]
 * 
 * @param tensors Vector of tensors to concatenate
 * @param dim Dimension along which to concatenate (default: 0)
 * @return Concatenated tensor
 */
Tensor cat(const std::vector<Tensor>& tensors, int dim = 0);

// GPU dispatch functions (internal use)
#ifdef VESPER_HIP_ENABLED
void cat_hip_dispatch(const std::vector<Tensor>& inputs, Tensor& output, int dim);
#endif

#ifdef VESPER_CUDA_ENABLED  
void cat_cuda_dispatch(const std::vector<Tensor>& inputs, Tensor& output, int dim);
#endif

} // namespace vesper::ops
