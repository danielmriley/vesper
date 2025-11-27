#pragma once
#include <vesper/core/tensor.h>

namespace vesper::ops {

/**
 * @brief Gather values along an axis using indices
 * 
 * Gathers values along an axis specified by dim.
 * 
 * For a 3-D tensor the output is specified by:
 *   out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
 *   out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
 *   out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
 * 
 * @param input  The source tensor
 * @param dim    The axis along which to index
 * @param index  The indices of elements to gather (Int32 or Int64)
 * @return Tensor of same shape as index, containing gathered values
 * 
 * Example:
 *   Tensor input = randn({3, 4}, Float32, device);  // [3, 4]
 *   Tensor index = tensor({0, 2}, Int32, device).reshape({2, 1});  // [2, 1]
 *   Tensor result = gather(input, 0, index);  // [2, 4] - rows 0 and 2
 */
Tensor gather(const Tensor& input, int64_t dim, const Tensor& index);

/**
 * @brief Scatter values along an axis using indices
 * 
 * Writes all values from src into self at the indices specified in index.
 * This is the reverse operation of gather.
 * 
 * For a 3-D tensor, self is updated as:
 *   self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
 *   self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
 *   self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
 * 
 * @param input  The destination tensor (will be cloned and modified)
 * @param dim    The axis along which to index
 * @param index  The indices of elements to scatter (Int32 or Int64)
 * @param src    The source tensor with values to scatter
 * @return New tensor with scattered values
 */
Tensor scatter(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src);

/**
 * @brief In-place scatter with scalar value
 * 
 * @param input  The tensor to modify in place
 * @param dim    The axis along which to index
 * @param index  The indices of elements to scatter
 * @param value  The scalar value to scatter
 * @return Reference to the modified tensor
 */
Tensor& scatter_(Tensor& input, int64_t dim, const Tensor& index, float value);

/**
 * @brief Select elements from x or y based on condition
 * 
 * Returns a tensor of elements selected from either x or y, depending on condition.
 * 
 *   out[i] = x[i] if condition[i] else y[i]
 * 
 * Supports broadcasting between condition, x, and y.
 * 
 * @param condition  Boolean tensor (or float where 0.0 = false, non-zero = true)
 * @param x          Tensor of values to select where condition is true
 * @param y          Tensor of values to select where condition is false
 * @return Tensor with selected values
 * 
 * Example:
 *   Tensor cond = tensor({1, 0, 1, 0}, Float32, device);
 *   Tensor x = tensor({1, 2, 3, 4}, Float32, device);
 *   Tensor y = tensor({5, 6, 7, 8}, Float32, device);
 *   Tensor result = where(cond, x, y);  // {1, 6, 3, 8}
 */
Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);

/**
 * @brief Fill elements where mask is true with a scalar value
 * 
 * Creates a copy of input with elements set to value where mask is true.
 * 
 * @param input  The source tensor
 * @param mask   Boolean mask tensor (same shape as input or broadcastable)
 * @param value  The value to fill
 * @return Tensor with masked values filled
 * 
 * Example:
 *   Tensor input = randn({4, 4}, Float32, device);
 *   Tensor mask = greater_than(input, 0.0f);  // Mask positive values
 *   Tensor result = masked_fill(input, mask, 0.0f);  // Zero out positives
 */
Tensor masked_fill(const Tensor& input, const Tensor& mask, float value);

/**
 * @brief In-place version of masked_fill
 * 
 * @param input  The tensor to modify in place
 * @param mask   Boolean mask tensor
 * @param value  The value to fill
 * @return Reference to the modified tensor
 */
Tensor& masked_fill_(Tensor& input, const Tensor& mask, float value);

// Backend dispatchers - HIP
void gather_hip_dispatch(const Tensor& input, int64_t dim, const Tensor& index, Tensor& out);
void scatter_hip_dispatch(Tensor& out, int64_t dim, const Tensor& index, const Tensor& src);
void scatter_scalar_hip_dispatch(Tensor& out, int64_t dim, const Tensor& index, float value);
void where_hip_dispatch(const Tensor& condition, const Tensor& x, const Tensor& y, Tensor& out);
void masked_fill_hip_dispatch(Tensor& out, const Tensor& mask, float value);

// Backend dispatchers - CUDA
void gather_cuda_dispatch(const Tensor& input, int64_t dim, const Tensor& index, Tensor& out);
void scatter_cuda_dispatch(Tensor& out, int64_t dim, const Tensor& index, const Tensor& src);
void scatter_scalar_cuda_dispatch(Tensor& out, int64_t dim, const Tensor& index, float value);
void where_cuda_dispatch(const Tensor& condition, const Tensor& x, const Tensor& y, Tensor& out);
void masked_fill_cuda_dispatch(Tensor& out, const Tensor& mask, float value);

} // namespace vesper::ops
