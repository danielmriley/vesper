#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace vesper {

// Computes the broadcasted shape of two tensors.
// Throws std::runtime_error if shapes are not compatible.
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape_a, const std::vector<int64_t>& shape_b);

// Computes the strides required to view a tensor as if it were broadcasted to `target_shape`.
// Returns a vector of strides of the same length as `target_shape`.
// Assumes `tensor_shape` is broadcastable to `target_shape`.
std::vector<int64_t> compute_broadcast_strides(const std::vector<int64_t>& tensor_shape, 
                                               const std::vector<int64_t>& tensor_strides, 
                                               const std::vector<int64_t>& target_shape);

} // namespace vesper
