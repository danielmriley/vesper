#pragma once
#include <vesper/core/tensor.h>
#include <utility>

namespace vesper::ops {

// Max Pooling 2D
// Input: [B, C, H, W]
// Returns a pair {Output, Indices}
// Output: [B, C, OutH, OutW]
// Indices: [B, C, OutH, OutW] (int64_t indices of max elements for backward pass)
std::pair<Tensor, Tensor> max_pool2d(const Tensor& input, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w);

// Backward pass for Max Pooling
// grad_output: [B, C, OutH, OutW]
// indices: [B, C, OutH, OutW] (from forward pass)
// input_shape: shape of original input [B, C, H, W]
// Returns: grad_input [B, C, H, W]
Tensor max_pool2d_backward(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape);

// Backend dispatchers
void max_pool2d_cpu_dispatch(const Tensor& input, Tensor& output, Tensor& indices, int kh, int kw, int sh, int sw, int ph, int pw);
void max_pool2d_cuda_dispatch(const Tensor& input, Tensor& output, Tensor& indices, int kh, int kw, int sh, int sw, int ph, int pw);
void max_pool2d_hip_dispatch(const Tensor& input, Tensor& output, Tensor& indices, int kh, int kw, int sh, int sw, int ph, int pw);

void max_pool2d_backward_cpu_dispatch(const Tensor& grad_output, const Tensor& indices, Tensor& grad_input);
void max_pool2d_backward_cuda_dispatch(const Tensor& grad_output, const Tensor& indices, Tensor& grad_input);
void max_pool2d_backward_hip_dispatch(const Tensor& grad_output, const Tensor& indices, Tensor& grad_input);

} // namespace vesper::ops
