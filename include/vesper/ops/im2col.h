#pragma once
#include <vesper/core/tensor.h>

namespace vesper::ops {

// Expands input image tensor into column vectors.
// Input: [Batch, Channels, Height, Width]
// Output: [Batch, Channels * KernelH * KernelW, OutputH * OutputW] (reshaped for matmul)
// Wait, standard im2col outputs 2D or 3D?
// Usually for batch GEMM: [Batch, Output_Size, Input_Size_Per_Patch]
// Or flattened: [Batch * OutputH * OutputW, Channels * KernelH * KernelW]
// Let's follow PyTorch/Caffe convention roughly but simplified.
// We want to do `Matmul(Weight, Col)`. Weight is [OutCh, InCh*KH*KW].
// So Col should be [InCh*KH*KW, Batch*OH*OW].
// Then Output = Weight * Col -> [OutCh, Batch*OH*OW].
// Finally reshape Output to [Batch, OutCh, OH, OW].
//
// Return shape: [Channels * KernelH * KernelW, Batch * OutputH * OutputW]
// (This is "im2col" followed by some permutation to prepare for GEMM)
//
// Actually, to support batching cleanly with our 2D GEMM, we often stack batches horizontally.
// Let's return: [Channels * KernelH * KernelW, Batch * OutputH * OutputW]
// 
// Arguments:
// input: [B, C, H, W]
// kernel_h, kernel_w
// stride_h, stride_w
// padding_h, padding_w
Tensor im2col(const Tensor& input, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w);

// The reverse operation for backward pass.
// Accumulates gradients from column format back to image format.
// grad_col: [Channels * KernelH * KernelW, Batch * OutputH * OutputW]
// input_shape: [B, C, H, W] (shape of the original image)
// Returns: [B, C, H, W]
Tensor col2im(const Tensor& grad_col, const std::vector<int64_t>& input_shape, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w);

// Backend dispatchers
void im2col_cpu_dispatch(const Tensor& input, Tensor& output, int kh, int kw, int sh, int sw, int ph, int pw);
void im2col_cuda_dispatch(const Tensor& input, Tensor& output, int kh, int kw, int sh, int sw, int ph, int pw);
void im2col_hip_dispatch(const Tensor& input, Tensor& output, int kh, int kw, int sh, int sw, int ph, int pw);

void col2im_cpu_dispatch(const Tensor& grad_col, Tensor& grad_img, int kh, int kw, int sh, int sw, int ph, int pw);
void col2im_cuda_dispatch(const Tensor& grad_col, Tensor& grad_img, int kh, int kw, int sh, int sw, int ph, int pw);
void col2im_hip_dispatch(const Tensor& grad_col, Tensor& grad_img, int kh, int kw, int sh, int sw, int ph, int pw);

} // namespace vesper::ops
