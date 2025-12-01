#pragma once

#include <vesper/core/tensor.h>

namespace vesper::ops {

/**
 * @brief Fused RMSNorm + Linear projection
 * 
 * Computes: output = RMSNorm(input, norm_weight, eps) @ linear_weight.T + bias
 * 
 * This fusion saves one full read+write of the intermediate normalized tensor,
 * significantly reducing memory bandwidth requirements.
 * 
 * @param input Input tensor [batch*seq, dim_in] or [batch, seq, dim_in]
 * @param norm_weight RMSNorm weight [dim_in]
 * @param linear_weight Linear layer weight [dim_out, dim_in]
 * @param bias Linear layer bias [dim_out] (optional, can be undefined)
 * @param eps Epsilon for numerical stability in RMSNorm
 * @return Output tensor [batch*seq, dim_out] or matching input batch dims
 */
Tensor fused_rmsnorm_linear(
    const Tensor& input,
    const Tensor& norm_weight,
    const Tensor& linear_weight,
    const Tensor& bias,
    float eps = 1e-5f);

/**
 * @brief Fused RMSNorm + Linear backward pass
 * 
 * Computes gradients for all parameters of the fused operation.
 * 
 * @param grad_output Gradient from downstream [batch*seq, dim_out]
 * @param input Original input [batch*seq, dim_in]
 * @param norm_weight RMSNorm weight [dim_in]
 * @param linear_weight Linear layer weight [dim_out, dim_in]
 * @param eps Epsilon used in forward pass
 * @param grad_input Output: gradient for input
 * @param grad_norm_weight Output: gradient for norm_weight (accumulated)
 * @param grad_linear_weight Output: gradient for linear_weight (accumulated)
 * @param grad_bias Output: gradient for bias (accumulated)
 */
void fused_rmsnorm_linear_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& norm_weight,
    const Tensor& linear_weight,
    float eps,
    Tensor& grad_input,
    Tensor& grad_norm_weight,
    Tensor& grad_linear_weight,
    Tensor& grad_bias);

// HIP dispatch functions
void fused_rmsnorm_linear_hip_dispatch(
    const Tensor& input,
    const Tensor& norm_weight,
    const Tensor& linear_weight,
    const Tensor& bias,
    float eps,
    Tensor& output);

void fused_rmsnorm_linear_backward_hip_dispatch(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& norm_weight,
    const Tensor& linear_weight,
    float eps,
    Tensor& grad_input,
    Tensor& grad_norm_weight,
    Tensor& grad_linear_weight,
    Tensor& grad_bias);

} // namespace vesper::ops
