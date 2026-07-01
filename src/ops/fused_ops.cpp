/**
 * @file fused_ops.cpp
 * @brief CPU implementations and autograd wrappers for fused operations
 */

#include <vesper/ops/fused_ops.h>
#include <vesper/ops/normalization.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/node.h>
#include <vesper/autograd/guard.h>
#include <stdexcept>

namespace vesper::ops {

// Declare HIP dispatch functions
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

// CPU fallback: just do the unfused operations
static Tensor fused_rmsnorm_linear_cpu(
    const Tensor& input,
    const Tensor& norm_weight,
    const Tensor& linear_weight,
    const Tensor& bias,
    float eps)
{
    // Step 1: RMSNorm
    std::vector<int64_t> normalized_shape = {input.shape().back()};
    Tensor normalized = rms_norm(input, normalized_shape, norm_weight, eps);
    
    // Step 2: Linear projection (matmul + bias)
    // input @ weight.T + bias
    // normalized: [B, S, D_in], weight: [D_out, D_in]
    // Result: [B, S, D_out]
    Tensor output = matmul(normalized, linear_weight.transpose(-2, -1));
    
    if (bias.defined()) {
        output = add(output, bias);
    }
    
    return output;
}

Tensor fused_rmsnorm_linear(
    const Tensor& input,
    const Tensor& norm_weight,
    const Tensor& linear_weight,
    const Tensor& bias,
    float eps)
{
    // Determine output shape
    auto in_shape = input.shape();
    std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end() - 1);
    out_shape.push_back(linear_weight.shape()[0]);  // dim_out
    
    // Check if we need gradients
    bool requires_grad = (input.requires_grad() || 
                         (norm_weight.defined() && norm_weight.requires_grad()) ||
                         linear_weight.requires_grad() ||
                         (bias.defined() && bias.requires_grad())) && 
                         autograd::grad_mode_enabled;
    
    Tensor output;
    
    if (input.device() == Device::HIP) {
        // Use fused GPU kernel
        output = vesper::empty(out_shape, input.dtype(), input.device(), requires_grad);
#if USE_HIP_BACKEND
        fused_rmsnorm_linear_hip_dispatch(input, norm_weight, linear_weight, bias, eps, output);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (input.device() == Device::CUDA) {
        // TODO: CUDA implementation - use CPU fallback for now
        output = fused_rmsnorm_linear_cpu(input, norm_weight, linear_weight, bias, eps);
        output.set_requires_grad(requires_grad);
    } else {
        // CPU fallback
        output = fused_rmsnorm_linear_cpu(input, norm_weight, linear_weight, bias, eps);
        output.set_requires_grad(requires_grad);
    }
    
    // Set up autograd.
    // CPU/CUDA fall back to fused_rmsnorm_linear_cpu, whose output already carries the
    // correct grad_node assembled from the real sub-ops (rms_norm -> matmul -> add), so the
    // unfused backward is exact. Only the fused HIP kernel produces an output with no
    // sub-op graph, so it is the only path that needs an explicit (stub) backward node.
    if (requires_grad && input.device() == Device::HIP) {
        auto node = std::make_shared<autograd::Node>();
        
        // Collect input edges
        if (input.requires_grad() && input.grad_node) {
            node->next_edges.push_back({input.grad_node});
        }
        if (norm_weight.defined() && norm_weight.requires_grad() && norm_weight.grad_node) {
            node->next_edges.push_back({norm_weight.grad_node});
        }
        if (linear_weight.requires_grad() && linear_weight.grad_node) {
            node->next_edges.push_back({linear_weight.grad_node});
        }
        if (bias.defined() && bias.requires_grad() && bias.grad_node) {
            node->next_edges.push_back({bias.grad_node});
        }
        
        // Capture inputs for backward
        node->backward_fn = [input=input, norm_weight=norm_weight, linear_weight=linear_weight, 
                            bias=bias, eps, result_weak=output.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            
            Tensor& grad_output = result->grad();
            
            // For CPU or as fallback: use unfused backward
            // Step 1: Recompute normalized input (needed for backward)
            std::vector<int64_t> normalized_shape = {input.shape().back()};
            Tensor normalized = rms_norm(input, normalized_shape, norm_weight, eps);
            
            // Step 2: Backward through linear
            // grad_normalized = grad_output @ linear_weight  
            // (grad_output: [..., D_out], weight: [D_out, D_in] -> [..., D_in])
            Tensor grad_normalized = matmul(grad_output, linear_weight);
            
            // Reshape for weight gradient computation
            auto shape = grad_output.shape();
            int64_t batch_seq = 1;
            for (size_t i = 0; i < shape.size() - 1; ++i) {
                batch_seq *= shape[i];
            }
            int64_t d_out = shape.back();
            int64_t d_in = linear_weight.shape()[1];
            
            // grad_linear_weight = grad_output.T @ normalized
            Tensor grad_output_2d = grad_output.reshape({batch_seq, d_out});
            Tensor normalized_2d = normalized.reshape({batch_seq, d_in});
            Tensor grad_linear_weight = matmul(grad_output_2d.transpose(0, 1), normalized_2d);
            
            // grad_bias = sum(grad_output, dim=0...-1)
            if (bias.defined() && bias.requires_grad()) {
                Tensor grad_bias = sum(grad_output_2d, 0, false);
                bias.accumulate_grad(grad_bias);
            }
            
            // Accumulate grad for linear_weight
            if (linear_weight.requires_grad()) {
                linear_weight.accumulate_grad(grad_linear_weight);
            }
            
            // Step 3: Backward through RMSNorm (simplified)
            // For proper implementation, we'd need the full RMSNorm backward
            // For now, propagate gradient scaled by inv_rms
            if (input.requires_grad()) {
                // Simplified: just pass through the gradient (proper impl needs RMSNorm backward)
                Tensor grad_input = grad_normalized.reshape(input.shape());
                input.accumulate_grad(grad_input);
            }
            
            // grad_norm_weight (simplified)
            if (norm_weight.defined() && norm_weight.requires_grad()) {
                // Would need proper accumulation
                Tensor grad_norm_weight = vesper::zeros(norm_weight.shape(), norm_weight.dtype(), 
                                                        norm_weight.device(), false);
                norm_weight.accumulate_grad(grad_norm_weight);
            }
        };
        
        output.grad_node = node;
    }
    
    return output;
}

void fused_rmsnorm_linear_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& norm_weight,
    const Tensor& linear_weight,
    float eps,
    Tensor& grad_input,
    Tensor& grad_norm_weight,
    Tensor& grad_linear_weight,
    Tensor& grad_bias)
{
    if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        fused_rmsnorm_linear_backward_hip_dispatch(
            grad_output, input, norm_weight, linear_weight, eps,
            grad_input, grad_norm_weight, grad_linear_weight, grad_bias);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("fused_rmsnorm_linear_backward: CPU implementation not available");
    }
}

} // namespace vesper::ops
