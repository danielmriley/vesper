/**
 * @file attention_ops.cpp
 * @brief Implementation of attention-related tensor operations
 * 
 * Chapter 33.5: Grouped Query Attention (GQA)
 */

#include <vesper/ops/attention_ops.h>
#include <vesper/ops/elementwise.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/node.h>
#include <vesper/autograd/guard.h>
#include <stdexcept>

namespace vesper::ops {

// Forward declaration of HIP dispatchers (defined in .hip file)
#ifdef USE_HIP_BACKEND
void repeat_kv_hip_dispatch(const Tensor& input, Tensor& output, int64_t n_rep);
void repeat_kv_backward_hip_dispatch(const Tensor& grad_output, Tensor& grad_input, int64_t n_rep);
#endif

// Forward declaration of CUDA dispatchers (defined in .cu file)
#ifdef USE_CUDA_BACKEND
void repeat_kv_cuda_dispatch(const Tensor& input, Tensor& output, int64_t n_rep);
void repeat_kv_backward_cuda_dispatch(const Tensor& grad_output, Tensor& grad_input, int64_t n_rep);
#endif

// CPU dispatch for repeat_kv
void repeat_kv_cpu_dispatch(const Tensor& input, Tensor& output, int64_t n_rep) {
    // input: [B, KV_H, S, D]
    // output: [B, KV_H * n_rep, S, D]
    
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    
    int64_t B = input.shape()[0];
    int64_t KV_H = input.shape()[1];
    int64_t S = input.shape()[2];
    int64_t D = input.shape()[3];
    
    int64_t Q_H = KV_H * n_rep;
    
    // For each output element, copy from the corresponding KV head
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t q_h = 0; q_h < Q_H; ++q_h) {
            int64_t kv_h = q_h / n_rep;  // Map query head to KV head
            for (int64_t s = 0; s < S; ++s) {
                for (int64_t d = 0; d < D; ++d) {
                    int64_t in_idx = ((b * KV_H + kv_h) * S + s) * D + d;
                    int64_t out_idx = ((b * Q_H + q_h) * S + s) * D + d;
                    out_ptr[out_idx] = in_ptr[in_idx];
                }
            }
        }
    }
}

// CPU dispatch for repeat_kv backward
void repeat_kv_backward_cpu_dispatch(const Tensor& grad_output, Tensor& grad_input, int64_t n_rep) {
    // grad_output: [B, Q_H, S, D]
    // grad_input: [B, KV_H, S, D]
    
    const float* grad_out_ptr = grad_output.data_ptr<float>();
    float* grad_in_ptr = grad_input.data_ptr<float>();
    
    int64_t B = grad_output.shape()[0];
    int64_t Q_H = grad_output.shape()[1];
    int64_t S = grad_output.shape()[2];
    int64_t D = grad_output.shape()[3];
    
    int64_t KV_H = Q_H / n_rep;
    
    // Zero the output first
    std::fill(grad_in_ptr, grad_in_ptr + grad_input.numel(), 0.0f);
    
    // Sum gradients from all query heads that share each KV head
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t q_h = 0; q_h < Q_H; ++q_h) {
            int64_t kv_h = q_h / n_rep;
            for (int64_t s = 0; s < S; ++s) {
                for (int64_t d = 0; d < D; ++d) {
                    int64_t out_idx = ((b * Q_H + q_h) * S + s) * D + d;
                    int64_t in_idx = ((b * KV_H + kv_h) * S + s) * D + d;
                    grad_in_ptr[in_idx] += grad_out_ptr[out_idx];
                }
            }
        }
    }
}

Tensor repeat_kv(const Tensor& x, int64_t n_rep) {
    if (n_rep == 1) {
        return x;  // No repetition needed (MHA case)
    }
    
    if (x.ndim() != 4) {
        throw std::runtime_error("repeat_kv: input must be 4D [Batch, KV_Heads, SeqLen, HeadDim]");
    }
    
    int64_t B = x.shape()[0];
    int64_t KV_H = x.shape()[1];
    int64_t S = x.shape()[2];
    int64_t D = x.shape()[3];
    int64_t Q_H = KV_H * n_rep;
    
    bool requires_grad = x.requires_grad() && autograd::grad_mode_enabled;
    Tensor output = vesper::empty({B, Q_H, S, D}, x.dtype(), x.device(), requires_grad);
    
    switch (x.device()) {
        case Device::CPU:
            repeat_kv_cpu_dispatch(x, output, n_rep);
            break;
        case Device::HIP:
#ifdef USE_HIP_BACKEND
            repeat_kv_hip_dispatch(x, output, n_rep);
#else
            throw std::runtime_error("repeat_kv: HIP backend not enabled");
#endif
            break;
        case Device::CUDA:
#ifdef USE_CUDA_BACKEND
            repeat_kv_cuda_dispatch(x, output, n_rep);
#else
            throw std::runtime_error("repeat_kv: CUDA backend not enabled");
#endif
            break;
        default:
            throw std::runtime_error("repeat_kv: unsupported device");
    }
    
    // Handle autograd
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        if (x.grad_node) {
            node->next_edges.push_back({x.grad_node});
        }
        
        // Capture needed values
        node->backward_fn = [x=x, n_rep, result_weak=output.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            Tensor& grad_output = result->grad();
            
            if (x.requires_grad()) {
                Tensor grad_input = repeat_kv_backward(grad_output, n_rep);
                x.accumulate_grad(grad_input);
            }
        };
        
        output.grad_node = node;
    }
    
    return output;
}

Tensor repeat_kv_backward(const Tensor& grad_output, int64_t n_rep) {
    if (n_rep == 1) {
        return grad_output;
    }
    
    if (grad_output.ndim() != 4) {
        throw std::runtime_error("repeat_kv_backward: grad_output must be 4D");
    }
    
    int64_t B = grad_output.shape()[0];
    int64_t Q_H = grad_output.shape()[1];
    int64_t S = grad_output.shape()[2];
    int64_t D = grad_output.shape()[3];
    int64_t KV_H = Q_H / n_rep;
    
    Tensor grad_input = vesper::zeros({B, KV_H, S, D}, grad_output.dtype(), grad_output.device());
    
    switch (grad_output.device()) {
        case Device::CPU:
            repeat_kv_backward_cpu_dispatch(grad_output, grad_input, n_rep);
            break;
        case Device::HIP:
#ifdef USE_HIP_BACKEND
            repeat_kv_backward_hip_dispatch(grad_output, grad_input, n_rep);
#else
            throw std::runtime_error("repeat_kv_backward: HIP backend not enabled");
#endif
            break;
        case Device::CUDA:
#ifdef USE_CUDA_BACKEND
            repeat_kv_backward_cuda_dispatch(grad_output, grad_input, n_rep);
#else
            throw std::runtime_error("repeat_kv_backward: CUDA backend not enabled");
#endif
            break;
        default:
            throw std::runtime_error("repeat_kv_backward: unsupported device");
    }
    
    return grad_input;
}

} // namespace vesper::ops
