#pragma once

#include <vesper/core/tensor.h>

namespace vesper {
namespace ops {

    // Public API
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor add(const Tensor& a, float b); // Scalar variant
    Tensor sub(const Tensor& a, const Tensor& b);
    Tensor sub(const Tensor& a, float b); // Scalar variant
    Tensor mul(const Tensor& a, const Tensor& b);
    Tensor mul(const Tensor& a, float b); // Scalar variant
    Tensor div(const Tensor& a, const Tensor& b);
    Tensor div(const Tensor& a, float b); // Scalar variant

    // In-place variants
    Tensor& add_(Tensor& a, const Tensor& b);
    Tensor& add_(Tensor& a, float b);
    Tensor& sub_(Tensor& a, const Tensor& b);
    Tensor& sub_(Tensor& a, float b);
    Tensor& mul_(Tensor& a, float b);
    
    // Fused operations for optimizer efficiency
    // lerp_: self = self * (1 - weight) + other * weight  (linear interpolation in-place)
    Tensor& lerp_(Tensor& self, const Tensor& other, float weight);
    // addcmul_: self = self + tensor1 * tensor2 * value
    Tensor& addcmul_(Tensor& self, const Tensor& tensor1, const Tensor& tensor2, float value);
    // adam_update_: Fused Adam parameter update to avoid temporary allocations
    // param = param - lr * m_hat / (sqrt(v_hat) + eps)
    // where m_hat = exp_avg / correction1, v_hat = exp_avg_sq / correction2
    void adam_update_(Tensor& param, const Tensor& exp_avg, const Tensor& exp_avg_sq,
                      float lr, float correction1, float correction2, float eps);
    
    // Backend dispatchers for fused ops
    void lerp_hip_dispatch(Tensor& self, const Tensor& other, float weight);
    void lerp_cuda_dispatch(Tensor& self, const Tensor& other, float weight);
    void addcmul_hip_dispatch(Tensor& self, const Tensor& t1, const Tensor& t2, float value);
    void addcmul_cuda_dispatch(Tensor& self, const Tensor& t1, const Tensor& t2, float value);
    void adam_update_hip_dispatch(Tensor& param, const Tensor& m, const Tensor& v,
                                   float lr, float c1, float c2, float eps);
    void adam_update_cuda_dispatch(Tensor& param, const Tensor& m, const Tensor& v,
                                    float lr, float c1, float c2, float eps);

    // Backend dispatchers
    void add_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
    void add_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out);
    void add_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
    void add_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out);

    void sub_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
    void sub_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out);
    void sub_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
    void sub_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out);

    void mul_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
    void mul_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out);
    void mul_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
    void mul_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out);

    void div_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
    void div_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out);
    void div_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
    void div_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out);

    // Unary Ops
    Tensor sqrt(const Tensor& a);
    Tensor sign(const Tensor& a);
    Tensor gelu(const Tensor& a); // Approximate tanh
    Tensor gelu_erf(const Tensor& a); // Exact erf
    
    // SiLU (Swish) activation: silu(x) = x * sigmoid(x)
    Tensor silu(const Tensor& a);
    void silu_(Tensor& a);  // In-place version
    
    // Fused SiLU * Multiply: silu_mul(gate, up) = silu(gate) * up
    // Key operation in SwiGLU FFN - saves memory bandwidth by fusing two ops
    Tensor silu_mul(const Tensor& gate, const Tensor& up);
    
    Tensor exp(const Tensor& a);
    Tensor log(const Tensor& a);
    Tensor cos(const Tensor& a);
    Tensor sin(const Tensor& a);
    
    void sqrt_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
    void sqrt_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
    
    void sign_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
    void sign_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);

    void gelu_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
    void gelu_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);

    void gelu_erf_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
    void gelu_erf_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);

    void silu_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
    void silu_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
    void silu_inplace_hip_dispatch(Tensor& a);
    void silu_inplace_cuda_dispatch(Tensor& a);
    
    // Fused silu_mul dispatch
    void silu_mul_hip_dispatch(const Tensor& gate, const Tensor& up, Tensor& out);
    void silu_mul_cuda_dispatch(const Tensor& gate, const Tensor& up, Tensor& out);
    void silu_mul_backward_hip_dispatch(const Tensor& grad_out, const Tensor& gate, const Tensor& up,
                                         Tensor& grad_gate, Tensor& grad_up);
    void silu_mul_backward_cuda_dispatch(const Tensor& grad_out, const Tensor& gate, const Tensor& up,
                                          Tensor& grad_gate, Tensor& grad_up);

    void exp_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
    void exp_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);

    void log_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
    void log_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);

    void cos_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
    void cos_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);

    void sin_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
    void sin_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);

    void gelu_backward_cpu_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input);
    void gelu_backward_cuda_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input);
    void gelu_backward_hip_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input);

    void gelu_erf_backward_cpu_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input);
    void gelu_erf_backward_cuda_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input);
    void gelu_erf_backward_hip_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input);

    void silu_backward_cpu_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input);
    void silu_backward_cuda_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input);
    void silu_backward_hip_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input);

}
}
