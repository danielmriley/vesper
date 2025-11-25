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

}
}
