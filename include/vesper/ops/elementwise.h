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

}
}
