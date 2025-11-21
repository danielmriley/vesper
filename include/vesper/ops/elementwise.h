#pragma once

#include <vesper/core/tensor.h>

namespace vesper {
namespace ops {

    // Public API
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor sub(const Tensor& a, const Tensor& b);
    Tensor mul(const Tensor& a, const Tensor& b);
    Tensor mul(const Tensor& a, float b); // Scalar variant
    Tensor div(const Tensor& a, const Tensor& b);
    Tensor div(const Tensor& a, float b); // Scalar variant

    // Backend dispatchers
    void add_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out);
    void sub_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out);
    void mul_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out);
    void div_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out);

}
}
