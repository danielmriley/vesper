#include <vesper/ops/elementwise.h>
#include <vesper/core/tensor.h>

namespace vesper::ops {

void add_cpu_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    auto* a_ptr = a.data_ptr<float>();
    auto* b_ptr = b.data_ptr<float>();
    auto* out_ptr = out.data_ptr<float>();
    
    // Handle broadcasting if necessary
    // For now, assuming simple broadcasting where b is scalar or 1D matching last dim
    // But the generic dispatch in elementwise.cpp handles the logic?
    // elementwise.cpp:
    // if (b.shape().size() == 1 && a.shape().size() > 0 && b.shape()[0] == a.shape().back()) { broadcast_b = true; }
    // But here we just get Tensors.
    // We need to know if we are broadcasting.
    
    // Let's check shapes.
    bool broadcast_b = (b.numel() != a.numel());
    size_t n = a.numel();
    size_t b_numel = b.numel();

    for(size_t i = 0; i < n; ++i) {
        float val_b = broadcast_b ? b_ptr[i % b_numel] : b_ptr[i];
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}

void sub_cpu_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    auto* a_ptr = a.data_ptr<float>();
    auto* b_ptr = b.data_ptr<float>();
    auto* out_ptr = out.data_ptr<float>();
    size_t n = a.numel();
    // sub currently enforces exact shape match in elementwise.cpp
    for(size_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] - b_ptr[i];
    }
}

void mul_cpu_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    auto* a_ptr = a.data_ptr<float>();
    auto* b_ptr = b.data_ptr<float>();
    auto* out_ptr = out.data_ptr<float>();
    size_t n = a.numel();
    // mul currently enforces exact shape match in elementwise.cpp
    for(size_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] * b_ptr[i];
    }
}

void div_cpu_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    auto* a_ptr = a.data_ptr<float>();
    auto* b_ptr = b.data_ptr<float>();
    auto* out_ptr = out.data_ptr<float>();
    size_t n = a.numel();
    // div currently enforces exact shape match in elementwise.cpp
    for(size_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] / b_ptr[i];
    }
}

} // namespace vesper::ops
