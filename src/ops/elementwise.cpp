#include <vesper/ops/elementwise.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <stdexcept>

namespace vesper::ops {

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape() || a.device() != b.device()) {
        throw std::runtime_error("Tensor shapes or devices do not match for add operation.");
    }

    Tensor result = empty(a.shape(), a.dtype(), a.device());

    switch(a.device()) {
        case Device::HIP:
            add_hip_dispatch(a, b, result);
            break;
        default:
            throw std::runtime_error("Device not supported for add operation.");
    }

    return result;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape() || a.device() != b.device()) {
        throw std::runtime_error("Tensor shapes or devices do not match for sub operation.");
    }

    Tensor result = empty(a.shape(), a.dtype(), a.device());

    switch(a.device()) {
        case Device::HIP:
            sub_hip_dispatch(a, b, result);
            break;
        default:
            throw std::runtime_error("Device not supported for sub operation.");
    }

    return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape() || a.device() != b.device()) {
        throw std::runtime_error("Tensor shapes or devices do not match for mul operation.");
    }

    Tensor result = empty(a.shape(), a.dtype(), a.device());

    switch(a.device()) {
        case Device::HIP:
            mul_hip_dispatch(a, b, result);
            break;
        default:
            throw std::runtime_error("Device not supported for mul operation.");
    }

    return result;
}

Tensor mul(const Tensor& a, float b) {
    auto b_tensor = full(a.shape(), a.dtype(), a.device(), b);
    return mul(a, b_tensor);
}

} // namespace vesper::ops
