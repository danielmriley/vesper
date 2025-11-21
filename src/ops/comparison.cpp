#include <vesper/ops/comparison.h>
#include <vesper/core/factories.h>
#include <stdexcept>

namespace vesper::ops {

void greater_than_cpu_dispatch(const Tensor& a, float b, Tensor& out) {
    const float* in_ptr = a.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    size_t n = a.numel();
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = (in_ptr[i] > b) ? 1.0f : 0.0f;
    }
}

Tensor greater_than(const Tensor& a, float b) {
    // This op does not support autograd, so `requires_grad` is false.
    Tensor result = empty(a.shape(), a.dtype(), a.device(), false);
    
    if (a.device() == Device::CPU) {
        greater_than_cpu_dispatch(a, b, result);
    } else if (a.device() == Device::HIP) {
#if USE_HIP_BACKEND
        greater_than_hip_dispatch(a, b, result);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (a.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        greater_than_cuda_dispatch(a, b, result);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for greater_than.");
    }
    
    return result;
}

} // namespace vesper::ops
