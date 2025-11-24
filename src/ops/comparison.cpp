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

void equal_cpu_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    size_t n = a.numel();
    
    // Assuming contiguous and same shape for MVP
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = (a_ptr[i] == b_ptr[i]) ? 1.0f : 0.0f;
    }
}

Tensor equal(const Tensor& a, const Tensor& b) {
    // Handle scalar broadcast (b is scalar)
    if (b.numel() == 1) {
        Tensor result = empty(a.shape(), a.dtype(), a.device(), false);
        float b_val;
        b.copy_to_host(&b_val); // Sync!
        
        if (a.device() == Device::CPU) {
             const float* a_ptr = a.data_ptr<float>();
             float* out_ptr = result.data_ptr<float>();
             size_t n = a.numel();
             for(size_t i=0; i<n; ++i) {
                 out_ptr[i] = (a_ptr[i] == b_val) ? 1.0f : 0.0f;
             }
        } else {
             throw std::runtime_error("equal: GPU not implemented for scalar broadcast yet");
        }
        return result;
    }

    if (a.shape() != b.shape()) {
        // For now, require exact shape match (no broadcasting)
        throw std::runtime_error("equal: shape mismatch");
    }
    
    Tensor result = empty(a.shape(), a.dtype(), a.device(), false);
    
    if (a.device() == Device::CPU) {
        equal_cpu_dispatch(a, b, result);
    } else {
        throw std::runtime_error("equal: only CPU supported for now");
    }
    
    return result;
}

} // namespace vesper::ops
