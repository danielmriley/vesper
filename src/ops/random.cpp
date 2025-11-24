#include <vesper/ops/random.h>
#include <random>
#include <stdexcept>

namespace vesper::ops {

void uniform_cpu_dispatch(Tensor& tensor, float min, float max) {
    if (tensor.dtype() != DType::Float32) {
        throw std::runtime_error("uniform_ only supports Float32");
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(min, max);
    
    float* data = tensor.data_ptr<float>();
    size_t n = tensor.numel();
    
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

void normal_cpu_dispatch(Tensor& tensor, float mean, float std) {
    if (tensor.dtype() != DType::Float32) {
        throw std::runtime_error("normal_ only supports Float32");
    }

    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(mean, std);
    
    float* data = tensor.data_ptr<float>();
    size_t n = tensor.numel();
    
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

void uniform_(Tensor& tensor, float min, float max) {
    if (tensor.device() == Device::CPU) {
        uniform_cpu_dispatch(tensor, min, max);
    } else if (tensor.device() == Device::HIP) {
#if USE_HIP_BACKEND
        uniform_hip_dispatch(tensor, min, max);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (tensor.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        uniform_cuda_dispatch(tensor, min, max);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for uniform_");
    }
}

void normal_(Tensor& tensor, float mean, float std) {
    if (tensor.device() == Device::CPU) {
        normal_cpu_dispatch(tensor, mean, std);
    } else if (tensor.device() == Device::HIP) {
#if USE_HIP_BACKEND
        normal_hip_dispatch(tensor, mean, std);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (tensor.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        normal_cuda_dispatch(tensor, mean, std);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for normal_");
    }
}

} // namespace vesper::ops
