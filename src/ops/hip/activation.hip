#include <vesper/nn/functional.h>
#include <hip/hip_runtime.h>
#include <cmath>

namespace vesper::nn::functional {

template <typename T>
__global__ void sigmoid_kernel(const T* in, T* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<T>(1) / (static_cast<T>(1) + exp(-in[idx]));
    }
}

void sigmoid_hip_dispatch(const Tensor& input, Tensor& output) {
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    hipLaunchKernelGGL(sigmoid_kernel<float>, dim3(blocks), dim3(threads), 0, 0,
        input.data_ptr<const float>(), output.data_ptr<float>(), input.numel());
}

template <typename T>
__global__ void relu_kernel(const T* in, T* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] > static_cast<T>(0) ? in[idx] : static_cast<T>(0);
    }
}

void relu_hip_dispatch(const Tensor& input, Tensor& output) {
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    hipLaunchKernelGGL(relu_kernel<float>, dim3(blocks), dim3(threads), 0, 0,
        input.data_ptr<const float>(), output.data_ptr<float>(), input.numel());
}

} // namespace vesper::nn::functional
