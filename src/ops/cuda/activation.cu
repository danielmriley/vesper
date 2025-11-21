#include <vesper/nn/functional.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace vesper::nn::functional {

template <typename T>
__global__ void sigmoid_kernel(const T* in, T* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<T>(1) / (static_cast<T>(1) + exp(-in[idx]));
    }
}

void sigmoid_cuda_dispatch(const Tensor& input, Tensor& output) {
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    sigmoid_kernel<float><<<blocks, threads>>>(
        input.data_ptr<const float>(), 
        output.data_ptr<float>(), 
        input.numel()
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

template <typename T>
__global__ void relu_kernel(const T* in, T* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] > static_cast<T>(0) ? in[idx] : static_cast<T>(0);
    }
}

void relu_cuda_dispatch(const Tensor& input, Tensor& output) {
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    relu_kernel<float><<<blocks, threads>>>(
        input.data_ptr<const float>(), 
        output.data_ptr<float>(), 
        input.numel()
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace vesper::nn::functional
