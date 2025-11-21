#include <vesper/ops/comparison.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace vesper::ops {

template <typename T>
__global__ void greater_than_scalar_kernel(const T* in, T* out, size_t n, T scalar) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (in[idx] > scalar) ? static_cast<T>(1) : static_cast<T>(0);
    }
}

void greater_than_cuda_dispatch(const Tensor& a, float b, Tensor& out) {
    const int threads = 256;
    const int blocks = (a.numel() + threads - 1) / threads;
    greater_than_scalar_kernel<float><<<blocks, threads>>>(
        a.data_ptr<const float>(), 
        out.data_ptr<float>(), 
        a.numel(), 
        b
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace vesper::ops
