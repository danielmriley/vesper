#include <vesper/ops/comparison.h>
#include <cuda_runtime.h>

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
    greater_than_scalar_kernel<float><<<dim3(blocks), dim3(threads), 0, 0>>>(
        a.data_ptr<const float>(), out.data_ptr<float>(), a.numel(), b);
}

} // namespace vesper::ops
