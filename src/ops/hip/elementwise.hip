#include <hip/hip_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/ops/elementwise.h>
#include <functional>

namespace vesper::ops {

// Generic kernel for any element-wise binary operation
template <typename T, typename Op>
__global__ void elementwise_binary_kernel(const T* a, const T* b, T* out, size_t n, Op op) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = op(a[idx], b[idx]);
    }
}

// Kernel for broadcasting addition: a [M, N] + b [N] -> out [M, N]
// b is broadcasted along the first dimension(s)
template <typename T>
__global__ void add_broadcast_kernel(const T* a, const T* b, T* out, size_t total_elements, size_t b_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out[idx] = a[idx] + b[idx % b_size];
    }
}

void add_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("Only Float32 is supported for now.");
    }

    const int threads_per_block = 256;
    const int num_blocks = (a.numel() + threads_per_block - 1) / threads_per_block;

    if (a.numel() == b.numel()) {
        auto kernel = elementwise_binary_kernel<float, std::plus<float>>;
        hipLaunchKernelGGL(
            kernel,
            dim3(num_blocks),
            dim3(threads_per_block),
            0, 0, // shared mem bytes, stream
            a.data_ptr<const float>(),
            b.data_ptr<const float>(),
            out.data_ptr<float>(),
            a.numel(),
            std::plus<float>()
        );
    } else {
        // Broadcasting case
        hipLaunchKernelGGL(
            add_broadcast_kernel<float>,
            dim3(num_blocks),
            dim3(threads_per_block),
            0, 0,
            a.data_ptr<const float>(),
            b.data_ptr<const float>(),
            out.data_ptr<float>(),
            a.numel(),
            b.numel()
        );
    }
}

void sub_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("Only Float32 is supported for now.");
    }
    const int threads = 256;
    const int blocks = (a.numel() + threads - 1) / threads;
    auto kernel = elementwise_binary_kernel<float, std::minus<float>>;
    hipLaunchKernelGGL(kernel,
        dim3(blocks), dim3(threads), 0, 0,
        a.data_ptr<const float>(), b.data_ptr<const float>(), out.data_ptr<float>(),
        a.numel(), std::minus<float>());
}

void mul_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("Only Float32 is supported for now.");
    }
    const int threads = 256;
    const int blocks = (a.numel() + threads - 1) / threads;
    auto kernel = elementwise_binary_kernel<float, std::multiplies<float>>;
    hipLaunchKernelGGL(kernel,
        dim3(blocks), dim3(threads), 0, 0,
        a.data_ptr<const float>(), b.data_ptr<const float>(), out.data_ptr<float>(),
        a.numel(), std::multiplies<float>());
}

void div_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("Only Float32 is supported for now.");
    }
    const int threads = 256;
    const int blocks = (a.numel() + threads - 1) / threads;
    auto kernel = elementwise_binary_kernel<float, std::divides<float>>;
    hipLaunchKernelGGL(kernel,
        dim3(blocks), dim3(threads), 0, 0,
        a.data_ptr<const float>(), b.data_ptr<const float>(), out.data_ptr<float>(),
        a.numel(), std::divides<float>());
}

} // namespace vesper::ops
