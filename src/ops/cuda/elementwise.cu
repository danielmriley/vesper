#include <cuda_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/ops/elementwise.h>
#include <string>
#include <stdexcept>

namespace vesper::ops {

// Functors for elementwise operations
template <typename T>
struct Add {
    __device__ T operator()(const T& a, const T& b) const { return a + b; }
};

template <typename T>
struct Sub {
    __device__ T operator()(const T& a, const T& b) const { return a - b; }
};

template <typename T>
struct Mul {
    __device__ T operator()(const T& a, const T& b) const { return a * b; }
};

template <typename T>
struct Div {
    __device__ T operator()(const T& a, const T& b) const { return a / b; }
};

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

void add_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("Only Float32 is supported for now.");
    }

    const int threads_per_block = 256;
    const int num_blocks = (a.numel() + threads_per_block - 1) / threads_per_block;

    if (a.numel() == b.numel()) {
        elementwise_binary_kernel<<<num_blocks, threads_per_block>>>(
            a.data_ptr<const float>(),
            b.data_ptr<const float>(),
            out.data_ptr<float>(),
            a.numel(),
            Add<float>()
        );
    } else {
        // Broadcasting case
        add_broadcast_kernel<<<num_blocks, threads_per_block>>>(
            a.data_ptr<const float>(),
            b.data_ptr<const float>(),
            out.data_ptr<float>(),
            a.numel(),
            b.numel()
        );
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void sub_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("Only Float32 is supported for now.");
    }
    const int threads = 256;
    const int blocks = (a.numel() + threads - 1) / threads;
    elementwise_binary_kernel<<<blocks, threads>>>(
        a.data_ptr<const float>(), 
        b.data_ptr<const float>(), 
        out.data_ptr<float>(),
        a.numel(), 
        Sub<float>()
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void mul_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("Only Float32 is supported for now.");
    }
    const int threads = 256;
    const int blocks = (a.numel() + threads - 1) / threads;
    elementwise_binary_kernel<<<blocks, threads>>>(
        a.data_ptr<const float>(), 
        b.data_ptr<const float>(), 
        out.data_ptr<float>(),
        a.numel(), 
        Mul<float>()
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void div_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("Only Float32 is supported for now.");
    }
    const int threads = 256;
    const int blocks = (a.numel() + threads - 1) / threads;
    elementwise_binary_kernel<<<blocks, threads>>>(
        a.data_ptr<const float>(), 
        b.data_ptr<const float>(), 
        out.data_ptr<float>(),
        a.numel(), 
        Div<float>()
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace vesper::ops
