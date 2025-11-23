#include <vesper/ops/copy.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace vesper::ops {

constexpr int MAX_DIMS = 4;

template <typename T>
__global__ void copy_strided_kernel(
    const T* in, const int64_t* strides_in,
    T* out,
    const int64_t* shape, int dims,
    size_t total_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        size_t offset_in = 0;
        size_t remaining = idx;

        #pragma unroll
        for (int i = dims - 1; i >= 0; --i) {
            size_t coord = remaining % shape[i];
            remaining /= shape[i];
            offset_in += coord * strides_in[i];
        }

        out[idx] = in[offset_in];
    }
}

void copy_strided_cuda_dispatch(const Tensor& src, Tensor& dst) {
    if (src.dtype() != DType::Float32) {
        throw std::runtime_error("copy_strided only supports Float32 currently.");
    }

    size_t n = dst.numel();
    int dims = dst.shape().size();
    
    if (dims > MAX_DIMS) {
        throw std::runtime_error("copy_strided supports up to 4 dims on GPU.");
    }

    int64_t* d_shape;
    int64_t* d_strides_in;
    
    cudaMalloc(&d_shape, dims * sizeof(int64_t));
    cudaMalloc(&d_strides_in, dims * sizeof(int64_t));
    
    cudaMemcpy(d_shape, dst.shape().data(), dims * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides_in, src.strides().data(), dims * sizeof(int64_t), cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    copy_strided_kernel<float><<<blocks, threads>>>(
        src.data_ptr<const float>(), d_strides_in,
        dst.data_ptr<float>(),
        d_shape, dims, n);
        
    cudaFree(d_shape);
    cudaFree(d_strides_in);
}

} // namespace vesper::ops
