#include <vesper/ops/copy.h>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <vector>

namespace vesper::ops {

constexpr int MAX_DIMS = 4;

// Kernel: out[idx] = in[strided_index(idx)]
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

void copy_strided_hip_dispatch(const Tensor& src, Tensor& dst) {
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
    
    hipMalloc(&d_shape, dims * sizeof(int64_t));
    hipMalloc(&d_strides_in, dims * sizeof(int64_t));
    
    hipMemcpy(d_shape, dst.shape().data(), dims * sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy(d_strides_in, src.strides().data(), dims * sizeof(int64_t), hipMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    hipLaunchKernelGGL(copy_strided_kernel<float>, 
        dim3(blocks), dim3(threads), 0, 0,
        src.data_ptr<const float>(), d_strides_in,
        dst.data_ptr<float>(),
        d_shape, dims, n);
        
    hipFree(d_shape);
    hipFree(d_strides_in);
}

} // namespace vesper::ops
