#include <vesper/core/tensor.h>
#include <vesper/core/stream.h>
#include <cuda_runtime.h>

namespace vesper::ops {

template <typename SrcT, typename DstT>
__global__ void cast_kernel(const SrcT* input, DstT* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = static_cast<DstT>(input[idx]);
    }
}

template <typename SrcT>
void dispatch_cast_dst(const Tensor& input, Tensor& output, size_t numel, cudaStream_t stream) {
    const SrcT* src_ptr = input.data_ptr<SrcT>();
    
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    switch (output.dtype()) {
        case DType::Float32:
            cast_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>( src_ptr, output.data_ptr<float>(), numel);
            break;
        case DType::Int32:
            cast_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>( src_ptr, output.data_ptr<int32_t>(), numel);
            break;
        // Add other types as needed
        default:
            throw std::runtime_error("Unsupported destination dtype for cast (HIP)");
    }
}

void cast_cuda_dispatch(const Tensor& input, Tensor& output) {
    size_t numel = input.numel();
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());

    switch (input.dtype()) {
        case DType::Float32:
            dispatch_cast_dst<float>(input, output, numel, stream);
            break;
        case DType::Int32:
            dispatch_cast_dst<int32_t>(input, output, numel, stream);
            break;
        default:
            throw std::runtime_error("Unsupported source dtype for cast (HIP)");
    }
}

} // namespace vesper::ops
