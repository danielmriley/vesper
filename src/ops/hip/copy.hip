#include <vesper/ops/copy.h>
#include <vesper/core/stream.h>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace vesper::ops {

#define MAX_DIMS 8

struct TensorInfo {
    int dims;
    int64_t shape[MAX_DIMS];
    int64_t strides[MAX_DIMS];
};

template <typename T>
__global__ void copy_strided_kernel(const T* src, T* dst, size_t n, 
                                    TensorInfo src_info, TensorInfo dst_info,
                                    bool src_contig, bool dst_contig) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t src_offset = 0;
    size_t dst_offset = 0;

    if (src_contig) {
        src_offset = idx;
    } else {
        size_t remaining = idx;
        for (int d = src_info.dims - 1; d >= 0; --d) {
            size_t coord = remaining % src_info.shape[d];
            remaining /= src_info.shape[d];
            src_offset += coord * src_info.strides[d];
        }
    }

    if (dst_contig) {
        dst_offset = idx;
    } else {
        size_t remaining = idx;
        for (int d = dst_info.dims - 1; d >= 0; --d) {
            size_t coord = remaining % dst_info.shape[d];
            remaining /= dst_info.shape[d];
            dst_offset += coord * dst_info.strides[d];
        }
    }

    dst[dst_offset] = src[src_offset];
}

void copy_strided_hip_dispatch(const Tensor& src, Tensor& dst) {
    size_t n = dst.numel();
    if (n == 0) return;

    TensorInfo src_info;
    src_info.dims = src.shape().size();
    if (src_info.dims > MAX_DIMS) throw std::runtime_error("Tensor rank exceeds MAX_DIMS for HIP copy");
    for(int i=0; i<src_info.dims; ++i) {
        src_info.shape[i] = src.shape()[i];
        src_info.strides[i] = src.strides()[i];
    }

    TensorInfo dst_info;
    dst_info.dims = dst.shape().size();
    if (dst_info.dims > MAX_DIMS) throw std::runtime_error("Tensor rank exceeds MAX_DIMS for HIP copy");
    for(int i=0; i<dst_info.dims; ++i) {
        dst_info.shape[i] = dst.shape()[i];
        dst_info.strides[i] = dst.strides()[i];
    }

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    // Dispatch based on dtype. For now assuming Float32 as per rest of codebase MVP
    if (src.dtype() == DType::Float32) {
        hipLaunchKernelGGL(copy_strided_kernel<float>, dim3(blocks), dim3(threads), 0, stream,
            src.data_ptr<const float>(),
            dst.data_ptr<float>(),
            n,
            src_info,
            dst_info,
            src.is_contiguous(),
            dst.is_contiguous()
        );
    } else {
        throw std::runtime_error("copy_strided_hip_dispatch only supports Float32 currently");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP kernel launch failed: ") + hipGetErrorString(err));
    }
}

} // namespace vesper::ops
