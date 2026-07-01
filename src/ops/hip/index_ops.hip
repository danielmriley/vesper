#include <vesper/ops/index_ops.h>
#include <vesper/core/stream.h>
#include <hip/hip_runtime.h>
#include <stdexcept>

namespace vesper::ops {

// Maximum tensor rank supported by the index kernels. The per-thread coordinate
// buffer is fixed at 8, so this is a hard limit.
static constexpr int64_t IDX_MAX_DIMS = 8;

// POD carrying a shape/stride array by value through the kernel argument buffer,
// replacing the per-call hipMalloc + hipMemcpy of a device array.
struct IdxDims {
    int64_t v[IDX_MAX_DIMS];
};

// ============================================================================
// HIP Kernels - Non-templated versions for each type combination
// ============================================================================

// Gather kernel - Int32 indices
__global__ void gather_kernel_int32(
    const float* __restrict__ input,
    const int32_t* __restrict__ indices,
    float* __restrict__ output,
    int64_t dim,
    int64_t ndim,
    IdxDims in_shape,
    IdxDims in_strides,
    IdxDims idx_shape,
    IdxDims out_strides,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;

    // Compute multi-dimensional coordinate from linear index
    int64_t coord[8];  // Max 8 dimensions
    int64_t tmp = tid;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        coord[d] = tmp % idx_shape.v[d];
        tmp /= idx_shape.v[d];
    }

    // Get index value
    int64_t idx_val = static_cast<int64_t>(indices[tid]);

    // Compute output linear index
    int64_t out_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        out_linear += coord[d] * out_strides.v[d];
    }

    if (idx_val < 0 || idx_val >= in_shape.v[dim]) { output[out_linear] = 0.0f; return; }

    // Build input coordinate (replace coord[dim] with idx_val)
    int64_t in_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t c = (d == dim) ? idx_val : coord[d];
        in_linear += c * in_strides.v[d];
    }

    output[out_linear] = input[in_linear];
}

// Gather kernel - Int64 indices
__global__ void gather_kernel_int64(
    const float* __restrict__ input,
    const int64_t* __restrict__ indices,
    float* __restrict__ output,
    int64_t dim,
    int64_t ndim,
    IdxDims in_shape,
    IdxDims in_strides,
    IdxDims idx_shape,
    IdxDims out_strides,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;

    int64_t coord[8];
    int64_t tmp = tid;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        coord[d] = tmp % idx_shape.v[d];
        tmp /= idx_shape.v[d];
    }

    int64_t idx_val = indices[tid];

    int64_t out_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        out_linear += coord[d] * out_strides.v[d];
    }

    if (idx_val < 0 || idx_val >= in_shape.v[dim]) { output[out_linear] = 0.0f; return; }

    int64_t in_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t c = (d == dim) ? idx_val : coord[d];
        in_linear += c * in_strides.v[d];
    }

    output[out_linear] = input[in_linear];
}

// Scatter kernel - Int32 indices
__global__ void scatter_kernel_int32(
    float* __restrict__ output,
    const int32_t* __restrict__ indices,
    const float* __restrict__ src,
    int64_t dim,
    int64_t ndim,
    IdxDims out_shape,
    IdxDims out_strides,
    IdxDims idx_shape,
    IdxDims src_strides,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;

    int64_t coord[8];
    int64_t tmp = tid;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        coord[d] = tmp % idx_shape.v[d];
        tmp /= idx_shape.v[d];
    }

    int64_t idx_val = static_cast<int64_t>(indices[tid]);
    if (idx_val < 0 || idx_val >= out_shape.v[dim]) return;

    int64_t out_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t c = (d == dim) ? idx_val : coord[d];
        out_linear += c * out_strides.v[d];
    }

    int64_t src_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        src_linear += coord[d] * src_strides.v[d];
    }

    output[out_linear] = src[src_linear];
}

// Scatter kernel - Int64 indices
__global__ void scatter_kernel_int64(
    float* __restrict__ output,
    const int64_t* __restrict__ indices,
    const float* __restrict__ src,
    int64_t dim,
    int64_t ndim,
    IdxDims out_shape,
    IdxDims out_strides,
    IdxDims idx_shape,
    IdxDims src_strides,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;

    int64_t coord[8];
    int64_t tmp = tid;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        coord[d] = tmp % idx_shape.v[d];
        tmp /= idx_shape.v[d];
    }

    int64_t idx_val = indices[tid];
    if (idx_val < 0 || idx_val >= out_shape.v[dim]) return;

    int64_t out_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t c = (d == dim) ? idx_val : coord[d];
        out_linear += c * out_strides.v[d];
    }

    int64_t src_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        src_linear += coord[d] * src_strides.v[d];
    }

    output[out_linear] = src[src_linear];
}

// Scatter scalar kernel - Int32 indices
__global__ void scatter_scalar_kernel_int32(
    float* __restrict__ output,
    const int32_t* __restrict__ indices,
    float value,
    int64_t dim,
    int64_t ndim,
    IdxDims out_shape,
    IdxDims out_strides,
    IdxDims idx_shape,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;

    int64_t coord[8];
    int64_t tmp = tid;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        coord[d] = tmp % idx_shape.v[d];
        tmp /= idx_shape.v[d];
    }

    int64_t idx_val = static_cast<int64_t>(indices[tid]);
    if (idx_val < 0 || idx_val >= out_shape.v[dim]) return;

    int64_t out_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t c = (d == dim) ? idx_val : coord[d];
        out_linear += c * out_strides.v[d];
    }

    output[out_linear] = value;
}

// Scatter scalar kernel - Int64 indices
__global__ void scatter_scalar_kernel_int64(
    float* __restrict__ output,
    const int64_t* __restrict__ indices,
    float value,
    int64_t dim,
    int64_t ndim,
    IdxDims out_shape,
    IdxDims out_strides,
    IdxDims idx_shape,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;

    int64_t coord[8];
    int64_t tmp = tid;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        coord[d] = tmp % idx_shape.v[d];
        tmp /= idx_shape.v[d];
    }

    int64_t idx_val = indices[tid];
    if (idx_val < 0 || idx_val >= out_shape.v[dim]) return;

    int64_t out_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t c = (d == dim) ? idx_val : coord[d];
        out_linear += c * out_strides.v[d];
    }

    output[out_linear] = value;
}

// Where kernel - select from x or y based on condition
__global__ void where_kernel_f32(
    const float* __restrict__ condition,
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ output,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;

    output[tid] = (condition[tid] != 0.0f) ? x[tid] : y[tid];
}

// Masked fill kernel
__global__ void masked_fill_kernel_f32(
    float* __restrict__ output,
    const float* __restrict__ mask,
    float value,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;

    if (mask[tid] != 0.0f) {
        output[tid] = value;
    }
}

// ============================================================================
// Dispatch Functions
// ============================================================================

// Copies up to IDX_MAX_DIMS entries of a shape/stride vector into a by-value POD.
static inline IdxDims to_idx_dims(const std::vector<int64_t>& src, int64_t ndim) {
    IdxDims d{};
    for (int64_t i = 0; i < ndim; ++i) d.v[i] = src[i];
    return d;
}

void gather_hip_dispatch(const Tensor& input, int64_t dim, const Tensor& index, Tensor& out) {
    int64_t ndim = static_cast<int64_t>(input.shape().size());
    int64_t numel = index.numel();

    if (ndim > IDX_MAX_DIMS) {
        throw std::runtime_error("gather: tensor rank exceeds max supported (8)");
    }

    // Shape/stride metadata rides inline in the kernel arguments (no hipMalloc).
    IdxDims in_shape    = to_idx_dims(input.shape(),   ndim);
    IdxDims in_strides  = to_idx_dims(input.strides(), ndim);
    IdxDims idx_shape   = to_idx_dims(index.shape(),   ndim);
    IdxDims out_strides = to_idx_dims(out.strides(),   ndim);

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    if (index.dtype() == DType::Int32) {
        hipLaunchKernelGGL(gather_kernel_int32, dim3(blocks), dim3(threads), 0, stream,
            input.data_ptr<float>(),
            index.data_ptr<int32_t>(),
            out.data_ptr<float>(),
            dim, ndim,
            in_shape, in_strides,
            idx_shape, out_strides,
            numel
        );
    } else if (index.dtype() == DType::Int64) {
        hipLaunchKernelGGL(gather_kernel_int64, dim3(blocks), dim3(threads), 0, stream,
            input.data_ptr<float>(),
            index.data_ptr<int64_t>(),
            out.data_ptr<float>(),
            dim, ndim,
            in_shape, in_strides,
            idx_shape, out_strides,
            numel
        );
    } else {
        throw std::runtime_error("gather: index tensor must be Int32 or Int64");
    }
}

void scatter_hip_dispatch(Tensor& out, int64_t dim, const Tensor& index, const Tensor& src) {
    int64_t ndim = static_cast<int64_t>(out.shape().size());
    int64_t numel = index.numel();

    if (ndim > IDX_MAX_DIMS) {
        throw std::runtime_error("scatter: tensor rank exceeds max supported (8)");
    }

    IdxDims out_shape   = to_idx_dims(out.shape(),     ndim);
    IdxDims out_strides = to_idx_dims(out.strides(),   ndim);
    IdxDims idx_shape   = to_idx_dims(index.shape(),   ndim);
    IdxDims src_strides = to_idx_dims(src.strides(),   ndim);

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    if (index.dtype() == DType::Int32) {
        hipLaunchKernelGGL(scatter_kernel_int32, dim3(blocks), dim3(threads), 0, stream,
            out.data_ptr<float>(),
            index.data_ptr<int32_t>(),
            src.data_ptr<float>(),
            dim, ndim,
            out_shape, out_strides,
            idx_shape, src_strides,
            numel
        );
    } else if (index.dtype() == DType::Int64) {
        hipLaunchKernelGGL(scatter_kernel_int64, dim3(blocks), dim3(threads), 0, stream,
            out.data_ptr<float>(),
            index.data_ptr<int64_t>(),
            src.data_ptr<float>(),
            dim, ndim,
            out_shape, out_strides,
            idx_shape, src_strides,
            numel
        );
    } else {
        throw std::runtime_error("scatter: index tensor must be Int32 or Int64");
    }
}

void scatter_scalar_hip_dispatch(Tensor& out, int64_t dim, const Tensor& index, float value) {
    int64_t ndim = static_cast<int64_t>(out.shape().size());
    int64_t numel = index.numel();

    if (ndim > IDX_MAX_DIMS) {
        throw std::runtime_error("scatter: tensor rank exceeds max supported (8)");
    }

    IdxDims out_shape   = to_idx_dims(out.shape(),   ndim);
    IdxDims out_strides = to_idx_dims(out.strides(), ndim);
    IdxDims idx_shape   = to_idx_dims(index.shape(), ndim);

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    if (index.dtype() == DType::Int32) {
        hipLaunchKernelGGL(scatter_scalar_kernel_int32, dim3(blocks), dim3(threads), 0, stream,
            out.data_ptr<float>(),
            index.data_ptr<int32_t>(),
            value,
            dim, ndim,
            out_shape, out_strides,
            idx_shape,
            numel
        );
    } else if (index.dtype() == DType::Int64) {
        hipLaunchKernelGGL(scatter_scalar_kernel_int64, dim3(blocks), dim3(threads), 0, stream,
            out.data_ptr<float>(),
            index.data_ptr<int64_t>(),
            value,
            dim, ndim,
            out_shape, out_strides,
            idx_shape,
            numel
        );
    } else {
        throw std::runtime_error("scatter: index tensor must be Int32 or Int64");
    }
}

void where_hip_dispatch(const Tensor& condition, const Tensor& x, const Tensor& y, Tensor& out) {
    int64_t numel = out.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    hipLaunchKernelGGL(where_kernel_f32, dim3(blocks), dim3(threads), 0, stream,
        condition.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        numel
    );
}

void masked_fill_hip_dispatch(Tensor& out, const Tensor& mask, float value) {
    int64_t numel = out.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    hipLaunchKernelGGL(masked_fill_kernel_f32, dim3(blocks), dim3(threads), 0, stream,
        out.data_ptr<float>(),
        mask.data_ptr<float>(),
        value,
        numel
    );
}

} // namespace vesper::ops
