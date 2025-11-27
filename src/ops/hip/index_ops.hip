#include <vesper/ops/index_ops.h>
#include <vesper/core/stream.h>
#include <hip/hip_runtime.h>

namespace vesper::ops {

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
    const int64_t* __restrict__ in_shape,
    const int64_t* __restrict__ in_strides,
    const int64_t* __restrict__ idx_shape,
    const int64_t* __restrict__ out_strides,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;
    
    // Compute multi-dimensional coordinate from linear index
    int64_t coord[8];  // Max 8 dimensions
    int64_t tmp = tid;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        coord[d] = tmp % idx_shape[d];
        tmp /= idx_shape[d];
    }
    
    // Get index value
    int64_t idx_val = static_cast<int64_t>(indices[tid]);
    
    // Build input coordinate (replace coord[dim] with idx_val)
    int64_t in_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t c = (d == dim) ? idx_val : coord[d];
        in_linear += c * in_strides[d];
    }
    
    // Compute output linear index
    int64_t out_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        out_linear += coord[d] * out_strides[d];
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
    const int64_t* __restrict__ in_shape,
    const int64_t* __restrict__ in_strides,
    const int64_t* __restrict__ idx_shape,
    const int64_t* __restrict__ out_strides,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;
    
    int64_t coord[8];
    int64_t tmp = tid;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        coord[d] = tmp % idx_shape[d];
        tmp /= idx_shape[d];
    }
    
    int64_t idx_val = indices[tid];
    
    int64_t in_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t c = (d == dim) ? idx_val : coord[d];
        in_linear += c * in_strides[d];
    }
    
    int64_t out_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        out_linear += coord[d] * out_strides[d];
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
    const int64_t* __restrict__ out_shape,
    const int64_t* __restrict__ out_strides,
    const int64_t* __restrict__ idx_shape,
    const int64_t* __restrict__ src_strides,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;
    
    int64_t coord[8];
    int64_t tmp = tid;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        coord[d] = tmp % idx_shape[d];
        tmp /= idx_shape[d];
    }
    
    int64_t idx_val = static_cast<int64_t>(indices[tid]);
    
    int64_t out_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t c = (d == dim) ? idx_val : coord[d];
        out_linear += c * out_strides[d];
    }
    
    int64_t src_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        src_linear += coord[d] * src_strides[d];
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
    const int64_t* __restrict__ out_shape,
    const int64_t* __restrict__ out_strides,
    const int64_t* __restrict__ idx_shape,
    const int64_t* __restrict__ src_strides,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;
    
    int64_t coord[8];
    int64_t tmp = tid;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        coord[d] = tmp % idx_shape[d];
        tmp /= idx_shape[d];
    }
    
    int64_t idx_val = indices[tid];
    
    int64_t out_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t c = (d == dim) ? idx_val : coord[d];
        out_linear += c * out_strides[d];
    }
    
    int64_t src_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        src_linear += coord[d] * src_strides[d];
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
    const int64_t* __restrict__ out_shape,
    const int64_t* __restrict__ out_strides,
    const int64_t* __restrict__ idx_shape,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;
    
    int64_t coord[8];
    int64_t tmp = tid;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        coord[d] = tmp % idx_shape[d];
        tmp /= idx_shape[d];
    }
    
    int64_t idx_val = static_cast<int64_t>(indices[tid]);
    
    int64_t out_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t c = (d == dim) ? idx_val : coord[d];
        out_linear += c * out_strides[d];
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
    const int64_t* __restrict__ out_shape,
    const int64_t* __restrict__ out_strides,
    const int64_t* __restrict__ idx_shape,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;
    
    int64_t coord[8];
    int64_t tmp = tid;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        coord[d] = tmp % idx_shape[d];
        tmp /= idx_shape[d];
    }
    
    int64_t idx_val = indices[tid];
    
    int64_t out_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t c = (d == dim) ? idx_val : coord[d];
        out_linear += c * out_strides[d];
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

void gather_hip_dispatch(const Tensor& input, int64_t dim, const Tensor& index, Tensor& out) {
    int64_t ndim = static_cast<int64_t>(input.shape().size());
    int64_t numel = index.numel();
    
    // Copy shape/stride info to device
    std::vector<int64_t> in_shape = input.shape();
    std::vector<int64_t> in_strides = input.strides();
    std::vector<int64_t> idx_shape = index.shape();
    std::vector<int64_t> out_strides = out.strides();
    
    int64_t *d_in_shape, *d_in_strides, *d_idx_shape, *d_out_strides;
    hipMalloc(&d_in_shape, ndim * sizeof(int64_t));
    hipMalloc(&d_in_strides, ndim * sizeof(int64_t));
    hipMalloc(&d_idx_shape, ndim * sizeof(int64_t));
    hipMalloc(&d_out_strides, ndim * sizeof(int64_t));
    
    hipMemcpy(d_in_shape, in_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy(d_in_strides, in_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy(d_idx_shape, idx_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy(d_out_strides, out_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    if (index.dtype() == DType::Int32) {
        hipLaunchKernelGGL(gather_kernel_int32, dim3(blocks), dim3(threads), 0, stream,
            input.data_ptr<float>(),
            index.data_ptr<int32_t>(),
            out.data_ptr<float>(),
            dim, ndim,
            d_in_shape, d_in_strides,
            d_idx_shape, d_out_strides,
            numel
        );
    } else if (index.dtype() == DType::Int64) {
        hipLaunchKernelGGL(gather_kernel_int64, dim3(blocks), dim3(threads), 0, stream,
            input.data_ptr<float>(),
            index.data_ptr<int64_t>(),
            out.data_ptr<float>(),
            dim, ndim,
            d_in_shape, d_in_strides,
            d_idx_shape, d_out_strides,
            numel
        );
    } else {
        throw std::runtime_error("gather: index tensor must be Int32 or Int64");
    }
    
    hipDeviceSynchronize();
    
    hipFree(d_in_shape);
    hipFree(d_in_strides);
    hipFree(d_idx_shape);
    hipFree(d_out_strides);
}

void scatter_hip_dispatch(Tensor& out, int64_t dim, const Tensor& index, const Tensor& src) {
    int64_t ndim = static_cast<int64_t>(out.shape().size());
    int64_t numel = index.numel();
    
    std::vector<int64_t> out_shape = out.shape();
    std::vector<int64_t> out_strides = out.strides();
    std::vector<int64_t> idx_shape = index.shape();
    std::vector<int64_t> src_strides = src.strides();
    
    int64_t *d_out_shape, *d_out_strides, *d_idx_shape, *d_src_strides;
    hipMalloc(&d_out_shape, ndim * sizeof(int64_t));
    hipMalloc(&d_out_strides, ndim * sizeof(int64_t));
    hipMalloc(&d_idx_shape, ndim * sizeof(int64_t));
    hipMalloc(&d_src_strides, ndim * sizeof(int64_t));
    
    hipMemcpy(d_out_shape, out_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy(d_out_strides, out_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy(d_idx_shape, idx_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy(d_src_strides, src_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    if (index.dtype() == DType::Int32) {
        hipLaunchKernelGGL(scatter_kernel_int32, dim3(blocks), dim3(threads), 0, stream,
            out.data_ptr<float>(),
            index.data_ptr<int32_t>(),
            src.data_ptr<float>(),
            dim, ndim,
            d_out_shape, d_out_strides,
            d_idx_shape, d_src_strides,
            numel
        );
    } else if (index.dtype() == DType::Int64) {
        hipLaunchKernelGGL(scatter_kernel_int64, dim3(blocks), dim3(threads), 0, stream,
            out.data_ptr<float>(),
            index.data_ptr<int64_t>(),
            src.data_ptr<float>(),
            dim, ndim,
            d_out_shape, d_out_strides,
            d_idx_shape, d_src_strides,
            numel
        );
    } else {
        throw std::runtime_error("scatter: index tensor must be Int32 or Int64");
    }
    
    hipDeviceSynchronize();
    
    hipFree(d_out_shape);
    hipFree(d_out_strides);
    hipFree(d_idx_shape);
    hipFree(d_src_strides);
}

void scatter_scalar_hip_dispatch(Tensor& out, int64_t dim, const Tensor& index, float value) {
    int64_t ndim = static_cast<int64_t>(out.shape().size());
    int64_t numel = index.numel();
    
    std::vector<int64_t> out_shape = out.shape();
    std::vector<int64_t> out_strides = out.strides();
    std::vector<int64_t> idx_shape = index.shape();
    
    int64_t *d_out_shape, *d_out_strides, *d_idx_shape;
    hipMalloc(&d_out_shape, ndim * sizeof(int64_t));
    hipMalloc(&d_out_strides, ndim * sizeof(int64_t));
    hipMalloc(&d_idx_shape, ndim * sizeof(int64_t));
    
    hipMemcpy(d_out_shape, out_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy(d_out_strides, out_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy(d_idx_shape, idx_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    if (index.dtype() == DType::Int32) {
        hipLaunchKernelGGL(scatter_scalar_kernel_int32, dim3(blocks), dim3(threads), 0, stream,
            out.data_ptr<float>(),
            index.data_ptr<int32_t>(),
            value,
            dim, ndim,
            d_out_shape, d_out_strides,
            d_idx_shape,
            numel
        );
    } else if (index.dtype() == DType::Int64) {
        hipLaunchKernelGGL(scatter_scalar_kernel_int64, dim3(blocks), dim3(threads), 0, stream,
            out.data_ptr<float>(),
            index.data_ptr<int64_t>(),
            value,
            dim, ndim,
            d_out_shape, d_out_strides,
            d_idx_shape,
            numel
        );
    } else {
        throw std::runtime_error("scatter: index tensor must be Int32 or Int64");
    }
    
    hipDeviceSynchronize();
    
    hipFree(d_out_shape);
    hipFree(d_out_strides);
    hipFree(d_idx_shape);
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
