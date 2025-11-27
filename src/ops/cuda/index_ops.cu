#include <vesper/ops/index_ops.h>
#include <vesper/core/stream.h>
#include <cuda_runtime.h>

namespace vesper::ops {

// ============================================================================
// CUDA Kernels
// ============================================================================

// Gather kernel - gathers values along a dimension
template <typename T, typename IndexT>
__global__ void gather_kernel(
    const T* __restrict__ input,
    const IndexT* __restrict__ indices,
    T* __restrict__ output,
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

// Scatter kernel - scatters values along a dimension
template <typename T, typename IndexT>
__global__ void scatter_kernel(
    T* __restrict__ output,
    const IndexT* __restrict__ indices,
    const T* __restrict__ src,
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
    
    // Build output coordinate
    int64_t out_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t c = (d == dim) ? idx_val : coord[d];
        out_linear += c * out_strides[d];
    }
    
    // Source linear index
    int64_t src_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        src_linear += coord[d] * src_strides[d];
    }
    
    output[out_linear] = src[src_linear];
}

// Scatter scalar kernel
template <typename T, typename IndexT>
__global__ void scatter_scalar_kernel(
    T* __restrict__ output,
    const IndexT* __restrict__ indices,
    T value,
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

// Where kernel - select from x or y based on condition
template <typename T>
__global__ void where_kernel(
    const T* __restrict__ condition,
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ output,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;
    
    output[tid] = (condition[tid] != static_cast<T>(0)) ? x[tid] : y[tid];
}

// Masked fill kernel
template <typename T>
__global__ void masked_fill_kernel(
    T* __restrict__ output,
    const T* __restrict__ mask,
    T value,
    int64_t numel
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;
    
    if (mask[tid] != static_cast<T>(0)) {
        output[tid] = value;
    }
}

// ============================================================================
// Dispatch Functions
// ============================================================================

void gather_cuda_dispatch(const Tensor& input, int64_t dim, const Tensor& index, Tensor& out) {
    int64_t ndim = static_cast<int64_t>(input.shape().size());
    int64_t numel = index.numel();
    
    // Copy shape/stride info to device
    std::vector<int64_t> in_shape = input.shape();
    std::vector<int64_t> in_strides = input.strides();
    std::vector<int64_t> idx_shape = index.shape();
    std::vector<int64_t> out_strides = out.strides();
    
    int64_t *d_in_shape, *d_in_strides, *d_idx_shape, *d_out_strides;
    cudaMalloc(&d_in_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_in_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_idx_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_out_strides, ndim * sizeof(int64_t));
    
    cudaMemcpy(d_in_shape, in_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_strides, in_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx_shape, idx_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
    
    if (index.dtype() == DType::Int32) {
        gather_kernel<float, int32_t><<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(),
            index.data_ptr<int32_t>(),
            out.data_ptr<float>(),
            dim, ndim,
            d_in_shape, d_in_strides,
            d_idx_shape, d_out_strides,
            numel
        );
    } else if (index.dtype() == DType::Int64) {
        gather_kernel<float, int64_t><<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(),
            index.data_ptr<int64_t>(),
            out.data_ptr<float>(),
            dim, ndim,
            d_in_shape, d_in_strides,
            d_idx_shape, d_out_strides,
            numel
        );
    } else {
        gather_kernel<float, float><<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(),
            index.data_ptr<float>(),
            out.data_ptr<float>(),
            dim, ndim,
            d_in_shape, d_in_strides,
            d_idx_shape, d_out_strides,
            numel
        );
    }
    
    cudaDeviceSynchronize();
    
    cudaFree(d_in_shape);
    cudaFree(d_in_strides);
    cudaFree(d_idx_shape);
    cudaFree(d_out_strides);
}

void scatter_cuda_dispatch(Tensor& out, int64_t dim, const Tensor& index, const Tensor& src) {
    int64_t ndim = static_cast<int64_t>(out.shape().size());
    int64_t numel = index.numel();
    
    std::vector<int64_t> out_shape = out.shape();
    std::vector<int64_t> out_strides = out.strides();
    std::vector<int64_t> idx_shape = index.shape();
    std::vector<int64_t> src_strides = src.strides();
    
    int64_t *d_out_shape, *d_out_strides, *d_idx_shape, *d_src_strides;
    cudaMalloc(&d_out_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_out_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_idx_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_src_strides, ndim * sizeof(int64_t));
    
    cudaMemcpy(d_out_shape, out_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx_shape, idx_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src_strides, src_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
    
    if (index.dtype() == DType::Int32) {
        scatter_kernel<float, int32_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<float>(),
            index.data_ptr<int32_t>(),
            src.data_ptr<float>(),
            dim, ndim,
            d_out_shape, d_out_strides,
            d_idx_shape, d_src_strides,
            numel
        );
    } else if (index.dtype() == DType::Int64) {
        scatter_kernel<float, int64_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<float>(),
            index.data_ptr<int64_t>(),
            src.data_ptr<float>(),
            dim, ndim,
            d_out_shape, d_out_strides,
            d_idx_shape, d_src_strides,
            numel
        );
    }
    
    cudaDeviceSynchronize();
    
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_idx_shape);
    cudaFree(d_src_strides);
}

void scatter_scalar_cuda_dispatch(Tensor& out, int64_t dim, const Tensor& index, float value) {
    int64_t ndim = static_cast<int64_t>(out.shape().size());
    int64_t numel = index.numel();
    
    std::vector<int64_t> out_shape = out.shape();
    std::vector<int64_t> out_strides = out.strides();
    std::vector<int64_t> idx_shape = index.shape();
    
    int64_t *d_out_shape, *d_out_strides, *d_idx_shape;
    cudaMalloc(&d_out_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_out_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_idx_shape, ndim * sizeof(int64_t));
    
    cudaMemcpy(d_out_shape, out_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx_shape, idx_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
    
    if (index.dtype() == DType::Int32) {
        scatter_scalar_kernel<float, int32_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<float>(),
            index.data_ptr<int32_t>(),
            value,
            dim, ndim,
            d_out_shape, d_out_strides,
            d_idx_shape,
            numel
        );
    } else if (index.dtype() == DType::Int64) {
        scatter_scalar_kernel<float, int64_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<float>(),
            index.data_ptr<int64_t>(),
            value,
            dim, ndim,
            d_out_shape, d_out_strides,
            d_idx_shape,
            numel
        );
    }
    
    cudaDeviceSynchronize();
    
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_idx_shape);
}

void where_cuda_dispatch(const Tensor& condition, const Tensor& x, const Tensor& y, Tensor& out) {
    int64_t numel = out.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
    
    where_kernel<float><<<blocks, threads, 0, stream>>>(
        condition.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        numel
    );
}

void masked_fill_cuda_dispatch(Tensor& out, const Tensor& mask, float value) {
    int64_t numel = out.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
    
    masked_fill_kernel<float><<<blocks, threads, 0, stream>>>(
        out.data_ptr<float>(),
        mask.data_ptr<float>(),
        value,
        numel
    );
}

} // namespace vesper::ops
