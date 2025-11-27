#include <cuda_runtime.h>
#include <vesper/ops/cat.h>
#include <vesper/core/macros.h>

namespace vesper::ops {

/**
 * GPU kernel for concatenating tensors along dimension 0.
 * Each thread handles one element of the output.
 */
template<typename T>
__global__ void cat_dim0_kernel(
    T* output,
    int num_tensors,
    const T* const* input_ptrs,
    const int64_t* input_dim0_sizes,
    int64_t elements_per_slice,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int64_t slice_idx = idx / elements_per_slice;
    int64_t within_slice = idx % elements_per_slice;
    
    int64_t tensor_idx = 0;
    int64_t slice_offset = 0;
    int64_t cumsum = 0;
    
    for (int64_t t = 0; t < num_tensors; ++t) {
        if (slice_idx < cumsum + input_dim0_sizes[t]) {
            tensor_idx = t;
            slice_offset = slice_idx - cumsum;
            break;
        }
        cumsum += input_dim0_sizes[t];
    }
    
    const T* input_data = input_ptrs[tensor_idx];
    output[idx] = input_data[slice_offset * elements_per_slice + within_slice];
}

/**
 * GPU kernel for concatenating tensors along dimension 1.
 */
template<typename T>
__global__ void cat_dim1_kernel(
    T* output,
    int num_tensors,
    const T* const* input_ptrs,
    const int64_t* input_dim1_sizes,
    int64_t batch_size,
    int64_t total_dim1,
    int64_t elements_per_token,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int64_t elements_per_batch = total_dim1 * elements_per_token;
    int64_t batch_idx = idx / elements_per_batch;
    int64_t within_batch = idx % elements_per_batch;
    int64_t seq_idx = within_batch / elements_per_token;
    int64_t within_token = within_batch % elements_per_token;
    
    int64_t tensor_idx = 0;
    int64_t local_seq_idx = seq_idx;
    int64_t cumsum = 0;
    
    for (int64_t t = 0; t < num_tensors; ++t) {
        if (seq_idx < cumsum + input_dim1_sizes[t]) {
            tensor_idx = t;
            local_seq_idx = seq_idx - cumsum;
            break;
        }
        cumsum += input_dim1_sizes[t];
    }
    
    const T* input_data = input_ptrs[tensor_idx];
    int64_t input_elements_per_batch = input_dim1_sizes[tensor_idx] * elements_per_token;
    int64_t input_idx = batch_idx * input_elements_per_batch + local_seq_idx * elements_per_token + within_token;
    
    output[idx] = input_data[input_idx];
}

/**
 * Generic kernel for concatenating along any dimension.
 */
template<typename T>
__global__ void cat_generic_kernel(
    T* output,
    int num_tensors,
    const T* const* input_ptrs,
    const int64_t* input_dim_sizes,
    int cat_dim,
    int64_t outer_size,
    int64_t inner_size,
    int64_t total_cat_dim,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int64_t inner_idx = idx % inner_size;
    int64_t remaining = idx / inner_size;
    int64_t cat_idx = remaining % total_cat_dim;
    int64_t outer_idx = remaining / total_cat_dim;
    
    int64_t tensor_idx = 0;
    int64_t local_cat_idx = cat_idx;
    int64_t cumsum = 0;
    
    for (int64_t t = 0; t < num_tensors; ++t) {
        if (cat_idx < cumsum + input_dim_sizes[t]) {
            tensor_idx = t;
            local_cat_idx = cat_idx - cumsum;
            break;
        }
        cumsum += input_dim_sizes[t];
    }
    
    const T* input_data = input_ptrs[tensor_idx];
    int64_t input_cat_size = input_dim_sizes[tensor_idx];
    int64_t input_idx = outer_idx * (input_cat_size * inner_size) + local_cat_idx * inner_size + inner_idx;
    
    output[idx] = input_data[input_idx];
}

void cat_cuda_dispatch(const std::vector<Tensor>& inputs, Tensor& output, int dim) {
    VESPER_CHECK(!inputs.empty(), "Cannot cat empty tensor list");
    
    int num_tensors = static_cast<int>(inputs.size());
    const auto& shape = inputs[0].shape();
    int ndim = static_cast<int>(shape.size());
    
    if (dim < 0) dim += ndim;
    VESPER_CHECK(dim >= 0 && dim < ndim, "Invalid dimension for cat");
    
    int64_t total_elements = output.numel();
    if (total_elements == 0) return;
    
    std::vector<const void*> input_ptrs_host(num_tensors);
    std::vector<int64_t> dim_sizes_host(num_tensors);
    
    for (int i = 0; i < num_tensors; ++i) {
        input_ptrs_host[i] = inputs[i].data_ptr<void>();
        dim_sizes_host[i] = inputs[i].shape()[dim];
    }
    
    void** input_ptrs_device;
    int64_t* dim_sizes_device;
    
    cudaMalloc(&input_ptrs_device, num_tensors * sizeof(void*));
    cudaMalloc(&dim_sizes_device, num_tensors * sizeof(int64_t));
    
    cudaMemcpy(input_ptrs_device, input_ptrs_host.data(), 
               num_tensors * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(dim_sizes_device, dim_sizes_host.data(),
               num_tensors * sizeof(int64_t), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    DType dtype = output.dtype();
    
    int64_t outer_size = 1;
    for (int d = 0; d < dim; ++d) {
        outer_size *= shape[d];
    }
    
    int64_t inner_size = 1;
    for (int d = dim + 1; d < ndim; ++d) {
        inner_size *= shape[d];
    }
    
    int64_t total_cat_dim = output.shape()[dim];
    
    if (dim == 0 && dtype == DType::Float32) {
        int64_t elements_per_slice = total_elements / total_cat_dim;
        cat_dim0_kernel<float><<<blocks, threads>>>(
            output.data_ptr<float>(),
            num_tensors,
            reinterpret_cast<const float* const*>(input_ptrs_device),
            dim_sizes_device,
            elements_per_slice,
            total_elements);
    } else if (dim == 0 && dtype == DType::Int32) {
        int64_t elements_per_slice = total_elements / total_cat_dim;
        cat_dim0_kernel<int32_t><<<blocks, threads>>>(
            output.data_ptr<int32_t>(),
            num_tensors,
            reinterpret_cast<const int32_t* const*>(input_ptrs_device),
            dim_sizes_device,
            elements_per_slice,
            total_elements);
    } else if (dim == 1 && dtype == DType::Float32) {
        cat_dim1_kernel<float><<<blocks, threads>>>(
            output.data_ptr<float>(),
            num_tensors,
            reinterpret_cast<const float* const*>(input_ptrs_device),
            dim_sizes_device,
            shape[0],
            total_cat_dim,
            inner_size,
            total_elements);
    } else if (dim == 1 && dtype == DType::Int32) {
        cat_dim1_kernel<int32_t><<<blocks, threads>>>(
            output.data_ptr<int32_t>(),
            num_tensors,
            reinterpret_cast<const int32_t* const*>(input_ptrs_device),
            dim_sizes_device,
            shape[0],
            total_cat_dim,
            inner_size,
            total_elements);
    } else if (dtype == DType::Float32) {
        cat_generic_kernel<float><<<blocks, threads>>>(
            output.data_ptr<float>(),
            num_tensors,
            reinterpret_cast<const float* const*>(input_ptrs_device),
            dim_sizes_device,
            dim,
            outer_size,
            inner_size,
            total_cat_dim,
            total_elements);
    } else if (dtype == DType::Int32) {
        cat_generic_kernel<int32_t><<<blocks, threads>>>(
            output.data_ptr<int32_t>(),
            num_tensors,
            reinterpret_cast<const int32_t* const*>(input_ptrs_device),
            dim_sizes_device,
            dim,
            outer_size,
            inner_size,
            total_cat_dim,
            total_elements);
    } else {
        cudaFree(input_ptrs_device);
        cudaFree(dim_sizes_device);
        VESPER_CHECK(false, "Unsupported dtype for cat");
    }
    
    cudaDeviceSynchronize();
    
    cudaFree(input_ptrs_device);
    cudaFree(dim_sizes_device);
}

} // namespace vesper::ops
