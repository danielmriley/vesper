#include <hip/hip_runtime.h>
#include <vesper/ops/cat.h>
#include <vesper/core/macros.h>

namespace vesper::ops {

/**
 * GPU kernel for concatenating tensors along dimension 0.
 * Each thread handles one element of the output.
 * 
 * @param output Output buffer
 * @param num_tensors Number of input tensors
 * @param input_ptrs Array of pointers to input tensor data
 * @param input_dim0_sizes Array of sizes along dimension 0 for each input
 * @param elements_per_slice Number of elements per slice (product of dims 1..N)
 * @param total_elements Total elements in output
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
    
    // Find which slice this element belongs to
    int64_t slice_idx = idx / elements_per_slice;
    int64_t within_slice = idx % elements_per_slice;
    
    // Find which tensor this slice belongs to
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
    
    // Read from input tensor
    const T* input_data = input_ptrs[tensor_idx];
    output[idx] = input_data[slice_offset * elements_per_slice + within_slice];
}

/**
 * GPU kernel for concatenating tensors along dimension 1.
 * Optimized for the common case of appending tokens to sequences.
 * 
 * @param output Output buffer [batch, total_seq_len, ...]
 * @param num_tensors Number of input tensors
 * @param input_ptrs Array of pointers to input tensor data
 * @param input_dim1_sizes Array of sizes along dimension 1 for each input
 * @param batch_size Size of dimension 0
 * @param elements_per_token Number of elements per position (product of dims 2..N)
 * @param total_elements Total elements in output
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
    
    // Calculate position in output tensor
    int64_t elements_per_batch = total_dim1 * elements_per_token;
    int64_t batch_idx = idx / elements_per_batch;
    int64_t within_batch = idx % elements_per_batch;
    int64_t seq_idx = within_batch / elements_per_token;
    int64_t within_token = within_batch % elements_per_token;
    
    // Find which tensor this sequence position belongs to
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
    
    // Read from input tensor
    const T* input_data = input_ptrs[tensor_idx];
    int64_t input_elements_per_batch = input_dim1_sizes[tensor_idx] * elements_per_token;
    int64_t input_idx = batch_idx * input_elements_per_batch + local_seq_idx * elements_per_token + within_token;
    
    output[idx] = input_data[input_idx];
}

/**
 * Generic kernel for concatenating along any dimension.
 * Less efficient than specialized kernels but handles all cases.
 */
template<typename T>
__global__ void cat_generic_kernel(
    T* output,
    int num_tensors,
    const T* const* input_ptrs,
    const int64_t* input_dim_sizes,  // Size of cat dimension for each input
    int cat_dim,
    int64_t outer_size,    // Product of dims before cat_dim
    int64_t inner_size,    // Product of dims after cat_dim
    int64_t total_cat_dim, // Sum of input dim sizes
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Decompose linear index into outer, cat, inner
    int64_t inner_idx = idx % inner_size;
    int64_t remaining = idx / inner_size;
    int64_t cat_idx = remaining % total_cat_dim;
    int64_t outer_idx = remaining / total_cat_dim;
    
    // Find which tensor
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
    
    // Compute input index
    const T* input_data = input_ptrs[tensor_idx];
    int64_t input_cat_size = input_dim_sizes[tensor_idx];
    int64_t input_idx = outer_idx * (input_cat_size * inner_size) + local_cat_idx * inner_size + inner_idx;
    
    output[idx] = input_data[input_idx];
}

void cat_hip_dispatch(const std::vector<Tensor>& inputs, Tensor& output, int dim) {
    VESPER_CHECK(!inputs.empty(), "Cannot cat empty tensor list");
    
    int num_tensors = static_cast<int>(inputs.size());
    const auto& shape = inputs[0].shape();
    int ndim = static_cast<int>(shape.size());
    
    // Handle negative dimension
    if (dim < 0) dim += ndim;
    VESPER_CHECK(dim >= 0 && dim < ndim, "Invalid dimension for cat");
    
    // Get total elements
    int64_t total_elements = output.numel();
    if (total_elements == 0) return;
    
    // Allocate device memory for metadata
    std::vector<const void*> input_ptrs_host(num_tensors);
    std::vector<int64_t> dim_sizes_host(num_tensors);
    
    for (int i = 0; i < num_tensors; ++i) {
        input_ptrs_host[i] = inputs[i].data_ptr<void>();
        dim_sizes_host[i] = inputs[i].shape()[dim];
    }
    
    void** input_ptrs_device;
    int64_t* dim_sizes_device;
    
    hipMalloc(&input_ptrs_device, num_tensors * sizeof(void*));
    hipMalloc(&dim_sizes_device, num_tensors * sizeof(int64_t));
    
    hipMemcpy(input_ptrs_device, input_ptrs_host.data(), 
              num_tensors * sizeof(void*), hipMemcpyHostToDevice);
    hipMemcpy(dim_sizes_device, dim_sizes_host.data(),
              num_tensors * sizeof(int64_t), hipMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    DType dtype = output.dtype();
    
    // Calculate outer and inner sizes for generic kernel
    int64_t outer_size = 1;
    for (int d = 0; d < dim; ++d) {
        outer_size *= shape[d];
    }
    
    int64_t inner_size = 1;
    for (int d = dim + 1; d < ndim; ++d) {
        inner_size *= shape[d];
    }
    
    int64_t total_cat_dim = output.shape()[dim];
    
    // Use specialized kernels for common cases
    if (dim == 0 && dtype == DType::Float32) {
        int64_t elements_per_slice = total_elements / total_cat_dim;
        hipLaunchKernelGGL(cat_dim0_kernel<float>, dim3(blocks), dim3(threads), 0, 0,
            output.data_ptr<float>(),
            num_tensors,
            reinterpret_cast<const float* const*>(input_ptrs_device),
            dim_sizes_device,
            elements_per_slice,
            total_elements);
    } else if (dim == 0 && dtype == DType::Int32) {
        int64_t elements_per_slice = total_elements / total_cat_dim;
        hipLaunchKernelGGL(cat_dim0_kernel<int32_t>, dim3(blocks), dim3(threads), 0, 0,
            output.data_ptr<int32_t>(),
            num_tensors,
            reinterpret_cast<const int32_t* const*>(input_ptrs_device),
            dim_sizes_device,
            elements_per_slice,
            total_elements);
    } else if (dim == 1 && dtype == DType::Float32) {
        hipLaunchKernelGGL(cat_dim1_kernel<float>, dim3(blocks), dim3(threads), 0, 0,
            output.data_ptr<float>(),
            num_tensors,
            reinterpret_cast<const float* const*>(input_ptrs_device),
            dim_sizes_device,
            shape[0],  // batch_size
            total_cat_dim,
            inner_size,
            total_elements);
    } else if (dim == 1 && dtype == DType::Int32) {
        hipLaunchKernelGGL(cat_dim1_kernel<int32_t>, dim3(blocks), dim3(threads), 0, 0,
            output.data_ptr<int32_t>(),
            num_tensors,
            reinterpret_cast<const int32_t* const*>(input_ptrs_device),
            dim_sizes_device,
            shape[0],  // batch_size
            total_cat_dim,
            inner_size,
            total_elements);
    } else if (dtype == DType::Float32) {
        hipLaunchKernelGGL(cat_generic_kernel<float>, dim3(blocks), dim3(threads), 0, 0,
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
        hipLaunchKernelGGL(cat_generic_kernel<int32_t>, dim3(blocks), dim3(threads), 0, 0,
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
        hipFree(input_ptrs_device);
        hipFree(dim_sizes_device);
        VESPER_CHECK(false, "Unsupported dtype for cat");
    }
    
    hipDeviceSynchronize();
    
    hipFree(input_ptrs_device);
    hipFree(dim_sizes_device);
}

} // namespace vesper::ops
