#include <hip/hip_runtime.h>
#include <vesper/ops/cat.h>
#include <vesper/core/macros.h>
#include <vesper/core/stream.h>
#include <vesper/core/allocator.h>
#include <vector>

namespace vesper::ops {

// Number of input tensors whose data pointers + cat-dim sizes are passed inline
// by value through the kernel argument buffer. HIP allows a few KB of kernel
// args; CatInputs<T> is ~272 bytes, well within budget. Above this count we
// fall back to a device scratch buffer (see cat_hip_launch).
static constexpr int CAT_MAX_INLINE = 16;

// POD kernel argument carrying either the inline pointer/size arrays (common
// case) or pointers to device scratch buffers holding the full arrays (fallback
// for num_tensors > CAT_MAX_INLINE). Trivially copyable so it can be passed by
// value to a __global__ function.
template <typename T>
struct CatInputs {
    const T* ptrs[CAT_MAX_INLINE];
    int64_t dim_sizes[CAT_MAX_INLINE];
    const T* const* ptrs_ext;      // non-null only in the fallback path
    const int64_t* dim_sizes_ext;  // non-null only in the fallback path
};

/**
 * GPU kernel for concatenating tensors along dimension 0.
 * Each thread handles one element of the output.
 */
template<typename T>
__global__ void cat_dim0_kernel(
    T* output,
    int num_tensors,
    CatInputs<T> args,
    int64_t elements_per_slice,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const T* const* input_ptrs = args.ptrs_ext ? args.ptrs_ext : args.ptrs;
    const int64_t* input_dim0_sizes = args.dim_sizes_ext ? args.dim_sizes_ext : args.dim_sizes;

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
 */
template<typename T>
__global__ void cat_dim1_kernel(
    T* output,
    int num_tensors,
    CatInputs<T> args,
    int64_t batch_size,
    int64_t total_dim1,
    int64_t elements_per_token,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const T* const* input_ptrs = args.ptrs_ext ? args.ptrs_ext : args.ptrs;
    const int64_t* input_dim1_sizes = args.dim_sizes_ext ? args.dim_sizes_ext : args.dim_sizes;

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
    CatInputs<T> args,
    int cat_dim,
    int64_t outer_size,    // Product of dims before cat_dim
    int64_t inner_size,    // Product of dims after cat_dim
    int64_t total_cat_dim, // Sum of input dim sizes
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const T* const* input_ptrs = args.ptrs_ext ? args.ptrs_ext : args.ptrs;
    const int64_t* input_dim_sizes = args.dim_sizes_ext ? args.dim_sizes_ext : args.dim_sizes;

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

// Fills the CatInputs argument and launches the appropriate kernel on `stream`.
// For num_tensors <= CAT_MAX_INLINE the pointer/size lists ride inline in the
// kernel arguments (no hipMalloc / hipMemcpy). Above the threshold we stage the
// lists in a CachingAllocator scratch buffer; that path is rare (e.g. a single
// final cat over all generated tokens) so the guarding stream sync is cheap.
template <typename T>
static void cat_hip_launch(
    const std::vector<Tensor>& inputs,
    Tensor& output,
    int dim,
    int num_tensors,
    int64_t total_elements,
    int64_t outer_size,
    int64_t inner_size,
    int64_t total_cat_dim,
    int64_t batch_size,
    hipStream_t stream
) {
    CatInputs<T> args{};  // value-init: *_ext are null -> inline arrays used

    // Host staging for the fallback path; must outlive the kernel's reads, which
    // the stream sync below guarantees.
    std::vector<const T*> h_ptrs;
    std::vector<int64_t> h_dims;

    void* d_ptrs = nullptr;
    void* d_dims = nullptr;
    Allocator* alloc = nullptr;
    const bool fallback = num_tensors > CAT_MAX_INLINE;

    if (!fallback) {
        for (int i = 0; i < num_tensors; ++i) {
            args.ptrs[i] = inputs[i].data_ptr<const T>();
            args.dim_sizes[i] = inputs[i].shape()[dim];
        }
    } else {
        h_ptrs.resize(num_tensors);
        h_dims.resize(num_tensors);
        for (int i = 0; i < num_tensors; ++i) {
            h_ptrs[i] = inputs[i].data_ptr<const T>();
            h_dims[i] = inputs[i].shape()[dim];
        }
        alloc = get_allocator(Device::HIP);
        d_ptrs = alloc->allocate(num_tensors * sizeof(const T*));
        d_dims = alloc->allocate(num_tensors * sizeof(int64_t));
        (void)hipMemcpyAsync(d_ptrs, h_ptrs.data(), num_tensors * sizeof(const T*),
                             hipMemcpyHostToDevice, stream);
        (void)hipMemcpyAsync(d_dims, h_dims.data(), num_tensors * sizeof(int64_t),
                             hipMemcpyHostToDevice, stream);
        args.ptrs_ext = reinterpret_cast<const T* const*>(d_ptrs);
        args.dim_sizes_ext = reinterpret_cast<const int64_t*>(d_dims);
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total_elements + threads - 1) / threads);

    if (dim == 0) {
        int64_t elements_per_slice = total_elements / total_cat_dim;
        hipLaunchKernelGGL(cat_dim0_kernel<T>, dim3(blocks), dim3(threads), 0, stream,
            output.data_ptr<T>(), num_tensors, args, elements_per_slice, total_elements);
    } else if (dim == 1) {
        hipLaunchKernelGGL(cat_dim1_kernel<T>, dim3(blocks), dim3(threads), 0, stream,
            output.data_ptr<T>(), num_tensors, args, batch_size, total_cat_dim,
            inner_size, total_elements);
    } else {
        hipLaunchKernelGGL(cat_generic_kernel<T>, dim3(blocks), dim3(threads), 0, stream,
            output.data_ptr<T>(), num_tensors, args, dim, outer_size, inner_size,
            total_cat_dim, total_elements);
    }

    if (fallback) {
        // Scratch buffers (and host staging) must survive the kernel's reads.
        // Sync only this stream, then return the buffers to the allocator pool.
        (void)hipStreamSynchronize(stream);
        alloc->free(d_ptrs);
        alloc->free(d_dims);
    }
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
    int64_t batch_size = shape[0];  // dim 0 extent, used by the dim==1 kernel

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    DType dtype = output.dtype();
    if (dtype == DType::Float32) {
        cat_hip_launch<float>(inputs, output, dim, num_tensors, total_elements,
                              outer_size, inner_size, total_cat_dim, batch_size, stream);
    } else if (dtype == DType::Int32) {
        cat_hip_launch<int32_t>(inputs, output, dim, num_tensors, total_elements,
                                outer_size, inner_size, total_cat_dim, batch_size, stream);
    } else {
        VESPER_CHECK(false, "Unsupported dtype for cat");
    }
}

} // namespace vesper::ops
