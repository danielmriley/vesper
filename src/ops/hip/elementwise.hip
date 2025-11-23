#include <hip/hip_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/ops/elementwise.h>
#include <vesper/core/vectorization.h>
#include <vesper/core/stream.h>
#include <functional>
#include <vector>

namespace vesper::ops {

// Helper to compute multi-dimensional index from linear index and strides
// Note: We only support up to MAX_DIMS dimensions for simplicity in the kernel.
constexpr int MAX_DIMS = 4; 

// Functors
struct Add { 
    __device__ float operator()(float a, float b) const { return a + b; }
    __device__ float4 operator()(float4 a, float b) const { 
        return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); 
    }
};
struct Sub { 
    __device__ float operator()(float a, float b) const { return a - b; }
    __device__ float4 operator()(float4 a, float b) const { 
        return make_float4(a.x - b, a.y - b, a.z - b, a.w - b); 
    }
};
struct Mul { 
    __device__ float operator()(float a, float b) const { return a * b; }
    __device__ float4 operator()(float4 a, float b) const { 
        return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); 
    }
};
struct Div { 
    __device__ float operator()(float a, float b) const { return a / b; }
    __device__ float4 operator()(float4 a, float b) const { 
        return make_float4(a.x / b, a.y / b, a.z / b, a.w / b); 
    }
};

// Generic kernel for any element-wise binary operation with broadcasting support
// We assume up to 4 dims.
// shape is the shape of the OUTPUT tensor.
// strides_a/b are the broadcast-compatible strides for inputs.
template <typename T, typename Op>
__global__ void elementwise_broadcast_kernel(
    const T* a, const int64_t* strides_a, 
    const T* b, const int64_t* strides_b, 
    T* out, 
    const int64_t* shape, int dims,
    size_t total_elements, Op op) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        // Convert linear index `idx` (of output) to offsets for a and b
        size_t offset_a = 0;
        size_t offset_b = 0;
        size_t remaining = idx;

        // Manual unroll or loop for coordinate transform
        // We iterate from last dim to first
        #pragma unroll
        for (int i = dims - 1; i >= 0; --i) {
            size_t coord = remaining % shape[i];
            remaining /= shape[i];
            offset_a += coord * strides_a[i];
            offset_b += coord * strides_b[i];
        }

        out[idx] = op(a[offset_a], b[offset_b]);
    }
}

// Helper to copy vector to device array
void copy_vec_to_dev(const std::vector<int64_t>& vec, int64_t* dev_ptr) {
    hipMemcpy(dev_ptr, vec.data(), vec.size() * sizeof(int64_t), hipMemcpyHostToDevice);
}

template<typename Op>
void launch_broadcast_kernel(const Tensor& a, const std::vector<int64_t>& strides_a, 
                           const Tensor& b, const std::vector<int64_t>& strides_b, 
                           Tensor& out, Op op) {
    size_t n = out.numel();
    int dims = out.shape().size();
    
    if (dims > MAX_DIMS) {
        throw std::runtime_error("Broadcasting currently supports up to 4 dimensions on GPU.");
    }

    int64_t* d_shape;
    int64_t* d_strides_a;
    int64_t* d_strides_b;
    
    hipMalloc(&d_shape, dims * sizeof(int64_t));
    hipMalloc(&d_strides_a, dims * sizeof(int64_t));
    hipMalloc(&d_strides_b, dims * sizeof(int64_t));
    
    copy_vec_to_dev(out.shape(), d_shape);
    copy_vec_to_dev(strides_a, d_strides_a);
    copy_vec_to_dev(strides_b, d_strides_b);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    hipLaunchKernelGGL(elementwise_broadcast_kernel<float, Op>, 
        dim3(blocks), dim3(threads), 0, stream,
        a.data_ptr<const float>(), d_strides_a,
        b.data_ptr<const float>(), d_strides_b,
        out.data_ptr<float>(),
        d_shape, dims, n, op);
        
    // Cleanup
    hipFree(d_shape);
    hipFree(d_strides_a);
    hipFree(d_strides_b);
}

// Scalar Kernel
// output is assumed contiguous. input `a` can be strided.
// shape is the shape of both a and out.
template <typename T, typename Op>
__global__ void elementwise_scalar_kernel(
    const T* a, const int64_t* strides_a,
    T* out,
    const int64_t* shape, int dims,
    size_t total_elements, T scalar, Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        size_t offset_a = 0;
        size_t remaining = idx;

        #pragma unroll
        for (int i = dims - 1; i >= 0; --i) {
            size_t coord = remaining % shape[i];
            remaining /= shape[i];
            offset_a += coord * strides_a[i];
        }

        out[idx] = op(a[offset_a], scalar);
    }
}

// Vectorized Scalar Kernel (float4)
template <typename Op>
__global__ void elementwise_scalar_kernel_vectorized(
    const float* a,
    float* out,
    size_t n_vec, float scalar, Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        float4 val_a = vesper::core::load_float4(a + idx * 4);
        float4 res = op(val_a, scalar);
        vesper::core::store_float4(out + idx * 4, res);
    }
}

bool is_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 16) == 0;
}

template<typename Op>
void launch_scalar_kernel(const Tensor& a, float b, Tensor& out, Op op) {
    size_t n = out.numel();
    
    // Check for vectorization
    bool can_vectorize = (n % 4 == 0) && 
                         is_aligned(a.data_ptr<float>()) && 
                         is_aligned(out.data_ptr<float>()) &&
                         a.is_contiguous();

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    if (can_vectorize) {
        size_t n_vec = n / 4;
        const int threads = 256;
        const int blocks = (n_vec + threads - 1) / threads;
        hipLaunchKernelGGL(elementwise_scalar_kernel_vectorized<Op>, 
            dim3(blocks), dim3(threads), 0, stream,
            a.data_ptr<float>(), out.data_ptr<float>(), n_vec, b, op
        );
        return;
    }

    // Fallback
    int dims = out.shape().size();
    
    if (dims > MAX_DIMS) {
        throw std::runtime_error("Scalar op currently supports up to 4 dimensions on GPU.");
    }

    int64_t* d_shape;
    int64_t* d_strides_a;
    
    hipMalloc(&d_shape, dims * sizeof(int64_t));
    hipMalloc(&d_strides_a, dims * sizeof(int64_t));
    
    copy_vec_to_dev(out.shape(), d_shape);
    copy_vec_to_dev(a.strides(), d_strides_a);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    hipLaunchKernelGGL(elementwise_scalar_kernel<float, Op>, 
        dim3(blocks), dim3(threads), 0, stream,
        a.data_ptr<const float>(), d_strides_a,
        out.data_ptr<float>(),
        d_shape, dims, n, b, op);
        
    hipFree(d_shape);
    hipFree(d_strides_a);
}

void add_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, std::plus<float>()); // Note: std::plus is not a device functor here, need struct
}

// Wait, std::plus works in CUDA if using __device__ lambda or if supported.
// But in HIP we used `std::plus<float>` in `add_hip_dispatch` before?
// Let's check the previous file content.
// Previous: `launch_broadcast_kernel(..., std::plus<float>())`
// And `elementwise_broadcast_kernel` template Op.
// Does `std::plus` have `__device__` operator()? Usually no.
// But HIP might support it if it's just a template instantiation?
// No, typically we need custom functors.
// The previous code `elementwise_binary_kernel<float, std::plus<float>>` worked because `hipLaunchKernelGGL` might be magic or I missed something?
// Ah, `std::plus` is NOT `__device__`.
// The previous implementation `src/ops/hip/elementwise.hip` used `std::plus<float>`.
// If that compiled, it's surprising.
// But to be safe and support `float4`, I should use my custom structs `Add`, `Sub` etc. defined above.

void add_hip_dispatch_v2(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, Add());
}

// I will replace the dispatches to use my structs.

void add_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, Add());
}

void add_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel(a, b, out, Add());
}

void sub_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, Sub());
}

void sub_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel(a, b, out, Sub());
}

void mul_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, Mul());
}

void mul_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel(a, b, out, Mul());
}

void div_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, Div());
}

void div_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel(a, b, out, Div());
}

} // namespace vesper::ops