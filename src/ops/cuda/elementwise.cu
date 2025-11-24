#include <cuda_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/ops/elementwise.h>
#include <vesper/core/vectorization.h>
#include <vesper/core/stream.h>
#include <vector>
#include <stdexcept>

namespace vesper::ops {

constexpr int MAX_DIMS = 4; 

// Functors
template <typename T> struct Add { 
    __device__ T operator()(const T& a, const T& b) const { return a + b; }
    __device__ float4 operator()(const float4& a, float b) const { 
        return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); 
    }
};
template <typename T> struct Sub { 
    __device__ T operator()(const T& a, const T& b) const { return a - b; } 
    __device__ float4 operator()(const float4& a, float b) const { 
        return make_float4(a.x - b, a.y - b, a.z - b, a.w - b); 
    }
};
template <typename T> struct Mul { 
    __device__ T operator()(const T& a, const T& b) const { return a * b; } 
    __device__ float4 operator()(const float4& a, float b) const { 
        return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); 
    }
};
template <typename T> struct Div { 
    __device__ T operator()(const T& a, const T& b) const { return a / b; } 
    __device__ float4 operator()(const float4& a, float b) const { 
        return make_float4(a.x / b, a.y / b, a.z / b, a.w / b); 
    }
};
template <typename T> struct Sqrt {
    __device__ T operator()(const T& a) const { return sqrtf(a); }
    __device__ float4 operator()(const float4& a) const {
        return make_float4(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w));
    }
};
template <typename T> struct Sign {
    __device__ T operator()(const T& a) const { 
        return (a > 0.0f) ? 1.0f : ((a < 0.0f) ? -1.0f : 0.0f); 
    }
    __device__ float4 operator()(const float4& a) const {
        auto s = [](float v) { return (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f); };
        return make_float4(s(a.x), s(a.y), s(a.z), s(a.w));
    }
};

// ... (elementwise_broadcast_kernel remains same for now) ...

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
        size_t offset_a = 0;
        size_t offset_b = 0;
        size_t remaining = idx;

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

// Unary Kernel
template <typename T, typename Op>
__global__ void elementwise_unary_kernel(
    const T* a, const int64_t* strides_a,
    T* out,
    const int64_t* shape, int dims,
    size_t total_elements, Op op)
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

        out[idx] = op(a[offset_a]);
    }
}

// Vectorized Unary Kernel
template <typename Op>
__global__ void elementwise_unary_kernel_vectorized(
    const float* a,
    float* out,
    size_t n_vec, Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        float4 val_a = vesper::core::load_float4(a + idx * 4);
        float4 res = op(val_a);
        vesper::core::store_float4(out + idx * 4, res);
    }
}

void copy_vec_to_dev_cuda(const std::vector<int64_t>& vec, int64_t* dev_ptr) {
    cudaMemcpy(dev_ptr, vec.data(), vec.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
}

template<typename Op>
void launch_broadcast_kernel_cuda(const Tensor& a, const std::vector<int64_t>& strides_a, 
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
    
    cudaMalloc(&d_shape, dims * sizeof(int64_t));
    cudaMalloc(&d_strides_a, dims * sizeof(int64_t));
    cudaMalloc(&d_strides_b, dims * sizeof(int64_t));
    
    copy_vec_to_dev_cuda(out.shape(), d_shape);
    copy_vec_to_dev_cuda(strides_a, d_strides_a);
    copy_vec_to_dev_cuda(strides_b, d_strides_b);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());

    elementwise_broadcast_kernel<<<blocks, threads, 0, stream>>>(
        a.data_ptr<const float>(), d_strides_a,
        b.data_ptr<const float>(), d_strides_b,
        out.data_ptr<float>(),
        d_shape, dims, n, op);
        
    cudaFree(d_shape);
    cudaFree(d_strides_a);
    cudaFree(d_strides_b);
}

// Scalar Kernel (Scalar)
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
// Assumes a and out are contiguous and aligned.
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

bool is_contiguous_layout(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides) {
    int64_t current_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        if (strides[i] != current_stride) return false;
        current_stride *= shape[i];
    }
    return true;
}

bool is_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 16) == 0;
}

template<typename Op>
void launch_scalar_kernel_cuda(const Tensor& a, float b, Tensor& out, Op op) {
    size_t n = out.numel();
    
    // Check for vectorization opportunity
    bool can_vectorize = (n % 4 == 0) && 
                         is_aligned(a.data_ptr<float>()) && 
                         is_aligned(out.data_ptr<float>()) &&
                         a.is_contiguous(); 

    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());

    if (can_vectorize) {
        size_t n_vec = n / 4;
        const int threads = 256;
        const int blocks = (n_vec + threads - 1) / threads;
        elementwise_scalar_kernel_vectorized<<<blocks, threads, 0, stream>>>(
            a.data_ptr<float>(), out.data_ptr<float>(), n_vec, b, op
        );
        return;
    }

    // Fallback to scalar strided kernel
    int dims = out.shape().size();
    if (dims > MAX_DIMS) {
        throw std::runtime_error("Scalar op currently supports up to 4 dimensions on GPU.");
    }

    int64_t* d_shape;
    int64_t* d_strides_a;
    
    cudaMalloc(&d_shape, dims * sizeof(int64_t));
    cudaMalloc(&d_strides_a, dims * sizeof(int64_t));
    
    copy_vec_to_dev_cuda(out.shape(), d_shape);
    copy_vec_to_dev_cuda(a.strides(), d_strides_a);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    elementwise_scalar_kernel<<<blocks, threads, 0, stream>>>(
        a.data_ptr<const float>(), d_strides_a,
        out.data_ptr<float>(),
        d_shape, dims, n, b, op);
        
    cudaFree(d_shape);
    cudaFree(d_strides_a);
}

template<typename Op>
void launch_unary_kernel_cuda(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out, Op op) {
    size_t n = out.numel();
    
    bool can_vectorize = (n % 4 == 0) && 
                         is_aligned(a.data_ptr<float>()) && 
                         is_aligned(out.data_ptr<float>()) &&
                         a.is_contiguous() &&
                         out.is_contiguous();

    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());

    if (can_vectorize) {
        size_t n_vec = n / 4;
        const int threads = 256;
        const int blocks = (n_vec + threads - 1) / threads;
        elementwise_unary_kernel_vectorized<<<blocks, threads, 0, stream>>>(
            a.data_ptr<float>(), out.data_ptr<float>(), n_vec, op
        );
        return;
    }

    int dims = out.shape().size();
    if (dims > MAX_DIMS) {
        throw std::runtime_error("Unary op currently supports up to 4 dimensions on GPU.");
    }

    int64_t* d_shape;
    int64_t* d_strides_a;
    
    cudaMalloc(&d_shape, dims * sizeof(int64_t));
    cudaMalloc(&d_strides_a, dims * sizeof(int64_t));
    
    copy_vec_to_dev_cuda(out.shape(), d_shape);
    copy_vec_to_dev_cuda(strides_a, d_strides_a);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    elementwise_unary_kernel<<<blocks, threads, 0, stream>>>(
        a.data_ptr<const float>(), d_strides_a,
        out.data_ptr<float>(),
        d_shape, dims, n, op);
        
    cudaFree(d_shape);
    cudaFree(d_strides_a);
}

void add_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel_cuda(a, strides_a, b, strides_b, out, Add<float>());
}

void add_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel_cuda(a, b, out, Add<float>());
}

void sub_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel_cuda(a, strides_a, b, strides_b, out, Sub<float>());
}

void sub_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel_cuda(a, b, out, Sub<float>());
}

void mul_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel_cuda(a, strides_a, b, strides_b, out, Mul<float>());
}

void mul_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel_cuda(a, b, out, Mul<float>());
}

void div_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel_cuda(a, strides_a, b, strides_b, out, Div<float>());
}

void div_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel_cuda(a, b, out, Div<float>());
}

void sqrt_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel_cuda(a, strides_a, out, Sqrt<float>());
}

void sign_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel_cuda(a, strides_a, out, Sign<float>());
}

} // namespace vesper::ops