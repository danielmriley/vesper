#include <cuda_runtime.h>
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

struct TensorMetadata {
    int64_t shape[MAX_DIMS];
    int64_t strides[MAX_DIMS];
    int dims;
};

// Functors
struct Add { 
    __device__ float operator()(float a, float b) const { return a + b; }
    __device__ float4 operator()(float4 a, float4 b) const { 
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); 
    }
    __device__ float4 operator()(float4 a, float b) const { 
        return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); 
    }
};
struct Sub { 
    __device__ float operator()(float a, float b) const { return a - b; }
    __device__ float4 operator()(float4 a, float4 b) const { 
        return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); 
    }
    __device__ float4 operator()(float4 a, float b) const { 
        return make_float4(a.x - b, a.y - b, a.z - b, a.w - b); 
    }
};
struct Mul { 
    __device__ float operator()(float a, float b) const { return a * b; }
    __device__ float4 operator()(float4 a, float4 b) const { 
        return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); 
    }
    __device__ float4 operator()(float4 a, float b) const { 
        return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); 
    }
};
struct Div { 
    __device__ float operator()(float a, float b) const { return a / b; }
    __device__ float4 operator()(float4 a, float4 b) const { 
        return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); 
    }
    __device__ float4 operator()(float4 a, float b) const { 
        return make_float4(a.x / b, a.y / b, a.z / b, a.w / b); 
    }
};
struct Sqrt {
    __device__ float operator()(float a) const { return sqrtf(a); }
    __device__ float4 operator()(float4 a) const {
        return make_float4(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w));
    }
};
struct Sign {
    __device__ float operator()(float a) const { 
        return (a > 0.0f) ? 1.0f : ((a < 0.0f) ? -1.0f : 0.0f); 
    }
    __device__ float4 operator()(float4 a) const {
        auto s = [](float v) { return (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f); };
        return make_float4(s(a.x), s(a.y), s(a.z), s(a.w));
    }
};

struct Gelu {
    __device__ float operator()(float x) const {
        const float k0 = 0.7978845608028654f; // sqrt(2/pi)
        const float k1 = 0.044715f;
        return 0.5f * x * (1.0f + tanh(k0 * (x + k1 * x * x * x)));
    }
    __device__ float4 operator()(float4 a) const {
        auto g = [](float x) {
            const float k0 = 0.7978845608028654f;
            const float k1 = 0.044715f;
            return 0.5f * x * (1.0f + tanh(k0 * (x + k1 * x * x * x)));
        };
        return make_float4(g(a.x), g(a.y), g(a.z), g(a.w));
    }
};

struct Exp {
    __device__ float operator()(float a) const { return expf(a); }
    __device__ float4 operator()(float4 a) const {
        return make_float4(expf(a.x), expf(a.y), expf(a.z), expf(a.w));
    }
};

struct Log {
    __device__ float operator()(float a) const { return logf(a); }
    __device__ float4 operator()(float4 a) const {
        return make_float4(logf(a.x), logf(a.y), logf(a.z), logf(a.w));
    }
};

struct Cos {
    __device__ float operator()(float a) const { return cosf(a); }
    __device__ float4 operator()(float4 a) const {
        return make_float4(cosf(a.x), cosf(a.y), cosf(a.z), cosf(a.w));
    }
};

struct Sin {
    __device__ float operator()(float a) const { return sinf(a); }
    __device__ float4 operator()(float4 a) const {
        return make_float4(sinf(a.x), sinf(a.y), sinf(a.z), sinf(a.w));
    }
};

// Generic kernel for any element-wise binary operation with broadcasting support
// We assume up to 4 dims.
// shape is the shape of the OUTPUT tensor.
// strides_a/b are the broadcast-compatible strides for inputs.
template <typename T, typename Op>
__global__ void elementwise_broadcast_kernel(
    const T* a, TensorMetadata meta_a,
    const T* b, TensorMetadata meta_b,
    T* out, TensorMetadata meta_out,
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
        for (int i = meta_out.dims - 1; i >= 0; --i) {
            size_t coord = remaining % meta_out.shape[i];
            remaining /= meta_out.shape[i];
            offset_a += coord * meta_a.strides[i];
            offset_b += coord * meta_b.strides[i];
        }

        out[idx] = op(a[offset_a], b[offset_b]);
    }
}

// Vectorized Binary Kernel
template <typename Op>
__global__ void elementwise_binary_kernel_vectorized(
    const float* a,
    const float* b,
    float* out,
    size_t n_vec, Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        float4 val_a = vesper::core::load_float4(a + idx * 4);
        float4 val_b = vesper::core::load_float4(b + idx * 4);
        float4 res = op(val_a, val_b);
        vesper::core::store_float4(out + idx * 4, res);
    }
}

// Unary Kernel
template <typename T, typename Op>
__global__ void elementwise_unary_kernel(
    const T* a, TensorMetadata meta_a,
    T* out, TensorMetadata meta_out,
    size_t total_elements, Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        size_t offset_a = 0;
        size_t remaining = idx;

        #pragma unroll
        for (int i = meta_out.dims - 1; i >= 0; --i) {
            size_t coord = remaining % meta_out.shape[i];
            remaining /= meta_out.shape[i];
            offset_a += coord * meta_a.strides[i];
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

// Helper to copy vector to device array
// void copy_vec_to_dev(const std::vector<int64_t>& vec, int64_t* dev_ptr) {
//     cudaMemcpy(dev_ptr, vec.data(), vec.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
// }

bool is_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 16) == 0;
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

    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());

    // Check for vectorization
    // We can vectorize if both inputs and output are contiguous and aligned
    // Or if one input is broadcasted as a scalar (stride 0) - but that's handled by scalar kernel usually.
    // Here we check if strides match contiguous layout.
    // Actually, simpler check: a.is_contiguous() && b.is_contiguous() && out.is_contiguous()
    // But strides_a might be different from a.strides() if we are broadcasting?
    // The caller passes strides_a/b which are the broadcasted strides.
    // If strides_a corresponds to a contiguous tensor of shape `out.shape()`, then it's contiguous.
    // But `a` might be smaller.
    // Vectorization is safe if:
    // 1. `out` is contiguous.
    // 2. `a` is contiguous and same shape as `out` (so strides_a are standard).
    // 3. `b` is contiguous and same shape as `out`.
    // OR if `a` or `b` are scalars (stride 0). But we need to handle that in the functor.
    // For now, let's just handle the case where A and B are same shape as Out and contiguous.
    
    bool a_contig = a.is_contiguous() && a.shape() == out.shape();
    bool b_contig = b.is_contiguous() && b.shape() == out.shape();
    bool out_contig = out.is_contiguous();

    if (a_contig && b_contig && out_contig && 
        is_aligned(a.data_ptr<float>()) && 
        is_aligned(b.data_ptr<float>()) && 
        is_aligned(out.data_ptr<float>()) && 
        n % 4 == 0) 
    {
        size_t n_vec = n / 4;
        const int threads = 256;
        const int blocks = (n_vec + threads - 1) / threads;
        elementwise_binary_kernel_vectorized<Op><<<dim3(blocks), dim3(threads), 0, stream>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n_vec, op
        );
        return;
    }

    TensorMetadata meta_a, meta_b, meta_out;
    meta_out.dims = dims;
    meta_a.dims = dims;
    meta_b.dims = dims;

    for (int i = 0; i < dims; ++i) {
        meta_out.shape[i] = out.shape()[i];
        meta_out.strides[i] = out.strides()[i];
        meta_a.strides[i] = strides_a[i];
        meta_b.strides[i] = strides_b[i];
    }

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    elementwise_broadcast_kernel<float, Op><<<dim3(blocks), dim3(threads), 0, stream>>>(
        a.data_ptr<const float>(), meta_a,
        b.data_ptr<const float>(), meta_b,
        out.data_ptr<float>(), meta_out,
        n, op);
}

// Scalar Kernel
// output is assumed contiguous. input `a` can be strided.
// shape is the shape of both a and out.
template <typename T, typename Op>
__global__ void elementwise_scalar_kernel(
    const T* a, TensorMetadata meta_a,
    T* out, TensorMetadata meta_out,
    size_t total_elements, T scalar, Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        size_t offset_a = 0;
        size_t remaining = idx;

        #pragma unroll
        for (int i = meta_out.dims - 1; i >= 0; --i) {
            size_t coord = remaining % meta_out.shape[i];
            remaining /= meta_out.shape[i];
            offset_a += coord * meta_a.strides[i];
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

template<typename Op>
void launch_scalar_kernel(const Tensor& a, float b, Tensor& out, Op op) {
    size_t n = out.numel();
    
    // Check for vectorization
    bool can_vectorize = (n % 4 == 0) && 
                         is_aligned(a.data_ptr<float>()) && 
                         is_aligned(out.data_ptr<float>()) &&
                         a.is_contiguous();

    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());

    if (can_vectorize) {
        size_t n_vec = n / 4;
        const int threads = 256;
        const int blocks = (n_vec + threads - 1) / threads;
        elementwise_scalar_kernel_vectorized<Op><<<dim3(blocks), dim3(threads), 0, stream>>>(
            a.data_ptr<float>(), out.data_ptr<float>(), n_vec, b, op
        );
        return;
    }

    // Fallback
    int dims = out.shape().size();
    
    if (dims > MAX_DIMS) {
        throw std::runtime_error("Scalar op currently supports up to 4 dimensions on GPU.");
    }

    TensorMetadata meta_a, meta_out;
    meta_out.dims = dims;
    meta_a.dims = dims;

    for (int i = 0; i < dims; ++i) {
        meta_out.shape[i] = out.shape()[i];
        meta_out.strides[i] = out.strides()[i];
        meta_a.strides[i] = a.strides()[i];
    }

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    elementwise_scalar_kernel<float, Op><<<dim3(blocks), dim3(threads), 0, stream>>>(
        a.data_ptr<const float>(), meta_a,
        out.data_ptr<float>(), meta_out,
        n, b, op);
}

template<typename Op>
void launch_unary_kernel(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out, Op op) {
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
        elementwise_unary_kernel_vectorized<Op><<<dim3(blocks), dim3(threads), 0, stream>>>(
            a.data_ptr<float>(), out.data_ptr<float>(), n_vec, op
        );
        return;
    }

    int dims = out.shape().size();
    if (dims > MAX_DIMS) {
        throw std::runtime_error("Unary op currently supports up to 4 dimensions on GPU.");
    }

    TensorMetadata meta_a, meta_out;
    meta_out.dims = dims;
    meta_a.dims = dims;

    for (int i = 0; i < dims; ++i) {
        meta_out.shape[i] = out.shape()[i];
        meta_out.strides[i] = out.strides()[i];
        meta_a.strides[i] = strides_a[i];
    }

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    elementwise_unary_kernel<float, Op><<<dim3(blocks), dim3(threads), 0, stream>>>(
        a.data_ptr<const float>(), meta_a,
        out.data_ptr<float>(), meta_out,
        n, op);
}

void add_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, Add());
}

void add_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel(a, b, out, Add());
}

void sub_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, Sub());
}

void sub_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel(a, b, out, Sub());
}

void mul_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, Mul());
}

void mul_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel(a, b, out, Mul());
}

void div_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, Div());
}

void div_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel(a, b, out, Div());
}

void sqrt_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Sqrt());
}

void sign_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Sign());
}

void gelu_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Gelu());
}

void exp_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Exp());
}

void log_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Log());
}

void cos_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Cos());
}

void sin_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Sin());
}

} // namespace vesper::ops
