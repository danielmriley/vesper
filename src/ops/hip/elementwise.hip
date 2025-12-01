#include <hip/hip_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/ops/elementwise.h>
#include <vesper/core/vectorization.h>
#include <vesper/core/stream.h>
#include <functional>
#include <vector>

namespace vesper::ops {

// Helper to compute multi-dimensional index from linear index and strides
// Support up to MAX_DIMS dimensions for GPU kernels.
// Increased from 4 to 8 to support higher-dimensional tensors (e.g., 5D RoPE intermediates).
constexpr int MAX_DIMS = 8; 

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

// SiLU (Swish) activation: silu(x) = x * sigmoid(x)
struct SiLU {
    __device__ float operator()(float x) const {
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        return x * sigmoid_x;
    }
    __device__ float4 operator()(float4 a) const {
        auto s = [](float x) {
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            return x * sigmoid_x;
        };
        return make_float4(s(a.x), s(a.y), s(a.z), s(a.w));
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
//     hipMemcpy(dev_ptr, vec.data(), vec.size() * sizeof(int64_t), hipMemcpyHostToDevice);
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
        throw std::runtime_error("Broadcasting currently supports up to 8 dimensions on GPU.");
    }

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

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
        hipLaunchKernelGGL(elementwise_binary_kernel_vectorized<Op>, 
            dim3(blocks), dim3(threads), 0, stream,
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

    hipLaunchKernelGGL((elementwise_broadcast_kernel<float, Op>), 
        dim3(blocks), dim3(threads), 0, stream,
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
        throw std::runtime_error("Scalar op currently supports up to 8 dimensions on GPU.");
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

    hipLaunchKernelGGL((elementwise_scalar_kernel<float, Op>), 
        dim3(blocks), dim3(threads), 0, stream,
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

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    if (can_vectorize) {
        size_t n_vec = n / 4;
        const int threads = 256;
        const int blocks = (n_vec + threads - 1) / threads;
        hipLaunchKernelGGL(elementwise_unary_kernel_vectorized<Op>, 
            dim3(blocks), dim3(threads), 0, stream,
            a.data_ptr<float>(), out.data_ptr<float>(), n_vec, op
        );
        return;
    }

    int dims = out.shape().size();
    if (dims > MAX_DIMS) {
        throw std::runtime_error("Unary op currently supports up to 8 dimensions on GPU.");
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

    hipLaunchKernelGGL((elementwise_unary_kernel<float, Op>), 
        dim3(blocks), dim3(threads), 0, stream,
        a.data_ptr<const float>(), meta_a,
        out.data_ptr<float>(), meta_out,
        n, op);
}

struct GeluBackward {
    __device__ float operator()(float grad, float x) const {
        const float k0 = 0.7978845608028654f; // sqrt(2/pi)
        const float k1 = 0.044715f;
        
        float x3 = x * x * x;
        float inner = k0 * (x + k1 * x3);
        float tanh_inner = tanh(inner);
        
        float dy_dx = k0 * (1.0f + 3.0f * k1 * x * x);
        float sech2 = 1.0f - tanh_inner * tanh_inner;
        
        float d_gelu = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * dy_dx;
        return grad * d_gelu;
    }

    __device__ float4 operator()(float4 grad, float4 x) const {
        auto op = [&](float g, float v) {
            const float k0 = 0.7978845608028654f;
            const float k1 = 0.044715f;
            float x3 = v * v * v;
            float inner = k0 * (v + k1 * x3);
            float tanh_inner = tanh(inner);
            float dy_dx = k0 * (1.0f + 3.0f * k1 * v * v);
            float sech2 = 1.0f - tanh_inner * tanh_inner;
            float d_gelu = 0.5f * (1.0f + tanh_inner) + 0.5f * v * sech2 * dy_dx;
            return g * d_gelu;
        };
        return make_float4(op(grad.x, x.x), op(grad.y, x.y), op(grad.z, x.z), op(grad.w, x.w));
    }
    
    // Fallback for mixed types if needed by template instantiation, though unlikely to be used in vectorized path
    __device__ float4 operator()(float4 grad, float x) const {
        auto op = [&](float g, float v) {
            const float k0 = 0.7978845608028654f;
            const float k1 = 0.044715f;
            float x3 = v * v * v;
            float inner = k0 * (v + k1 * x3);
            float tanh_inner = tanh(inner);
            float dy_dx = k0 * (1.0f + 3.0f * k1 * v * v);
            float sech2 = 1.0f - tanh_inner * tanh_inner;
            float d_gelu = 0.5f * (1.0f + tanh_inner) + 0.5f * v * sech2 * dy_dx;
            return g * d_gelu;
        };
        return make_float4(op(grad.x, x), op(grad.y, x), op(grad.z, x), op(grad.w, x));
    }
};

struct GeluErf {
    __device__ float operator()(float x) const {
        return 0.5f * x * (1.0f + erff(x * 0.70710678118f));
    }
    __device__ float4 operator()(float4 a) const {
        auto g = [](float x) {
            return 0.5f * x * (1.0f + erff(x * 0.70710678118f));
        };
        return make_float4(g(a.x), g(a.y), g(a.z), g(a.w));
    }
};

void gelu_backward_hip_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input) {
    launch_broadcast_kernel(grad, grad.strides(), input, input.strides(), grad_input, GeluBackward());
}

void gelu_erf_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, GeluErf());
}

struct GeluErfBackward {
    __device__ float operator()(float g, float x) const {
        const float kAlpha = 0.70710678118654752440f; // 1/sqrt(2)
        const float kBeta = 0.39894228040143267794f;  // 1/sqrt(2*pi)
        float cdf = 0.5f * (1.0f + erff(x * kAlpha));
        float pdf = kBeta * expf(-0.5f * x * x);
        return g * (cdf + x * pdf);
    }

    __device__ float4 operator()(float4 g, float4 x) const {
        auto op = [&](float grad, float val) {
            const float kAlpha = 0.70710678118654752440f;
            const float kBeta = 0.39894228040143267794f;
            float cdf = 0.5f * (1.0f + erff(val * kAlpha));
            float pdf = kBeta * expf(-0.5f * val * val);
            return grad * (cdf + val * pdf);
        };
        return make_float4(op(g.x, x.x), op(g.y, x.y), op(g.z, x.z), op(g.w, x.w));
    }

    __device__ float4 operator()(float4 g, float x) const {
        auto op = [&](float grad, float val) {
            const float kAlpha = 0.70710678118654752440f;
            const float kBeta = 0.39894228040143267794f;
            float cdf = 0.5f * (1.0f + erff(val * kAlpha));
            float pdf = kBeta * expf(-0.5f * val * val);
            return grad * (cdf + val * pdf);
        };
        return make_float4(op(g.x, x), op(g.y, x), op(g.z, x), op(g.w, x));
    }
};

void gelu_erf_backward_hip_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input) {
    launch_broadcast_kernel(grad, grad.strides(), input, input.strides(), grad_input, GeluErfBackward());
}

// SiLU backward: d/dx[silu(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
struct SiLUBackward {
    __device__ float operator()(float g, float x) const {
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float d_silu = sigmoid_x * (1.0f + x * (1.0f - sigmoid_x));
        return g * d_silu;
    }

    __device__ float4 operator()(float4 g, float4 x) const {
        auto op = [](float grad, float val) {
            float sigmoid_x = 1.0f / (1.0f + expf(-val));
            float d_silu = sigmoid_x * (1.0f + val * (1.0f - sigmoid_x));
            return grad * d_silu;
        };
        return make_float4(op(g.x, x.x), op(g.y, x.y), op(g.z, x.z), op(g.w, x.w));
    }

    __device__ float4 operator()(float4 g, float x) const {
        auto op = [](float grad, float val) {
            float sigmoid_x = 1.0f / (1.0f + expf(-val));
            float d_silu = sigmoid_x * (1.0f + val * (1.0f - sigmoid_x));
            return grad * d_silu;
        };
        return make_float4(op(g.x, x), op(g.y, x), op(g.z, x), op(g.w, x));
    }
};

// ============================================================================
// Fused SiLU * Multiply: silu(gate) * up
// This is a key operation in SwiGLU FFN that benefits from fusion
// ============================================================================
struct SiLUMul {
    __device__ float operator()(float gate, float up) const {
        float sigmoid_g = 1.0f / (1.0f + expf(-gate));
        float silu_g = gate * sigmoid_g;
        return silu_g * up;
    }

    __device__ float4 operator()(float4 gate, float4 up) const {
        auto op = [](float g, float u) {
            float sigmoid_g = 1.0f / (1.0f + expf(-g));
            return (g * sigmoid_g) * u;
        };
        return make_float4(op(gate.x, up.x), op(gate.y, up.y), 
                           op(gate.z, up.z), op(gate.w, up.w));
    }
};

// Backward for silu_mul: d/dgate[silu(gate) * up] and d/dup[silu(gate) * up]
// d/dup = silu(gate)
// d/dgate = up * d_silu(gate) where d_silu(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
struct SiLUMulBackward {
    __device__ void operator()(float grad_out, float gate, float up, 
                                float& grad_gate, float& grad_up) const {
        float sigmoid_g = 1.0f / (1.0f + expf(-gate));
        float silu_g = gate * sigmoid_g;
        float d_silu = sigmoid_g * (1.0f + gate * (1.0f - sigmoid_g));
        
        grad_up = grad_out * silu_g;
        grad_gate = grad_out * up * d_silu;
    }
};

// Fused SiLU*Mul kernel
__global__ void silu_mul_kernel(const float* gate, const float* up, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float u = up[idx];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        out[idx] = (g * sigmoid_g) * u;
    }
}

// Vectorized fused SiLU*Mul
__global__ void silu_mul_kernel_vec4(const float4* gate, const float4* up, float4* out, size_t n_vec) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        float4 g = gate[idx];
        float4 u = up[idx];
        auto op = [](float gate_v, float up_v) {
            float sigmoid_g = 1.0f / (1.0f + expf(-gate_v));
            return (gate_v * sigmoid_g) * up_v;
        };
        out[idx] = make_float4(op(g.x, u.x), op(g.y, u.y), op(g.z, u.z), op(g.w, u.w));
    }
}

// Backward kernel for silu_mul
__global__ void silu_mul_backward_kernel(
    const float* grad_out, const float* gate, const float* up,
    float* grad_gate, float* grad_up, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float go = grad_out[idx];
        float g = gate[idx];
        float u = up[idx];
        
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        float silu_g = g * sigmoid_g;
        float d_silu = sigmoid_g * (1.0f + g * (1.0f - sigmoid_g));
        
        grad_up[idx] = go * silu_g;
        grad_gate[idx] = go * u * d_silu;
    }
}

void silu_mul_hip_dispatch(const Tensor& gate, const Tensor& up, Tensor& out) {
    size_t n = gate.numel();
    const int threads = 256;
    
    hipStream_t stream = static_cast<hipStream_t>(
        vesper::Stream::current(vesper::Device::HIP).raw_handle());
    
    if (n >= 4 && gate.is_contiguous() && up.is_contiguous() && (n % 4 == 0)) {
        // Vectorized path
        size_t n_vec = n / 4;
        int blocks = (n_vec + threads - 1) / threads;
        hipLaunchKernelGGL(silu_mul_kernel_vec4, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const float4*>(gate.data_ptr<float>()),
            reinterpret_cast<const float4*>(up.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()), n_vec);
    } else {
        // Scalar path
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(silu_mul_kernel, dim3(blocks), dim3(threads), 0, stream,
            gate.data_ptr<float>(), up.data_ptr<float>(), out.data_ptr<float>(), n);
    }
}

void silu_mul_backward_hip_dispatch(const Tensor& grad_out, const Tensor& gate, const Tensor& up,
                                     Tensor& grad_gate, Tensor& grad_up) {
    size_t n = gate.numel();
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    hipStream_t stream = static_cast<hipStream_t>(
        vesper::Stream::current(vesper::Device::HIP).raw_handle());
    
    hipLaunchKernelGGL(silu_mul_backward_kernel, dim3(blocks), dim3(threads), 0, stream,
        grad_out.data_ptr<float>(), gate.data_ptr<float>(), up.data_ptr<float>(),
        grad_gate.data_ptr<float>(), grad_up.data_ptr<float>(), n);
}

void silu_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, SiLU());
}

void silu_backward_hip_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input) {
    launch_broadcast_kernel(grad, grad.strides(), input, input.strides(), grad_input, SiLUBackward());
}

// In-place SiLU kernel
__global__ void silu_inplace_kernel(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        data[idx] = x * sigmoid_x;
    }
}

// Vectorized in-place SiLU
__global__ void silu_inplace_kernel_vec4(float4* data, size_t n_vec) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        float4 v = data[idx];
        auto silu_scalar = [](float x) {
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            return x * sigmoid_x;
        };
        data[idx] = make_float4(silu_scalar(v.x), silu_scalar(v.y), 
                                 silu_scalar(v.z), silu_scalar(v.w));
    }
}

void silu_inplace_hip_dispatch(Tensor& a) {
    size_t n = a.numel();
    const int threads = 256;
    
    hipStream_t stream = static_cast<hipStream_t>(
        vesper::Stream::current(vesper::Device::HIP).raw_handle());
    
    if (n >= 4 && a.is_contiguous() && (n % 4 == 0)) {
        // Vectorized path
        size_t n_vec = n / 4;
        int blocks = (n_vec + threads - 1) / threads;
        hipLaunchKernelGGL(silu_inplace_kernel_vec4, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<float4*>(a.data_ptr<float>()), n_vec);
    } else {
        // Scalar path
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(silu_inplace_kernel, dim3(blocks), dim3(threads), 0, stream,
            a.data_ptr<float>(), n);
    }
}

void add_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, Add());
}
void sub_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, Sub());
}
void mul_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, Mul());
}
void div_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    launch_broadcast_kernel(a, strides_a, b, strides_b, out, Div());
}

void add_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel(a, b, out, Add());
}
void sub_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel(a, b, out, Sub());
}
void mul_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel(a, b, out, Mul());
}
void div_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out) {
    launch_scalar_kernel(a, b, out, Div());
}

void sqrt_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Sqrt());
}
void sign_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Sign());
}
void gelu_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Gelu());
}
void exp_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Exp());
}
void log_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Log());
}
void cos_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Cos());
}
void sin_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    launch_unary_kernel(a, strides_a, out, Sin());
}

// =============================================================================
// Fused Operations for Optimizer Efficiency
// =============================================================================

// lerp kernel: self = self * (1 - weight) + other * weight
__global__ void lerp_kernel(float* __restrict__ self, 
                            const float* __restrict__ other,
                            float weight, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float one_minus_weight = 1.0f - weight;
        self[idx] = self[idx] * one_minus_weight + other[idx] * weight;
    }
}

__global__ void lerp_kernel_vec4(float4* __restrict__ self,
                                  const float4* __restrict__ other,
                                  float weight, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float one_minus_weight = 1.0f - weight;
        float4 s = self[idx];
        float4 o = other[idx];
        self[idx] = make_float4(
            s.x * one_minus_weight + o.x * weight,
            s.y * one_minus_weight + o.y * weight,
            s.z * one_minus_weight + o.z * weight,
            s.w * one_minus_weight + o.w * weight
        );
    }
}

void lerp_hip_dispatch(Tensor& self, const Tensor& other, float weight) {
    size_t n = self.numel();
    const int threads = 256;
    
    hipStream_t stream = static_cast<hipStream_t>(
        vesper::Stream::current(vesper::Device::HIP).raw_handle());
    
    if (n >= 4 && self.is_contiguous() && other.is_contiguous() && (n % 4 == 0)) {
        size_t n_vec = n / 4;
        int blocks = (n_vec + threads - 1) / threads;
        hipLaunchKernelGGL(lerp_kernel_vec4, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<float4*>(self.data_ptr<float>()),
            reinterpret_cast<const float4*>(other.data_ptr<const float>()),
            weight, n_vec);
    } else {
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(lerp_kernel, dim3(blocks), dim3(threads), 0, stream,
            self.data_ptr<float>(),
            other.data_ptr<const float>(),
            weight, n);
    }
}

// addcmul kernel: self = self + t1 * t2 * value
__global__ void addcmul_kernel(float* __restrict__ self,
                                const float* __restrict__ t1,
                                const float* __restrict__ t2,
                                float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        self[idx] += t1[idx] * t2[idx] * value;
    }
}

__global__ void addcmul_kernel_vec4(float4* __restrict__ self,
                                     const float4* __restrict__ t1,
                                     const float4* __restrict__ t2,
                                     float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 s = self[idx];
        float4 a = t1[idx];
        float4 b = t2[idx];
        self[idx] = make_float4(
            s.x + a.x * b.x * value,
            s.y + a.y * b.y * value,
            s.z + a.z * b.z * value,
            s.w + a.w * b.w * value
        );
    }
}

void addcmul_hip_dispatch(Tensor& self, const Tensor& t1, const Tensor& t2, float value) {
    size_t n = self.numel();
    const int threads = 256;
    
    hipStream_t stream = static_cast<hipStream_t>(
        vesper::Stream::current(vesper::Device::HIP).raw_handle());
    
    if (n >= 4 && self.is_contiguous() && t1.is_contiguous() && t2.is_contiguous() && (n % 4 == 0)) {
        size_t n_vec = n / 4;
        int blocks = (n_vec + threads - 1) / threads;
        hipLaunchKernelGGL(addcmul_kernel_vec4, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<float4*>(self.data_ptr<float>()),
            reinterpret_cast<const float4*>(t1.data_ptr<const float>()),
            reinterpret_cast<const float4*>(t2.data_ptr<const float>()),
            value, n_vec);
    } else {
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(addcmul_kernel, dim3(blocks), dim3(threads), 0, stream,
            self.data_ptr<float>(),
            t1.data_ptr<const float>(),
            t2.data_ptr<const float>(),
            value, n);
    }
}

// adam_update kernel: param = param - lr * (m / c1) / (sqrt(v / c2) + eps)
// Fused kernel to avoid 6 temporary tensor allocations per parameter
__global__ void adam_update_kernel(float* __restrict__ param,
                                    const float* __restrict__ m,
                                    const float* __restrict__ v,
                                    float lr, float c1, float c2, float eps, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float m_hat = m[idx] / c1;
        float v_hat = v[idx] / c2;
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

__global__ void adam_update_kernel_vec4(float4* __restrict__ param,
                                         const float4* __restrict__ m,
                                         const float4* __restrict__ v,
                                         float lr, float c1, float c2, float eps, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 p = param[idx];
        float4 m_val = m[idx];
        float4 v_val = v[idx];
        
        float4 m_hat = make_float4(m_val.x / c1, m_val.y / c1, m_val.z / c1, m_val.w / c1);
        float4 v_hat = make_float4(v_val.x / c2, v_val.y / c2, v_val.z / c2, v_val.w / c2);
        
        param[idx] = make_float4(
            p.x - lr * m_hat.x / (sqrtf(v_hat.x) + eps),
            p.y - lr * m_hat.y / (sqrtf(v_hat.y) + eps),
            p.z - lr * m_hat.z / (sqrtf(v_hat.z) + eps),
            p.w - lr * m_hat.w / (sqrtf(v_hat.w) + eps)
        );
    }
}

void adam_update_hip_dispatch(Tensor& param, const Tensor& m, const Tensor& v,
                               float lr, float c1, float c2, float eps) {
    size_t n = param.numel();
    const int threads = 256;
    
    hipStream_t stream = static_cast<hipStream_t>(
        vesper::Stream::current(vesper::Device::HIP).raw_handle());
    
    if (n >= 4 && param.is_contiguous() && m.is_contiguous() && v.is_contiguous() && (n % 4 == 0)) {
        size_t n_vec = n / 4;
        int blocks = (n_vec + threads - 1) / threads;
        hipLaunchKernelGGL(adam_update_kernel_vec4, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<const float4*>(m.data_ptr<const float>()),
            reinterpret_cast<const float4*>(v.data_ptr<const float>()),
            lr, c1, c2, eps, n_vec);
    } else {
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(adam_update_kernel, dim3(blocks), dim3(threads), 0, stream,
            param.data_ptr<float>(),
            m.data_ptr<const float>(),
            v.data_ptr<const float>(),
            lr, c1, c2, eps, n);
    }
}

} // namespace vesper::ops
