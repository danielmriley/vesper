#include <vesper/core/tensor.h>
#include <vesper/core/stream.h>
#include <vesper/core/half.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Check for BF16 hardware support
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#include <hip/hip_bfloat16.h>
#define HIP_BF16_SUPPORTED 1
#else
#define HIP_BF16_SUPPORTED 0
#endif

namespace vesper::ops {

// Device-side FP16 conversion helpers using HIP intrinsics
__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half(f);
}

__device__ __forceinline__ float half_to_float(__half h) {
    return __half2float(h);
}

// Device-side BF16 conversion helpers
// BF16 is simply truncated FP32 (upper 16 bits)
__device__ __forceinline__ uint16_t float_to_bf16_bits(float f) {
    uint32_t bits = __float_as_uint(f);
    // Round to nearest even
    uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    return __uint_as_float(bits);
}

// Specific cast kernels to avoid template parsing issues with hipLaunchKernelGGL
__global__ void cast_float_to_float_kernel(const float* input, float* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx];
    }
}

__global__ void cast_float_to_int32_kernel(const float* input, int32_t* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = static_cast<int32_t>(input[idx]);
    }
}

__global__ void cast_float_to_int64_kernel(const float* input, int64_t* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = static_cast<int64_t>(input[idx]);
    }
}

// Specialized kernels for FP32 <-> FP16 conversion
__global__ void cast_float_to_half_kernel(const float* input, uint16_t* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        __half h = float_to_half(input[idx]);
        output[idx] = *reinterpret_cast<uint16_t*>(&h);
    }
}

__global__ void cast_half_to_float_kernel(const uint16_t* input, float* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        __half h = *reinterpret_cast<const __half*>(&input[idx]);
        output[idx] = half_to_float(h);
    }
}

// Specialized kernels for FP32 <-> BF16 conversion
__global__ void cast_float_to_bf16_kernel(const float* input, uint16_t* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = float_to_bf16_bits(input[idx]);
    }
}

__global__ void cast_bf16_to_float_kernel(const uint16_t* input, float* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = bf16_bits_to_float(input[idx]);
    }
}

// FP16 <-> BF16 conversion (via FP32)
__global__ void cast_half_to_bf16_kernel(const uint16_t* input, uint16_t* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        __half h = *reinterpret_cast<const __half*>(&input[idx]);
        float f = half_to_float(h);
        output[idx] = float_to_bf16_bits(f);
    }
}

__global__ void cast_bf16_to_half_kernel(const uint16_t* input, uint16_t* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float f = bf16_bits_to_float(input[idx]);
        __half h = float_to_half(f);
        output[idx] = *reinterpret_cast<uint16_t*>(&h);
    }
}

// Int32 cast kernels
__global__ void cast_int32_to_float_kernel(const int32_t* input, float* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = static_cast<float>(input[idx]);
    }
}

__global__ void cast_int32_to_int32_kernel(const int32_t* input, int32_t* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx];
    }
}

__global__ void cast_int32_to_int64_kernel(const int32_t* input, int64_t* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = static_cast<int64_t>(input[idx]);
    }
}

// Int64 cast kernels
__global__ void cast_int64_to_float_kernel(const int64_t* input, float* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = static_cast<float>(input[idx]);
    }
}

__global__ void cast_int64_to_int32_kernel(const int64_t* input, int32_t* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = static_cast<int32_t>(input[idx]);
    }
}

__global__ void cast_int64_to_int64_kernel(const int64_t* input, int64_t* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx];
    }
}

// Dispatch function for FP32 source
void dispatch_cast_from_float32(const Tensor& input, Tensor& output, size_t numel, hipStream_t stream) {
    const float* src_ptr = input.data_ptr<float>();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    switch (output.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(cast_float_to_float_kernel, dim3(blocks), dim3(threads), 0, stream, 
                               src_ptr, output.data_ptr<float>(), numel);
            break;
        case DType::Float16:
            hipLaunchKernelGGL(cast_float_to_half_kernel, dim3(blocks), dim3(threads), 0, stream,
                               src_ptr, output.data_ptr<uint16_t>(), numel);
            break;
        case DType::BFloat16:
            hipLaunchKernelGGL(cast_float_to_bf16_kernel, dim3(blocks), dim3(threads), 0, stream,
                               src_ptr, output.data_ptr<uint16_t>(), numel);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(cast_float_to_int32_kernel, dim3(blocks), dim3(threads), 0, stream, 
                               src_ptr, output.data_ptr<int32_t>(), numel);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(cast_float_to_int64_kernel, dim3(blocks), dim3(threads), 0, stream,
                               src_ptr, output.data_ptr<int64_t>(), numel);
            break;
        default:
            throw std::runtime_error("Unsupported destination dtype for cast from Float32 (HIP)");
    }
}

// Dispatch function for FP16 source
void dispatch_cast_from_float16(const Tensor& input, Tensor& output, size_t numel, hipStream_t stream) {
    const uint16_t* src_ptr = input.data_ptr<uint16_t>();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    switch (output.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(cast_half_to_float_kernel, dim3(blocks), dim3(threads), 0, stream,
                               src_ptr, output.data_ptr<float>(), numel);
            break;
        case DType::Float16:
            // Same type, just copy
            (void)hipMemcpyAsync(output.data_ptr<uint16_t>(), src_ptr, numel * sizeof(uint16_t), 
                          hipMemcpyDeviceToDevice, stream);
            break;
        case DType::BFloat16:
            hipLaunchKernelGGL(cast_half_to_bf16_kernel, dim3(blocks), dim3(threads), 0, stream,
                               src_ptr, output.data_ptr<uint16_t>(), numel);
            break;
        default:
            throw std::runtime_error("Unsupported destination dtype for cast from Float16 (HIP)");
    }
}

// Dispatch function for BF16 source
void dispatch_cast_from_bfloat16(const Tensor& input, Tensor& output, size_t numel, hipStream_t stream) {
    const uint16_t* src_ptr = input.data_ptr<uint16_t>();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    switch (output.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(cast_bf16_to_float_kernel, dim3(blocks), dim3(threads), 0, stream,
                               src_ptr, output.data_ptr<float>(), numel);
            break;
        case DType::Float16:
            hipLaunchKernelGGL(cast_bf16_to_half_kernel, dim3(blocks), dim3(threads), 0, stream,
                               src_ptr, output.data_ptr<uint16_t>(), numel);
            break;
        case DType::BFloat16:
            // Same type, just copy
            (void)hipMemcpyAsync(output.data_ptr<uint16_t>(), src_ptr, numel * sizeof(uint16_t),
                          hipMemcpyDeviceToDevice, stream);
            break;
        default:
            throw std::runtime_error("Unsupported destination dtype for cast from BFloat16 (HIP)");
    }
}

// Dispatch function for Int32 source
void dispatch_cast_from_int32(const Tensor& input, Tensor& output, size_t numel, hipStream_t stream) {
    const int32_t* src_ptr = input.data_ptr<int32_t>();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    switch (output.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(cast_int32_to_float_kernel, dim3(blocks), dim3(threads), 0, stream, 
                               src_ptr, output.data_ptr<float>(), numel);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(cast_int32_to_int32_kernel, dim3(blocks), dim3(threads), 0, stream, 
                               src_ptr, output.data_ptr<int32_t>(), numel);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(cast_int32_to_int64_kernel, dim3(blocks), dim3(threads), 0, stream,
                               src_ptr, output.data_ptr<int64_t>(), numel);
            break;
        default:
            throw std::runtime_error("Unsupported destination dtype for cast from Int32 (HIP)");
    }
}

// Dispatch function for Int64 source
void dispatch_cast_from_int64(const Tensor& input, Tensor& output, size_t numel, hipStream_t stream) {
    const int64_t* src_ptr = input.data_ptr<int64_t>();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    switch (output.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(cast_int64_to_float_kernel, dim3(blocks), dim3(threads), 0, stream,
                               src_ptr, output.data_ptr<float>(), numel);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(cast_int64_to_int32_kernel, dim3(blocks), dim3(threads), 0, stream,
                               src_ptr, output.data_ptr<int32_t>(), numel);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(cast_int64_to_int64_kernel, dim3(blocks), dim3(threads), 0, stream,
                               src_ptr, output.data_ptr<int64_t>(), numel);
            break;
        default:
            throw std::runtime_error("Unsupported destination dtype for cast from Int64 (HIP)");
    }
}

void cast_hip_dispatch(const Tensor& input, Tensor& output) {
    size_t numel = input.numel();
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    switch (input.dtype()) {
        case DType::Float32:
            dispatch_cast_from_float32(input, output, numel, stream);
            break;
        case DType::Float16:
            dispatch_cast_from_float16(input, output, numel, stream);
            break;
        case DType::BFloat16:
            dispatch_cast_from_bfloat16(input, output, numel, stream);
            break;
        case DType::Int32:
            dispatch_cast_from_int32(input, output, numel, stream);
            break;
        case DType::Int64:
            dispatch_cast_from_int64(input, output, numel, stream);
            break;
        default:
            throw std::runtime_error("Unsupported source dtype for cast (HIP)");
    }
}

} // namespace vesper::ops
