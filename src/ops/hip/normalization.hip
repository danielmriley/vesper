#include <hip/hip_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/ops/normalization.h>
#include <vesper/core/stream.h>
#include <stdexcept>
#include <cmath>

namespace vesper::ops {

// --- Softmax HIP ---

// Warp size for AMD GPUs (RDNA/CDNA use 32 or 64 depending on mode)
constexpr int WARP_SIZE_HIP = 64;  // AMD wavefront size

// Warp-level reduction using shuffle operations
__device__ __forceinline__ float warpReduceMax_hip(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE_HIP / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum_hip(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE_HIP / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Block-level reduction using warp reductions
__device__ float blockReduceMax_hip(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE_HIP;
    int wid = threadIdx.x / WARP_SIZE_HIP;
    
    val = warpReduceMax_hip(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // First warp reduces across warps
    int numWarps = (blockDim.x + WARP_SIZE_HIP - 1) / WARP_SIZE_HIP;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : -INFINITY;
    if (wid == 0) val = warpReduceMax_hip(val);
    
    return val;
}

__device__ float blockReduceSum_hip(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE_HIP;
    int wid = threadIdx.x / WARP_SIZE_HIP;
    
    val = warpReduceSum_hip(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    int numWarps = (blockDim.x + WARP_SIZE_HIP - 1) / WARP_SIZE_HIP;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum_hip(val);
    
    return val;
}

// Optimized softmax kernel using online algorithm (single pass for max+sum)
// This reduces memory bandwidth by 3x compared to naive 3-pass implementation
__global__ void softmax_online_kernel(const float* __restrict__ input, 
                                       float* __restrict__ output, 
                                       int64_t rows, int64_t cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    extern __shared__ float shared_mem[];
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    // Online softmax: track max and sum of exp(x - max) in single pass
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    
    // First pass: compute local max and sum using online algorithm
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = row_input[i];
        if (val > thread_max) {
            // Rescale existing sum when we find a new max
            thread_sum = thread_sum * expf(thread_max - val) + 1.0f;
            thread_max = val;
        } else {
            thread_sum += expf(val - thread_max);
        }
    }
    
    // Reduce max across block
    float block_max = blockReduceMax_hip(thread_max, shared_mem);
    __syncthreads();
    
    // Rescale thread_sum to use global max
    if (thread_max > -INFINITY) {
        thread_sum *= expf(thread_max - block_max);
    } else {
        thread_sum = 0.0f;
    }
    
    // Reduce sum across block
    float block_sum = blockReduceSum_hip(thread_sum, shared_mem);
    
    // Broadcast results
    __shared__ float s_max, s_sum;
    if (threadIdx.x == 0) {
        s_max = block_max;
        s_sum = block_sum;
    }
    __syncthreads();
    
    float inv_sum = 1.0f / s_sum;
    
    // Second pass: write normalized output
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_output[i] = expf(row_input[i] - s_max) * inv_sum;
    }
}

// Vectorized version for when cols is divisible by 4
__global__ void softmax_online_vec4_kernel(const float* __restrict__ input, 
                                            float* __restrict__ output, 
                                            int64_t rows, int64_t cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    extern __shared__ float shared_mem[];
    
    const float4* row_input = reinterpret_cast<const float4*>(input + row * cols);
    float4* row_output = reinterpret_cast<float4*>(output + row * cols);
    int64_t cols4 = cols / 4;
    
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    
    // Online algorithm with vectorized loads
    for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
        float4 v = row_input[i];
        
        // Process each element
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float val = (j == 0) ? v.x : (j == 1) ? v.y : (j == 2) ? v.z : v.w;
            if (val > thread_max) {
                thread_sum = thread_sum * expf(thread_max - val) + 1.0f;
                thread_max = val;
            } else {
                thread_sum += expf(val - thread_max);
            }
        }
    }
    
    float block_max = blockReduceMax_hip(thread_max, shared_mem);
    __syncthreads();
    
    if (thread_max > -INFINITY) {
        thread_sum *= expf(thread_max - block_max);
    } else {
        thread_sum = 0.0f;
    }
    
    float block_sum = blockReduceSum_hip(thread_sum, shared_mem);
    
    __shared__ float s_max, s_sum;
    if (threadIdx.x == 0) {
        s_max = block_max;
        s_sum = block_sum;
    }
    __syncthreads();
    
    float inv_sum = 1.0f / s_sum;
    
    for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
        float4 v = row_input[i];
        float4 out;
        out.x = expf(v.x - s_max) * inv_sum;
        out.y = expf(v.y - s_max) * inv_sum;
        out.z = expf(v.z - s_max) * inv_sum;
        out.w = expf(v.w - s_max) * inv_sum;
        row_output[i] = out;
    }
}

// Legacy kernel for backward compatibility
__global__ void softmax_last_dim_kernel(const float* input, float* output, int64_t rows, int64_t cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // 1. Find Max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = input[row * cols + i];
        if (val > max_val) max_val = val;
    }
    
    extern __shared__ float shared_mem[];
    float* sdata = shared_mem;
    
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    
    // 2. Sum Exps
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum_exp += expf(input[row * cols + i] - max_val);
    }
    
    sdata[threadIdx.x] = sum_exp;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_exp = sdata[0];
    
    // 3. Write Output
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        output[row * cols + i] = expf(input[row * cols + i] - max_val) / sum_exp;
    }
}

void softmax_hip_dispatch(const Tensor& input, int64_t dim, Tensor& output) {
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    
    if (dim != ndim - 1) {
        throw std::runtime_error("Softmax HIP only supports last dimension for now.");
    }
    
    // Handle non-contiguous input by making it contiguous first
    Tensor contiguous_input = input.is_contiguous() ? input : input.contiguous();
    
    int64_t cols = contiguous_input.shape()[ndim - 1];
    int64_t rows = contiguous_input.numel() / cols;
    
    int threads = 256;
    int blocks = rows;
    // Shared memory for warp-level reductions (max 8 warps with 256 threads)
    size_t shared_mem_size = (threads / WARP_SIZE_HIP + 1) * sizeof(float);
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    // Use vectorized kernel when possible (cols divisible by 4 and aligned)
    if (cols % 4 == 0 && cols >= 64) {
        hipLaunchKernelGGL(softmax_online_vec4_kernel, dim3(blocks), dim3(threads), shared_mem_size, stream,
            contiguous_input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    } else {
        hipLaunchKernelGGL(softmax_online_kernel, dim3(blocks), dim3(threads), shared_mem_size, stream,
            contiguous_input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    }
}

// --- Layer Norm HIP ---

__global__ void layernorm_kernel(const float* input, float* output, 
                                 const float* weight, const float* bias,
                                 int64_t rows, int64_t cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    extern __shared__ float sdata[]; // size: threads * 2 (for mean and var)
    float* s_mean = sdata;
    float* s_var = sdata + blockDim.x;

    // 1. Mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += input[row * cols + i];
    }
    s_mean[threadIdx.x] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_mean[threadIdx.x] += s_mean[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = s_mean[0] / cols;
    
    // 2. Variance
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = input[row * cols + i] - mean;
        sum_sq += diff * diff;
    }
    s_var[threadIdx.x] = sum_sq;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_var[threadIdx.x] += s_var[threadIdx.x + s];
        }
        __syncthreads();
    }
    float var = s_var[0] / cols;
    float inv_std = rsqrtf(var + eps);
    
    // 3. Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = (input[row * cols + i] - mean) * inv_std;
        float w = weight ? weight[i] : 1.0f;
        float b = bias ? bias[i] : 0.0f;
        output[row * cols + i] = val * w + b;
    }
}

void layer_norm_hip_dispatch(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                              const Tensor& weight, const Tensor& bias, float eps, Tensor& output) {
    int64_t norm_size = 1;
    for (auto s : normalized_shape) norm_size *= s;
    int64_t rows = input.numel() / norm_size;
    
    int threads = 256;
    int blocks = rows;
    size_t shared_mem_size = threads * 2 * sizeof(float);
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    const float* w_ptr = weight.defined() ? weight.data_ptr<float>() : nullptr;
    const float* b_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    hipLaunchKernelGGL(layernorm_kernel, dim3(blocks), dim3(threads), shared_mem_size, stream,
        input.data_ptr<float>(), output.data_ptr<float>(),
        w_ptr, b_ptr,
        rows, norm_size, eps
    );
}

// --- RMS Norm HIP ---

__global__ void rmsnorm_kernel(const float* input, float* output, 
                               const float* weight,
                               int64_t rows, int64_t cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    extern __shared__ float sdata[];
    
    // 1. Sum Squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = input[row * cols + i];
        sum_sq += val * val;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float rms = sqrtf(sdata[0] / cols + eps);
    float inv_rms = 1.0f / rms;
    
    // 2. Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        output[row * cols + i] = input[row * cols + i] * inv_rms * w;
    }
}

void rms_norm_hip_dispatch(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                            const Tensor& weight, float eps, Tensor& output) {
    int64_t norm_size = 1;
    for (auto s : normalized_shape) norm_size *= s;
    int64_t rows = input.numel() / norm_size;
    
    int threads = 256;
    int blocks = rows;
    size_t shared_mem_size = threads * sizeof(float);
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    const float* w_ptr = weight.defined() ? weight.data_ptr<float>() : nullptr;
    
    hipLaunchKernelGGL(rmsnorm_kernel, dim3(blocks), dim3(threads), shared_mem_size, stream,
        input.data_ptr<float>(), output.data_ptr<float>(),
        w_ptr,
        rows, norm_size, eps
    );
}

// --- LayerNorm Backward HIP ---

__global__ void layernorm_backward_kernel(
    const float* grad_output, const float* input, const float* weight,
    float* grad_input, float* grad_weight, float* grad_bias,
    int64_t rows, int64_t cols, float eps) 
{
    int row = blockIdx.x;
    if (row >= rows) return;
    
    extern __shared__ float sdata[];
    float* s_mean = sdata;
    float* s_var = sdata + blockDim.x;
    float* s_sum_grad_xhat = sdata + 2 * blockDim.x;
    float* s_sum_grad_xhat_xhat = sdata + 3 * blockDim.x;

    // 1. Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += input[row * cols + i];
    }
    s_mean[threadIdx.x] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_mean[threadIdx.x] += s_mean[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = s_mean[0] / cols;
    
    // 2. Compute variance
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = input[row * cols + i] - mean;
        sum_sq += diff * diff;
    }
    s_var[threadIdx.x] = sum_sq;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_var[threadIdx.x] += s_var[threadIdx.x + s];
        }
        __syncthreads();
    }
    float var = s_var[0] / cols;
    float inv_std = rsqrtf(var + eps);
    
    // 3. Compute sum_grad_xhat and sum_grad_xhat_xhat
    float local_sum_grad_xhat = 0.0f;
    float local_sum_grad_xhat_xhat = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        float grad_xhat = grad_output[row * cols + i] * w;
        float xhat = (input[row * cols + i] - mean) * inv_std;
        local_sum_grad_xhat += grad_xhat;
        local_sum_grad_xhat_xhat += grad_xhat * xhat;
    }
    s_sum_grad_xhat[threadIdx.x] = local_sum_grad_xhat;
    s_sum_grad_xhat_xhat[threadIdx.x] = local_sum_grad_xhat_xhat;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum_grad_xhat[threadIdx.x] += s_sum_grad_xhat[threadIdx.x + s];
            s_sum_grad_xhat_xhat[threadIdx.x] += s_sum_grad_xhat_xhat[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_grad_xhat = s_sum_grad_xhat[0];
    float sum_grad_xhat_xhat = s_sum_grad_xhat_xhat[0];
    
    // 4. Compute grad_input
    float N = static_cast<float>(cols);
    if (grad_input) {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float w = weight ? weight[i] : 1.0f;
            float grad_xhat = grad_output[row * cols + i] * w;
            float xhat = (input[row * cols + i] - mean) * inv_std;
            grad_input[row * cols + i] = (1.0f / N) * inv_std * 
                (N * grad_xhat - sum_grad_xhat - xhat * sum_grad_xhat_xhat);
        }
    }
    
    // 5. Accumulate grad_weight and grad_bias (using atomicAdd across rows)
    if (grad_weight) {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float xhat = (input[row * cols + i] - mean) * inv_std;
            atomicAdd(&grad_weight[i], grad_output[row * cols + i] * xhat);
        }
    }
    
    if (grad_bias) {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            atomicAdd(&grad_bias[i], grad_output[row * cols + i]);
        }
    }
}

void layer_norm_backward_hip_dispatch(const Tensor& grad_output, const Tensor& input, 
                                      const Tensor& weight, const std::vector<int64_t>& normalized_shape,
                                      float eps, Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias) {
    int64_t norm_size = 1;
    for (auto s : normalized_shape) norm_size *= s;
    int64_t rows = input.numel() / norm_size;
    
    int threads = 256;
    int blocks = rows;
    size_t shared_mem_size = threads * 4 * sizeof(float);  // mean, var, sum_grad_xhat, sum_grad_xhat_xhat
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    const float* w_ptr = weight.defined() ? weight.data_ptr<float>() : nullptr;
    float* grad_in_ptr = grad_input.defined() ? grad_input.data_ptr<float>() : nullptr;
    float* grad_w_ptr = grad_weight.defined() ? grad_weight.data_ptr<float>() : nullptr;
    float* grad_b_ptr = grad_bias.defined() ? grad_bias.data_ptr<float>() : nullptr;
    
    hipLaunchKernelGGL(layernorm_backward_kernel, dim3(blocks), dim3(threads), shared_mem_size, stream,
        grad_output.data_ptr<float>(), input.data_ptr<float>(), w_ptr,
        grad_in_ptr, grad_w_ptr, grad_b_ptr,
        rows, norm_size, eps
    );
}

// --- RMSNorm Backward HIP ---

__global__ void rmsnorm_backward_kernel(
    const float* grad_output, const float* input, const float* weight,
    float* grad_input, float* grad_weight,
    int64_t rows, int64_t cols, float eps) 
{
    int row = blockIdx.x;
    if (row >= rows) return;
    
    extern __shared__ float sdata[];
    float* s_sum_sq = sdata;
    float* s_sum_grad_wx = sdata + blockDim.x;

    // 1. Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = input[row * cols + i];
        sum_sq += val * val;
    }
    s_sum_sq[threadIdx.x] = sum_sq;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }
    float ms = s_sum_sq[0] / cols;
    float rms = sqrtf(ms + eps);
    float inv_rms = 1.0f / rms;
    float inv_rms_cubed = inv_rms * inv_rms * inv_rms;
    
    // 2. Compute sum(grad_y * weight * x)
    float local_sum_grad_wx = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        local_sum_grad_wx += grad_output[row * cols + i] * w * input[row * cols + i];
    }
    s_sum_grad_wx[threadIdx.x] = local_sum_grad_wx;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum_grad_wx[threadIdx.x] += s_sum_grad_wx[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_grad_wx = s_sum_grad_wx[0];
    
    // 3. Compute grad_input
    float N = static_cast<float>(cols);
    if (grad_input) {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float w = weight ? weight[i] : 1.0f;
            float grad_w = grad_output[row * cols + i] * w;
            grad_input[row * cols + i] = inv_rms * grad_w 
                - input[row * cols + i] * sum_grad_wx * inv_rms_cubed / N;
        }
    }
    
    // 4. Accumulate grad_weight (using atomicAdd across rows)
    if (grad_weight) {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float x_normalized = input[row * cols + i] * inv_rms;
            atomicAdd(&grad_weight[i], grad_output[row * cols + i] * x_normalized);
        }
    }
}

void rms_norm_backward_hip_dispatch(const Tensor& grad_output, const Tensor& input, 
                                    const Tensor& weight, const std::vector<int64_t>& normalized_shape,
                                    float eps, Tensor& grad_input, Tensor& grad_weight) {
    int64_t norm_size = 1;
    for (auto s : normalized_shape) norm_size *= s;
    int64_t rows = input.numel() / norm_size;
    
    int threads = 256;
    int blocks = rows;
    size_t shared_mem_size = threads * 2 * sizeof(float);  // sum_sq, sum_grad_wx
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    const float* w_ptr = weight.defined() ? weight.data_ptr<float>() : nullptr;
    float* grad_in_ptr = grad_input.defined() ? grad_input.data_ptr<float>() : nullptr;
    float* grad_w_ptr = grad_weight.defined() ? grad_weight.data_ptr<float>() : nullptr;
    
    hipLaunchKernelGGL(rmsnorm_backward_kernel, dim3(blocks), dim3(threads), shared_mem_size, stream,
        grad_output.data_ptr<float>(), input.data_ptr<float>(), w_ptr,
        grad_in_ptr, grad_w_ptr,
        rows, norm_size, eps
    );
}

} // namespace vesper::ops
