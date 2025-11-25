#include <hip/hip_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/ops/normalization.h>
#include <vesper/core/stream.h>
#include <stdexcept>
#include <cmath>

namespace vesper::ops {

// --- Softmax HIP ---

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
    if (!input.is_contiguous()) {
        throw std::runtime_error("Softmax HIP only supports contiguous tensors for now.");
    }
    
    int64_t cols = input.shape()[ndim - 1];
    int64_t rows = input.numel() / cols;
    
    int threads = 256;
    int blocks = rows;
    size_t shared_mem_size = threads * sizeof(float);
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    hipLaunchKernelGGL(softmax_last_dim_kernel, dim3(blocks), dim3(threads), shared_mem_size, stream,
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols
    );
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
