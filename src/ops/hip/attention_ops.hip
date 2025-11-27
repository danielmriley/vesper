/**
 * @file attention_ops.hip
 * @brief HIP kernels for attention-related operations
 * 
 * Chapter 33.5: Grouped Query Attention (GQA)
 */

#include <hip/hip_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/core/stream.h>
#include <vesper/ops/attention_ops.h>

namespace vesper::ops {

// =============================================================================
// repeat_kv kernel: Expand KV heads to match query heads
// =============================================================================

__global__ void repeat_kv_kernel(
    const float* __restrict__ input,   // [B, KV_H, S, D]
    float* __restrict__ output,         // [B, Q_H, S, D]
    int64_t batch, int64_t kv_heads, int64_t seq_len, int64_t head_dim,
    int64_t n_rep)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t q_heads = kv_heads * n_rep;
    int64_t total = batch * q_heads * seq_len * head_dim;
    
    if (idx >= total) return;
    
    // Decode output index: [b, q_h, s, d]
    int64_t d = idx % head_dim;
    int64_t s = (idx / head_dim) % seq_len;
    int64_t q_h = (idx / (head_dim * seq_len)) % q_heads;
    int64_t b = idx / (head_dim * seq_len * q_heads);
    
    // Map query head to KV head
    int64_t kv_h = q_h / n_rep;
    
    // Read from input
    int64_t in_idx = ((b * kv_heads + kv_h) * seq_len + s) * head_dim + d;
    output[idx] = input[in_idx];
}

// Vectorized version for better memory throughput
__global__ void repeat_kv_kernel_vec4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int64_t batch, int64_t kv_heads, int64_t seq_len, int64_t head_dim_div4,
    int64_t n_rep)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t q_heads = kv_heads * n_rep;
    int64_t total = batch * q_heads * seq_len * head_dim_div4;
    
    if (idx >= total) return;
    
    int64_t d4 = idx % head_dim_div4;
    int64_t s = (idx / head_dim_div4) % seq_len;
    int64_t q_h = (idx / (head_dim_div4 * seq_len)) % q_heads;
    int64_t b = idx / (head_dim_div4 * seq_len * q_heads);
    
    int64_t kv_h = q_h / n_rep;
    
    int64_t in_idx = ((b * kv_heads + kv_h) * seq_len + s) * head_dim_div4 + d4;
    output[idx] = input[in_idx];
}

void repeat_kv_hip_dispatch(const Tensor& input, Tensor& output, int64_t n_rep) {
    int64_t B = input.shape()[0];
    int64_t KV_H = input.shape()[1];
    int64_t S = input.shape()[2];
    int64_t D = input.shape()[3];
    int64_t Q_H = KV_H * n_rep;
    int64_t total = B * Q_H * S * D;
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(input.device()).raw_handle());
    
    int threads = 256;
    
    // Use vectorized kernel if head_dim is divisible by 4
    if (D % 4 == 0) {
        int64_t total_vec = total / 4;
        int blocks = (total_vec + threads - 1) / threads;
        
        hipLaunchKernelGGL(repeat_kv_kernel_vec4,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            B, KV_H, S, D / 4, n_rep);
    } else {
        int blocks = (total + threads - 1) / threads;
        
        hipLaunchKernelGGL(repeat_kv_kernel,
            dim3(blocks), dim3(threads), 0, stream,
            input.data_ptr<float>(), output.data_ptr<float>(),
            B, KV_H, S, D, n_rep);
    }
}

// =============================================================================
// repeat_kv_backward kernel: Sum gradients over repeated dimension
// =============================================================================

__global__ void repeat_kv_backward_kernel(
    const float* __restrict__ grad_out,  // [B, Q_H, S, D]
    float* __restrict__ grad_in,          // [B, KV_H, S, D]
    int64_t batch, int64_t kv_heads, int64_t q_heads, int64_t seq_len, int64_t head_dim,
    int64_t n_rep)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch * kv_heads * seq_len * head_dim;
    
    if (idx >= total) return;
    
    // Decode input (grad_in) index: [b, kv_h, s, d]
    int64_t d = idx % head_dim;
    int64_t s = (idx / head_dim) % seq_len;
    int64_t kv_h = (idx / (head_dim * seq_len)) % kv_heads;
    int64_t b = idx / (head_dim * seq_len * kv_heads);
    
    // Sum gradients from all Q heads that share this KV head
    float sum = 0.0f;
    for (int64_t r = 0; r < n_rep; ++r) {
        int64_t q_h = kv_h * n_rep + r;
        int64_t out_idx = ((b * q_heads + q_h) * seq_len + s) * head_dim + d;
        sum += grad_out[out_idx];
    }
    
    grad_in[idx] = sum;
}

// Vectorized backward kernel
__global__ void repeat_kv_backward_kernel_vec4(
    const float4* __restrict__ grad_out,
    float4* __restrict__ grad_in,
    int64_t batch, int64_t kv_heads, int64_t q_heads, int64_t seq_len, int64_t head_dim_div4,
    int64_t n_rep)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch * kv_heads * seq_len * head_dim_div4;
    
    if (idx >= total) return;
    
    int64_t d4 = idx % head_dim_div4;
    int64_t s = (idx / head_dim_div4) % seq_len;
    int64_t kv_h = (idx / (head_dim_div4 * seq_len)) % kv_heads;
    int64_t b = idx / (head_dim_div4 * seq_len * kv_heads);
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int64_t r = 0; r < n_rep; ++r) {
        int64_t q_h = kv_h * n_rep + r;
        int64_t out_idx = ((b * q_heads + q_h) * seq_len + s) * head_dim_div4 + d4;
        float4 val = grad_out[out_idx];
        sum.x += val.x;
        sum.y += val.y;
        sum.z += val.z;
        sum.w += val.w;
    }
    
    grad_in[idx] = sum;
}

void repeat_kv_backward_hip_dispatch(const Tensor& grad_output, Tensor& grad_input, int64_t n_rep) {
    int64_t B = grad_output.shape()[0];
    int64_t Q_H = grad_output.shape()[1];
    int64_t S = grad_output.shape()[2];
    int64_t D = grad_output.shape()[3];
    int64_t KV_H = Q_H / n_rep;
    int64_t total = B * KV_H * S * D;
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(grad_output.device()).raw_handle());
    
    int threads = 256;
    
    if (D % 4 == 0) {
        int64_t total_vec = total / 4;
        int blocks = (total_vec + threads - 1) / threads;
        
        hipLaunchKernelGGL(repeat_kv_backward_kernel_vec4,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const float4*>(grad_output.data_ptr<float>()),
            reinterpret_cast<float4*>(grad_input.data_ptr<float>()),
            B, KV_H, Q_H, S, D / 4, n_rep);
    } else {
        int blocks = (total + threads - 1) / threads;
        
        hipLaunchKernelGGL(repeat_kv_backward_kernel,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data_ptr<float>(), grad_input.data_ptr<float>(),
            B, KV_H, Q_H, S, D, n_rep);
    }
}

} // namespace vesper::ops
