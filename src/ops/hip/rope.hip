#include <hip/hip_runtime.h>
#include <vesper/core/stream.h>

namespace vesper::nn {

// Basic RoPE kernel - each thread handles one dimension pair for one position
__global__ void rope_kernel(
    float* __restrict__ x,
    const float* __restrict__ freqs_cos,
    const float* __restrict__ freqs_sin,
    int batch, int heads, int seq_len, int head_dim)
{
    // Grid dimensions:
    // blockIdx.x = batch * heads index
    // blockIdx.y = position in sequence
    // threadIdx.x = dimension pair index (0 to head_dim/2 - 1)
    
    int batch_head = blockIdx.x;
    int pos = blockIdx.y;
    int pair_idx = threadIdx.x;
    
    int half_dim = head_dim / 2;
    
    if (pair_idx >= half_dim || pos >= seq_len) return;
    
    int b = batch_head / heads;
    int h = batch_head % heads;
    
    if (b >= batch) return;
    
    // Index into x: [b, h, pos, pair_idx*2] and [b, h, pos, pair_idx*2+1]
    int x_base = ((b * heads + h) * seq_len + pos) * head_dim;
    int idx0 = x_base + pair_idx * 2;
    int idx1 = x_base + pair_idx * 2 + 1;
    
    // Index into freqs: [pos, pair_idx]
    int freq_idx = pos * half_dim + pair_idx;
    
    float x0 = x[idx0];
    float x1 = x[idx1];
    float cos_val = freqs_cos[freq_idx];
    float sin_val = freqs_sin[freq_idx];
    
    // Apply rotation: [cos -sin; sin cos] * [x0; x1]
    x[idx0] = x0 * cos_val - x1 * sin_val;
    x[idx1] = x0 * sin_val + x1 * cos_val;
}

// Inverse RoPE kernel (for backward pass)
__global__ void rope_inverse_kernel(
    float* __restrict__ x,
    const float* __restrict__ freqs_cos,
    const float* __restrict__ freqs_sin,
    int batch, int heads, int seq_len, int head_dim)
{
    int batch_head = blockIdx.x;
    int pos = blockIdx.y;
    int pair_idx = threadIdx.x;
    
    int half_dim = head_dim / 2;
    
    if (pair_idx >= half_dim || pos >= seq_len) return;
    
    int b = batch_head / heads;
    int h = batch_head % heads;
    
    if (b >= batch) return;
    
    int x_base = ((b * heads + h) * seq_len + pos) * head_dim;
    int idx0 = x_base + pair_idx * 2;
    int idx1 = x_base + pair_idx * 2 + 1;
    
    int freq_idx = pos * half_dim + pair_idx;
    
    float x0 = x[idx0];
    float x1 = x[idx1];
    float cos_val = freqs_cos[freq_idx];
    float sin_val = freqs_sin[freq_idx];
    
    // Inverse rotation: [cos sin; -sin cos] * [x0; x1]
    x[idx0] = x0 * cos_val + x1 * sin_val;
    x[idx1] = -x0 * sin_val + x1 * cos_val;
}

// Vectorized RoPE kernel using float4 for better memory bandwidth
// Each thread handles 2 dimension pairs (4 floats)
__global__ void rope_vectorized_kernel(
    float* __restrict__ x,
    const float* __restrict__ freqs_cos,
    const float* __restrict__ freqs_sin,
    int batch, int heads, int seq_len, int head_dim)
{
    int batch_head = blockIdx.x;
    int pos = blockIdx.y;
    int vec_idx = threadIdx.x;  // Each thread handles 2 pairs (4 floats)
    
    int half_dim = head_dim / 2;
    int num_vec = half_dim / 2;  // Number of float4 vectors per position
    
    if (vec_idx >= num_vec || pos >= seq_len) return;
    
    int b = batch_head / heads;
    int h = batch_head % heads;
    
    if (b >= batch) return;
    
    // Base index for this position
    int x_base = ((b * heads + h) * seq_len + pos) * head_dim;
    int pair_start = vec_idx * 2;  // Starting pair index
    
    // Load 4 values (2 dimension pairs)
    float4* x_vec = reinterpret_cast<float4*>(&x[x_base + pair_start * 2]);
    float4 vals = *x_vec;
    
    // Load cos/sin for both pairs using __ldg for read-only data cache
    int freq_base = pos * half_dim + pair_start;
    float cos0 = __ldg(&freqs_cos[freq_base]);
    float sin0 = __ldg(&freqs_sin[freq_base]);
    float cos1 = __ldg(&freqs_cos[freq_base + 1]);
    float sin1 = __ldg(&freqs_sin[freq_base + 1]);
    
    // Apply rotation to both pairs
    float4 result;
    result.x = vals.x * cos0 - vals.y * sin0;  // pair 0, element 0
    result.y = vals.x * sin0 + vals.y * cos0;  // pair 0, element 1
    result.z = vals.z * cos1 - vals.w * sin1;  // pair 1, element 0
    result.w = vals.z * sin1 + vals.w * cos1;  // pair 1, element 1
    
    *x_vec = result;
}

// Vectorized inverse RoPE kernel
__global__ void rope_inverse_vectorized_kernel(
    float* __restrict__ x,
    const float* __restrict__ freqs_cos,
    const float* __restrict__ freqs_sin,
    int batch, int heads, int seq_len, int head_dim)
{
    int batch_head = blockIdx.x;
    int pos = blockIdx.y;
    int vec_idx = threadIdx.x;
    
    int half_dim = head_dim / 2;
    int num_vec = half_dim / 2;
    
    if (vec_idx >= num_vec || pos >= seq_len) return;
    
    int b = batch_head / heads;
    int h = batch_head % heads;
    
    if (b >= batch) return;
    
    int x_base = ((b * heads + h) * seq_len + pos) * head_dim;
    int pair_start = vec_idx * 2;
    
    float4* x_vec = reinterpret_cast<float4*>(&x[x_base + pair_start * 2]);
    float4 vals = *x_vec;
    
    // Use __ldg for read-only frequency data
    int freq_base = pos * half_dim + pair_start;
    float cos0 = __ldg(&freqs_cos[freq_base]);
    float sin0 = __ldg(&freqs_sin[freq_base]);
    float cos1 = __ldg(&freqs_cos[freq_base + 1]);
    float sin1 = __ldg(&freqs_sin[freq_base + 1]);
    
    // Inverse rotation
    float4 result;
    result.x = vals.x * cos0 + vals.y * sin0;
    result.y = -vals.x * sin0 + vals.y * cos0;
    result.z = vals.z * cos1 + vals.w * sin1;
    result.w = -vals.z * sin1 + vals.w * cos1;
    
    *x_vec = result;
}

// Host dispatch function for forward RoPE
void apply_rope_hip(float* x, const float* freqs_cos, const float* freqs_sin,
                    int batch, int heads, int seq_len, int head_dim) {
    hipStream_t stream = static_cast<hipStream_t>(
        vesper::Stream::current(vesper::Device::HIP).raw_handle());
    
    int half_dim = head_dim / 2;
    
    // Choose between basic and vectorized kernel
    // Vectorized requires head_dim divisible by 4
    if (head_dim >= 8 && head_dim % 4 == 0) {
        int num_vec = half_dim / 2;
        dim3 threads(num_vec);
        dim3 blocks(batch * heads, seq_len);
        
        hipLaunchKernelGGL(rope_vectorized_kernel, blocks, threads, 0, stream,
            x, freqs_cos, freqs_sin, batch, heads, seq_len, head_dim);
    } else {
        // Basic kernel: one thread per dimension pair
        dim3 threads(half_dim);
        dim3 blocks(batch * heads, seq_len);
        
        hipLaunchKernelGGL(rope_kernel, blocks, threads, 0, stream,
            x, freqs_cos, freqs_sin, batch, heads, seq_len, head_dim);
    }
    
    // Check for launch errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("RoPE HIP kernel launch failed: ") + 
                                hipGetErrorString(err));
    }
}

// Host dispatch function for inverse RoPE
void apply_rope_inverse_hip(float* x, const float* freqs_cos, const float* freqs_sin,
                            int batch, int heads, int seq_len, int head_dim) {
    hipStream_t stream = static_cast<hipStream_t>(
        vesper::Stream::current(vesper::Device::HIP).raw_handle());
    
    int half_dim = head_dim / 2;
    
    if (head_dim >= 8 && head_dim % 4 == 0) {
        int num_vec = half_dim / 2;
        dim3 threads(num_vec);
        dim3 blocks(batch * heads, seq_len);
        
        hipLaunchKernelGGL(rope_inverse_vectorized_kernel, blocks, threads, 0, stream,
            x, freqs_cos, freqs_sin, batch, heads, seq_len, head_dim);
    } else {
        dim3 threads(half_dim);
        dim3 blocks(batch * heads, seq_len);
        
        hipLaunchKernelGGL(rope_inverse_kernel, blocks, threads, 0, stream,
            x, freqs_cos, freqs_sin, batch, heads, seq_len, head_dim);
    }
    
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("RoPE inverse HIP kernel launch failed: ") + 
                                hipGetErrorString(err));
    }
}

} // namespace vesper::nn
