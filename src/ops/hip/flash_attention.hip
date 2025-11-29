#include <vesper/ops/flash_attention.h>
#include <vesper/core/tensor.h>
#include <vesper/core/stream.h>
#include <hip/hip_runtime.h>
#include <cfloat>
#include <cmath>
#include <chrono>

namespace vesper::ops {

// Tile sizes for Flash Attention - optimized for AMD consumer GPUs
constexpr int BLOCK_M = 32;   // Query tile size
constexpr int BLOCK_N = 32;   // Key/Value tile size  
constexpr int THREADS_PER_BLOCK = 128;

// Simple LCG random number generator for dropout on GPU
__device__ inline uint32_t lcg_random(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ inline float lcg_uniform(uint32_t& state) {
    return static_cast<float>(lcg_random(state)) / 4294967296.0f;
}

/**
 * Flash Attention Forward Kernel with Dropout Support
 * 
 * Processes one query position per iteration.
 * Uses online softmax to avoid materializing the N×N attention matrix.
 */
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ Q,  // [B, H, N, D]
    const float* __restrict__ K,  // [B, H, N, D]
    const float* __restrict__ V,  // [B, H, N, D]
    float* __restrict__ O,        // [B, H, N, D]
    float* __restrict__ L,        // [B, H, N] - log-sum-exp
    float* __restrict__ dropout_mask,  // [B, H, N, N] or nullptr
    int B, int H, int N, int D,
    float scale, bool is_causal,
    float dropout_p, bool use_dropout,
    uint64_t seed)
{
    const int batch_head = blockIdx.y;
    const int b = batch_head / H;
    const int h = batch_head % H;
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B || q_idx >= N) return;
    
    // Initialize RNG state for this thread
    uint32_t rng_state = static_cast<uint32_t>(seed + batch_head * N + q_idx);
    
    // Dropout scale (inverted dropout)
    float dropout_scale = use_dropout ? 1.0f / (1.0f - dropout_p) : 1.0f;
    
    // Base pointers for this (batch, head)
    const float* Q_bh = Q + ((int64_t)b * H + h) * N * D;
    const float* K_bh = K + ((int64_t)b * H + h) * N * D;
    const float* V_bh = V + ((int64_t)b * H + h) * N * D;
    float* O_bh = O + ((int64_t)b * H + h) * N * D;
    float* L_bh = L + ((int64_t)b * H + h) * N;
    float* mask_bh = dropout_mask ? dropout_mask + ((int64_t)b * H + h) * N * N : nullptr;
    
    // Accumulators - stored in local memory for larger D
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    
    // Determine K/V range
    int kv_end = is_causal ? (q_idx + 1) : N;
    
    // First pass: find max and compute sum
    for (int kv_idx = 0; kv_idx < kv_end; ++kv_idx) {
        // Compute Q[q_idx] · K[kv_idx]
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += Q_bh[q_idx * D + d] * K_bh[kv_idx * D + d];
        }
        score *= scale;
        
        // Update running max
        float m_new = fmaxf(m_prev, score);
        float exp_prev = expf(m_prev - m_new);
        float exp_score = expf(score - m_new);
        
        l_prev = l_prev * exp_prev + exp_score;
        m_prev = m_new;
    }
    
    // Second pass: compute weighted sum of V with dropout
    float inv_l = 1.0f / (l_prev + 1e-10f);
    
    // Reset RNG for dropout to be deterministic
    rng_state = static_cast<uint32_t>(seed + batch_head * N + q_idx);
    
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        
        // Reset RNG for each d dimension to maintain consistency
        uint32_t d_rng_state = rng_state;
        
        for (int kv_idx = 0; kv_idx < kv_end; ++kv_idx) {
            // Recompute attention score
            float score = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                score += Q_bh[q_idx * D + dd] * K_bh[kv_idx * D + dd];
            }
            score *= scale;
            
            // Compute attention weight
            float p = expf(score - m_prev) * inv_l;
            
            // Apply dropout (only compute mask once per d=0, then reuse)
            if (use_dropout) {
                // Generate dropout decision
                bool drop = lcg_uniform(d_rng_state) < dropout_p;
                
                if (d == 0 && mask_bh) {
                    // Save mask on first dimension pass (1.0 = kept, 0.0 = dropped)
                    mask_bh[q_idx * N + kv_idx] = drop ? 0.0f : 1.0f;
                }
                
                if (drop) {
                    p = 0.0f;
                } else {
                    p *= dropout_scale;
                }
            }
            
            acc += p * V_bh[kv_idx * D + d];
        }
        
        O_bh[q_idx * D + d] = acc;
    }
    
    // Store log-sum-exp for backward pass
    L_bh[q_idx] = m_prev + logf(l_prev + 1e-10f);
}

/**
 * Flash Attention Backward Kernel with Dropout Support
 */
__global__ void flash_attention_backward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ dropout_mask,  // Can be nullptr
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int B, int H, int N, int D,
    float scale, bool is_causal,
    float dropout_p, bool use_dropout)
{
    const int batch_head = blockIdx.y;
    const int b = batch_head / H;
    const int h = batch_head % H;
    const int kv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B || kv_idx >= N) return;
    
    float dropout_scale = use_dropout ? 1.0f / (1.0f - dropout_p) : 1.0f;
    
    // Base pointers
    const float* Q_bh = Q + ((int64_t)b * H + h) * N * D;
    const float* K_bh = K + ((int64_t)b * H + h) * N * D;
    const float* V_bh = V + ((int64_t)b * H + h) * N * D;
    const float* O_bh = O + ((int64_t)b * H + h) * N * D;
    const float* dO_bh = dO + ((int64_t)b * H + h) * N * D;
    const float* L_bh = L + ((int64_t)b * H + h) * N;
    const float* mask_bh = dropout_mask ? dropout_mask + ((int64_t)b * H + h) * N * N : nullptr;
    float* dQ_bh = dQ + ((int64_t)b * H + h) * N * D;
    float* dK_bh = dK + ((int64_t)b * H + h) * N * D;
    float* dV_bh = dV + ((int64_t)b * H + h) * N * D;
    
    // Determine query range that attends to this K/V
    int q_start = is_causal ? kv_idx : 0;
    
    // Accumulate gradients for this K/V position
    for (int d = 0; d < D; ++d) {
        float dk_sum = 0.0f;
        float dv_sum = 0.0f;
        
        for (int q_idx = q_start; q_idx < N; ++q_idx) {
            // Check dropout mask (0.0 = dropped, 1.0 = kept)
            bool dropped = use_dropout && mask_bh && (mask_bh[q_idx * N + kv_idx] < 0.5f);
            if (dropped) continue;
            
            // Compute attention score
            float score = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                score += Q_bh[q_idx * D + dd] * K_bh[kv_idx * D + dd];
            }
            score *= scale;
            
            // Compute attention probability
            float lse = L_bh[q_idx];
            float p = expf(score - lse);
            
            // Apply dropout scaling
            if (use_dropout && !dropped) {
                p *= dropout_scale;
            }
            
            // Compute Di = sum_j(dO_j * O_j)
            float di = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                di += dO_bh[q_idx * D + dd] * O_bh[q_idx * D + dd];
            }
            
            // dP = dO · V for this position
            float dp = dO_bh[q_idx * D + d] * V_bh[kv_idx * D + d];
            
            // Through softmax: dS = P * (dP_full - Di)
            float dp_full = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                dp_full += dO_bh[q_idx * D + dd] * V_bh[kv_idx * D + dd];
            }
            float ds = p * (dp_full - di);
            
            // Accumulate dK: dK += dS * Q * scale
            dk_sum += ds * Q_bh[q_idx * D + d] * scale;
            
            // Accumulate dV: dV += P * dO  
            dv_sum += p * dO_bh[q_idx * D + d];
            
            // Accumulate dQ using atomics
            atomicAdd(&dQ_bh[q_idx * D + d], ds * K_bh[kv_idx * D + d] * scale);
        }
        
        // Write dK and dV
        dK_bh[kv_idx * D + d] = dk_sum;
        dV_bh[kv_idx * D + d] = dv_sum;
    }
}

// Dispatch functions
void flash_attention_hip_dispatch(
    const Tensor& q, const Tensor& k, const Tensor& v,
    Tensor& output, Tensor& lse,
    float scale, bool is_causal,
    float dropout_p, bool training, Tensor* dropout_mask_out) {
    
    int64_t B = q.shape()[0];
    int64_t H = q.shape()[1];
    int64_t N = q.shape()[2];
    int64_t D = q.shape()[3];
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, B * H);
    
    bool use_dropout = training && dropout_p > 0.0f;
    float* mask_ptr = (use_dropout && dropout_mask_out) ? 
                        dropout_mask_out->data_ptr<float>() : nullptr;
    
    // Use a seed based on current time for randomness
    uint64_t seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now()
                                          .time_since_epoch().count());
    
    hipLaunchKernelGGL(
        flash_attention_forward_kernel,
        blocks, threads, 0, stream,
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        output.data_ptr<float>(), lse.data_ptr<float>(),
        mask_ptr,
        B, H, N, D, scale, is_causal,
        dropout_p, use_dropout, seed);
}

void flash_attention_backward_hip_dispatch(
    const Tensor& grad_output,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& output, const Tensor& lse,
    Tensor& grad_q, Tensor& grad_k, Tensor& grad_v,
    float scale, bool is_causal,
    float dropout_p, const Tensor* dropout_mask) {
    
    int64_t B = q.shape()[0];
    int64_t H = q.shape()[1];
    int64_t N = q.shape()[2];
    int64_t D = q.shape()[3];
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, B * H);
    
    // Zero the dQ tensor (because of atomic adds)
    (void)hipMemsetAsync(grad_q.data_ptr<float>(), 0, grad_q.numel() * sizeof(float), stream);
    
    bool use_dropout = dropout_p > 0.0f && dropout_mask != nullptr && dropout_mask->defined();
    const float* mask_ptr = use_dropout ? dropout_mask->data_ptr<float>() : nullptr;
    
    hipLaunchKernelGGL(
        flash_attention_backward_kernel,
        blocks, threads, 0, stream,
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        output.data_ptr<float>(), grad_output.data_ptr<float>(), lse.data_ptr<float>(),
        mask_ptr,
        grad_q.data_ptr<float>(), grad_k.data_ptr<float>(), grad_v.data_ptr<float>(),
        B, H, N, D, scale, is_causal,
        dropout_p, use_dropout);
}

} // namespace vesper::ops
