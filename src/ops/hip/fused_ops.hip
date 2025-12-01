/**
 * @file fused_ops.hip
 * @brief Fused GPU kernels for training acceleration
 * 
 * This file contains fused kernels that combine multiple operations
 * to reduce memory bandwidth requirements.
 */

#include <hip/hip_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/core/stream.h>
#include <vesper/core/factories.h>
#include <vesper/ops/fused_ops.h>
#include <cmath>

namespace vesper::ops {

// =============================================================================
// Fused RMSNorm + Linear Projection (Optimized Tiled Version)
// =============================================================================
// 
// Computes: output = RMSNorm(input, norm_weight, eps) @ linear_weight.T + bias
// 
// Strategy:
// - Phase 1: One kernel pass to compute inv_rms for each row, store to global memory
// - Phase 2: Tiled GEMM kernel that reads inv_rms and applies normalization on-the-fly
//
// This is better than trying to do everything in one kernel because:
// 1. RMS reduction requires different parallelism (all threads per row)
// 2. GEMM requires different parallelism (2D tiling)
// 3. Two separate optimized kernels > one suboptimal fused kernel

// Tiling parameters matching main GEMM kernel for best performance
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS_PER_BLOCK = 256;

// Phase 1: Compute inverse RMS for each row
// Grid: (rows / 256, 1)
// Block: 256 threads
__global__ void compute_inv_rms_kernel(
    const float* __restrict__ input,   // [rows, dim]
    float* __restrict__ inv_rms,       // [rows]
    int64_t rows,
    int64_t dim,
    float eps)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    const float* input_row = input + row * dim;
    
    // Compute sum of squares for this row
    float sum_sq = 0.0f;
    
    // Vectorized accumulation
    int64_t dim_vec = dim / 4;
    const float4* input_row_vec = reinterpret_cast<const float4*>(input_row);
    
    for (int k = 0; k < dim_vec; ++k) {
        float4 v = input_row_vec[k];
        sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    
    // Handle remainder
    for (int k = dim_vec * 4; k < dim; ++k) {
        float v = input_row[k];
        sum_sq += v * v;
    }
    
    float rms = sqrtf(sum_sq / static_cast<float>(dim) + eps);
    inv_rms[row] = 1.0f / rms;
}

// Phase 2: Tiled GEMM with on-the-fly normalization
// Computes: C[m,n] = sum_k( input[m,k] * inv_rms[m] * norm_weight[k] * weight[n,k] ) + bias[n]
//
// Grid: (N / BN, M / BM)
// Block: 256 threads
__global__ void gemm_with_norm_kernel(
    const float* __restrict__ input,        // [M, K]
    const float* __restrict__ inv_rms,      // [M]
    const float* __restrict__ norm_weight,  // [K]
    const float* __restrict__ weight,       // [N, K] - weight is stored row-major, we access as W[n,k]
    const float* __restrict__ bias,         // [N] or nullptr
    float* __restrict__ output,             // [M, N]
    int M, int N, int K)
{
    int tid = threadIdx.x;
    int ty = tid / (BN / TN);  // 0..15
    int tx = tid % (BN / TN);  // 0..15

    __shared__ float sA[BM][BK + 1];  // +1 to avoid bank conflicts
    __shared__ float sB[BK][BN + 1];  // +1 to avoid bank conflicts

    float rC[TM][TN] = {0.0f};
    float rA[TM];
    float rB[TN];

    int block_row = blockIdx.y * BM;  // Output row block
    int block_col = blockIdx.x * BN;  // Output col block

    // Load indices for cooperative loading
    int row_a = tid >> 2;        // tid / 4 -> 0..63 (loads BM=64 rows)
    int col_a = (tid & 3) << 2;  // (tid % 4) * 4 -> 0,4,8,12 (loads BK=16 cols as float4)
    
    int row_b = tid >> 4;        // tid / 16 -> 0..15 (loads BK=16 rows)
    int col_b = (tid & 15) << 2; // (tid % 16) * 4 -> 0,4,8,...,60 (loads BN=64 cols as float4)

    for (int k = 0; k < K; k += BK) {
        // === Load A tile (input, normalized on-the-fly) ===
        int global_row_a = block_row + row_a;
        int global_col_a = k + col_a;
        
        float4 a_vals = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (global_row_a < M && (global_col_a + 3) < K) {
            const float* ptr = input + global_row_a * K + global_col_a;
            a_vals = *reinterpret_cast<const float4*>(ptr);
            
            // Apply normalization: input * inv_rms * norm_weight
            float scale = inv_rms[global_row_a];
            const float4* nw = reinterpret_cast<const float4*>(norm_weight + global_col_a);
            float4 nw_vals = *nw;
            
            a_vals.x = a_vals.x * scale * nw_vals.x;
            a_vals.y = a_vals.y * scale * nw_vals.y;
            a_vals.z = a_vals.z * scale * nw_vals.z;
            a_vals.w = a_vals.w * scale * nw_vals.w;
        }
        
        sA[row_a][col_a + 0] = a_vals.x;
        sA[row_a][col_a + 1] = a_vals.y;
        sA[row_a][col_a + 2] = a_vals.z;
        sA[row_a][col_a + 3] = a_vals.w;

        // === Load B tile (transposed weight) ===
        // weight is [N, K], we need W^T which is [K, N]
        // sB[bk][bn] = weight[block_col + bn, k + bk]
        int global_col_b = block_col + col_b;
        int global_row_b = k + row_b;
        
        float4 b_vals = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (global_row_b < K && (global_col_b + 3) < N) {
            // Load 4 consecutive weights from 4 different output neurons
            // weight[global_col_b + 0, global_row_b], weight[global_col_b + 1, global_row_b], ...
            b_vals.x = weight[(global_col_b + 0) * K + global_row_b];
            b_vals.y = weight[(global_col_b + 1) * K + global_row_b];
            b_vals.z = weight[(global_col_b + 2) * K + global_row_b];
            b_vals.w = weight[(global_col_b + 3) * K + global_row_b];
        }
        
        sB[row_b][col_b + 0] = b_vals.x;
        sB[row_b][col_b + 1] = b_vals.y;
        sB[row_b][col_b + 2] = b_vals.z;
        sB[row_b][col_b + 3] = b_vals.w;

        __syncthreads();

        // === Compute register tile ===
        #pragma unroll
        for (int sub_k = 0; sub_k < BK; ++sub_k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                rA[i] = sA[ty * TM + i][sub_k];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                rB[j] = sB[sub_k][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    rC[i][j] += rA[i] * rB[j];
                }
            }
        }

        __syncthreads();
    }

    // === Store result with bias ===
    int global_c_base_row = block_row + ty * TM;
    int global_c_base_col = block_col + tx * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = global_c_base_row + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int col = global_c_base_col + j;
                if (col < N) {
                    float val = rC[i][j];
                    if (bias) {
                        val += bias[col];
                    }
                    output[row * N + col] = val;
                }
            }
        }
    }
}

void fused_rmsnorm_linear_hip_dispatch(
    const Tensor& input,
    const Tensor& norm_weight,
    const Tensor& linear_weight,
    const Tensor& bias,
    float eps,
    Tensor& output)
{
    // Input: [B*S, D_in] or [B, S, D_in]
    int64_t rows = input.numel() / input.shape().back();
    int64_t K = input.shape().back();  // dim_in
    int64_t N = linear_weight.shape()[0];  // dim_out
    int M = static_cast<int>(rows);
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    // Allocate temporary buffer for inv_rms
    Tensor inv_rms = vesper::empty({rows}, DType::Float32, Device::HIP, false);
    
    // Phase 1: Compute inv_rms for each row
    {
        int threads = 256;
        int blocks = (M + threads - 1) / threads;
        hipLaunchKernelGGL(compute_inv_rms_kernel, dim3(blocks), dim3(threads), 0, stream,
            input.data_ptr<float>(),
            inv_rms.data_ptr<float>(),
            rows, K, eps);
    }
    
    // Phase 2: Tiled GEMM with normalization
    {
        dim3 threads(THREADS_PER_BLOCK);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
        
        const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
        
        hipLaunchKernelGGL(gemm_with_norm_kernel, blocks, threads, 0, stream,
            input.data_ptr<float>(),
            inv_rms.data_ptr<float>(),
            norm_weight.data_ptr<float>(),
            linear_weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            M, static_cast<int>(N), static_cast<int>(K));
    }
}

// =============================================================================
// Fused RMSNorm + Linear Backward
// =============================================================================

// Backward pass for fused RMSNorm + Linear
// Given: grad_output [rows, dim_out]
// Computes: grad_input [rows, dim_in], grad_norm_weight [dim_in], 
//           grad_linear_weight [dim_out, dim_in], grad_bias [dim_out]

__global__ void fused_rmsnorm_linear_backward_kernel(
    const float* __restrict__ grad_output,   // [rows, dim_out]
    const float* __restrict__ input,         // [rows, dim_in]
    const float* __restrict__ norm_weight,   // [dim_in]
    const float* __restrict__ linear_weight, // [dim_out, dim_in]
    float* __restrict__ grad_input,          // [rows, dim_in]
    float* __restrict__ grad_norm_weight,    // [dim_in] - accumulated with atomics
    float* __restrict__ grad_linear_weight,  // [dim_out, dim_in] - accumulated with atomics
    float* __restrict__ grad_bias,           // [dim_out] - accumulated with atomics
    int64_t rows,
    int64_t dim_in,
    int64_t dim_out,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    const float* input_row = input + row * dim_in;
    const float* grad_out_row = grad_output + row * dim_out;
    float* grad_in_row = grad_input + row * dim_in;
    
    // --- Recompute RMS ---
    float sum_sq = 0.0f;
    for (int k = tid; k < dim_in; k += blockDim.x) {
        float val = input_row[k];
        sum_sq += val * val;
    }
    
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down(sum_sq, offset);
    }
    
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = blockDim.x / warpSize;
    
    __shared__ float warp_sums[32];
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();
    
    if (tid < num_warps) {
        sum_sq = warp_sums[tid];
    } else {
        sum_sq = 0.0f;
    }
    
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down(sum_sq, offset);
        }
    }
    
    __shared__ float s_inv_rms;
    __shared__ float s_inv_rms_cubed;
    if (tid == 0) {
        float ms = sum_sq / static_cast<float>(dim_in);
        float rms = sqrtf(ms + eps);
        s_inv_rms = 1.0f / rms;
        s_inv_rms_cubed = s_inv_rms * s_inv_rms * s_inv_rms;
    }
    __syncthreads();
    
    float inv_rms = s_inv_rms;
    float inv_rms_cubed = s_inv_rms_cubed;
    
    // --- Compute grad_input ---
    // d_loss/d_input = sum_j(grad_out[j] * W[j][:]) * d_normalized/d_input
    // where d_normalized/d_input = inv_rms * norm_weight - input * inv_rms^3 * (sum(grad*weight*norm_weight*input)) / dim_in
    
    // First pass: compute grad w.r.t. normalized input
    // grad_normalized[k] = sum_j(grad_out[j] * W[j][k]) * norm_weight[k]
    
    // Also accumulate grad_bias and grad_linear_weight
    for (int j = tid; j < dim_out; j += blockDim.x) {
        float g_out = grad_out_row[j];
        
        // grad_bias[j] += grad_out[j]
        if (grad_bias) {
            atomicAdd(&grad_bias[j], g_out);
        }
        
        // grad_linear_weight[j][k] += grad_out[j] * normalized_input[k]
        if (grad_linear_weight) {
            for (int k = 0; k < dim_in; ++k) {
                float normalized = input_row[k] * inv_rms * (norm_weight ? norm_weight[k] : 1.0f);
                atomicAdd(&grad_linear_weight[j * dim_in + k], g_out * normalized);
            }
        }
    }
    __syncthreads();
    
    // Compute grad_input using the RMSNorm backward formula
    // First: grad_normalized = W.T @ grad_out for each position
    for (int k = tid; k < dim_in; k += blockDim.x) {
        float grad_normalized = 0.0f;
        for (int j = 0; j < dim_out; ++j) {
            grad_normalized += grad_out_row[j] * linear_weight[j * dim_in + k];
        }
        
        float nw = norm_weight ? norm_weight[k] : 1.0f;
        grad_normalized *= nw;  // grad w.r.t. x_normalized (before norm_weight scaling)
        
        // Now apply RMSNorm backward
        // For now, simplified version
        grad_in_row[k] = grad_normalized * inv_rms;
        
        // grad_norm_weight accumulation
        if (grad_norm_weight && norm_weight) {
            float x_normalized = input_row[k] * inv_rms;
            float grad_nw_k = 0.0f;
            for (int j = 0; j < dim_out; ++j) {
                grad_nw_k += grad_out_row[j] * linear_weight[j * dim_in + k];
            }
            grad_nw_k *= x_normalized;
            atomicAdd(&grad_norm_weight[k], grad_nw_k);
        }
    }
    
    // TODO: Complete RMSNorm backward (currently missing the normalization correction term)
}

void fused_rmsnorm_linear_backward_hip_dispatch(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& norm_weight,
    const Tensor& linear_weight,
    float eps,
    Tensor& grad_input,
    Tensor& grad_norm_weight,
    Tensor& grad_linear_weight,
    Tensor& grad_bias)
{
    int64_t rows = input.numel() / input.shape().back();
    int64_t dim_in = input.shape().back();
    int64_t dim_out = linear_weight.shape()[0];
    
    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks(rows);
    
    size_t shared_mem = 64 * sizeof(float);
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    const float* norm_w_ptr = norm_weight.defined() ? norm_weight.data_ptr<float>() : nullptr;
    float* grad_nw_ptr = grad_norm_weight.defined() ? grad_norm_weight.data_ptr<float>() : nullptr;
    float* grad_lw_ptr = grad_linear_weight.defined() ? grad_linear_weight.data_ptr<float>() : nullptr;
    float* grad_b_ptr = grad_bias.defined() ? grad_bias.data_ptr<float>() : nullptr;
    
    hipLaunchKernelGGL(fused_rmsnorm_linear_backward_kernel, blocks, threads, shared_mem, stream,
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        norm_w_ptr,
        linear_weight.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        grad_nw_ptr,
        grad_lw_ptr,
        grad_b_ptr,
        rows, dim_in, dim_out, eps);
}

} // namespace vesper::ops
