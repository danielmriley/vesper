#include <cuda_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/core/stream.h>
#include <vesper/ops/flash_attention.h>
#include <cfloat>
#include <cmath>

namespace vesper::ops {

// =============================================================================
// Constants & Configuration
// =============================================================================

constexpr int BLOCK_M = 64;   // Tile size for Queries
constexpr int BLOCK_N = 64;   // Tile size for Keys/Values
constexpr int HEAD_DIM = 64;  // Fixed head dimension for optimization
constexpr int WARP_SIZE = 32;

// =============================================================================
// Helper Functions
// =============================================================================

__device__ inline float atomicAdd_float(float* address, float val) {
    return atomicAdd(address, val);
}

// =============================================================================
// Forward Kernel
// =============================================================================

// Grid: (N / BLOCK_M, B * H)
// Block: (BLOCK_M) threads. Each thread handles one query row.
__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q, // [B, H, N, D]
    const float* __restrict__ K, // [B, H, N, D]
    const float* __restrict__ V, // [B, H, N, D]
    float* __restrict__ O,       // [B, H, N, D]
    float* __restrict__ L,       // [B, H, N] (LogSumExp)
    int N, int D, float scale, bool is_causal) 
{
    // 1. Setup Indices
    int bx = blockIdx.x; // Query block index
    int by = blockIdx.y; // Batch * Head index
    int tx = threadIdx.x;

    int b = by / (gridDim.y / ((gridDim.y == 0) ? 1 : 1)); // Assuming gridDim.y = B*H. 
    // Actually passed B*H as gridDim.y.
    // We need H to calculate offsets.
    // Let's assume gridDim.y = B * H.
    // We don't have H passed in directly, but we can infer or pass it.
    // Let's pass strides or full dims.
    // For simplicity, let's re-calculate pointers based on flat index.
    
    // Pointers to the start of the sequence for this specific batch & head
    int64_t offset_q = (int64_t)by * N * D;
    int64_t offset_l = (int64_t)by * N;

    const float* q_ptr = Q + offset_q;
    const float* k_ptr = K + offset_q; // Assuming same shape
    const float* v_ptr = V + offset_q;
    float* o_ptr = O + offset_q;
    float* l_ptr = L + offset_l;

    // 2. Initialize State (Online Softmax)
    // m_i: running max
    // l_i: running sum (denominator)
    // acc_i: running output (numerator)
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc_i[HEAD_DIM];
    
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        acc_i[d] = 0.0f;
    }

    // 3. Load Query into Registers
    // Each thread loads one query row: Q[bx * BLOCK_M + tx]
    int q_idx = bx * BLOCK_M + tx;
    float q_reg[HEAD_DIM];

    if (q_idx < N) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            q_reg[d] = q_ptr[q_idx * D + d];
        }
    } else {
        // Out of bounds threads just idle (but participate in syncs if needed)
        // We'll mask them out later.
    }

    // 4. Shared Memory for K and V tiles
    __shared__ float sK[BLOCK_N][HEAD_DIM];
    __shared__ float sV[BLOCK_N][HEAD_DIM];

    // 5. Loop over Key/Value Blocks
    int kv_end = is_causal ? (bx + 1) * BLOCK_M : N;
    // Align kv_end to BLOCK_N
    kv_end = (kv_end + BLOCK_N - 1) / BLOCK_N * BLOCK_N;
    if (kv_end > N) kv_end = N;

    for (int j = 0; j < kv_end; j += BLOCK_N) {
        // 5.1 Load K/V Tile into Shared Memory
        // We have BLOCK_M threads. We need to load BLOCK_N * D elements.
        // Since BLOCK_M = BLOCK_N = 64, and D = 64.
        // Each thread can load one row of K and V.
        int kv_idx = j + tx;
        
        if (kv_idx < N) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                sK[tx][d] = k_ptr[kv_idx * D + d];
                sV[tx][d] = v_ptr[kv_idx * D + d];
            }
        } else {
            // Zero padding for safety
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                sK[tx][d] = 0.0f;
                sV[tx][d] = 0.0f;
            }
        }
        __syncthreads();

        // 5.2 Compute Attention (if valid query)
        if (q_idx < N) {
            float scores[BLOCK_N];
            float row_m = -FLT_MAX;

            // Compute Scores S = Q * K^T
            #pragma unroll
            for (int k = 0; k < BLOCK_N; ++k) {
                int global_kv = j + k;
                if (is_causal && global_kv > q_idx) {
                    scores[k] = -FLT_MAX;
                } else if (global_kv >= N) {
                    scores[k] = -FLT_MAX;
                } else {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dot += q_reg[d] * sK[k][d];
                    }
                    scores[k] = dot * scale;
                }
                row_m = fmaxf(row_m, scores[k]);
            }

            // 5.3 Online Softmax Update
            // New max
            float m_new = fmaxf(m_i, row_m);
            float exp_diff = expf(m_i - m_new); // Scale factor for existing acc
            float exp_row = expf(row_m - m_new); // Scale factor for new scores (if row_m is max)
            
            // Update accumulator
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                acc_i[d] *= exp_diff;
            }
            l_i *= exp_diff;

            // Accumulate new block
            #pragma unroll
            for (int k = 0; k < BLOCK_N; ++k) {
                // P_ij = exp(S_ij - m_new)
                // But we only computed scores.
                // We need exp(scores[k] - m_new)
                float p = expf(scores[k] - m_new);
                l_i += p;
                
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    acc_i[d] += p * sV[k][d];
                }
            }
            m_i = m_new;
        }
        __syncthreads();
    }

    // 6. Write Output
    if (q_idx < N) {
        float inv_l = 1.0f / l_i;
        l_ptr[q_idx] = m_i + logf(l_i); // Store LSE for backward

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            o_ptr[q_idx * D + d] = acc_i[d] * inv_l;
        }
    }
}

// =============================================================================
// Backward Kernel
// =============================================================================

// Grid: (N / BLOCK_N, B * H)
// Parallelize over K/V blocks (j).
// Inner loop over Q blocks (i).
__global__ void flash_attn_bwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int N, int D, float scale, bool is_causal)
{
    int bx = blockIdx.x; // KV block index
    int by = blockIdx.y; // Batch * Head
    int tx = threadIdx.x;

    int64_t offset = (int64_t)by * N * D;
    int64_t offset_l = (int64_t)by * N;

    // Pointers
    const float* k_ptr = K + offset;
    const float* v_ptr = V + offset;
    const float* q_ptr = Q + offset;
    const float* o_ptr = O + offset;
    const float* do_ptr = dO + offset;
    const float* l_ptr = L + offset_l;
    
    float* dq_ptr = dQ + offset;
    float* dk_ptr = dK + offset;
    float* dv_ptr = dV + offset;

    // 1. Load K, V for this block into Registers
    // Each thread handles one KV row in the block
    int kv_idx = bx * BLOCK_N + tx;
    float k_reg[HEAD_DIM];
    float v_reg[HEAD_DIM];
    float dk_acc[HEAD_DIM];
    float dv_acc[HEAD_DIM];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        dk_acc[d] = 0.0f;
        dv_acc[d] = 0.0f;
    }

    if (kv_idx < N) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            k_reg[d] = k_ptr[kv_idx * D + d];
            v_reg[d] = v_ptr[kv_idx * D + d];
        }
    }

    // 2. Shared Memory for Q, dO, O, L
    __shared__ float sQ[BLOCK_M][HEAD_DIM];
    __shared__ float sdO[BLOCK_M][HEAD_DIM];
    // We don't strictly need O in shared if we compute D term differently, but standard formula uses it.
    // dP = P * (dP_unscaled - D)
    // D = sum(dO * O)
    // We can precompute D for the Q block.
    __shared__ float sD[BLOCK_M]; 

    // 3. Loop over Query Blocks
    // If causal, we only loop over Q blocks where i >= j
    int q_start = is_causal ? bx * BLOCK_N : 0;
    // Align
    q_start = (q_start / BLOCK_M) * BLOCK_M;

    for (int i = q_start; i < N; i += BLOCK_M) {
        // 3.1 Load Q, dO, O, L into Shared
        int q_idx = i + tx;
        float l_val = 0.0f;
        float d_val = 0.0f;

        if (q_idx < N) {
            l_val = l_ptr[q_idx];
            
            // Compute D term = dot(dO, O)
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                float val_q = q_ptr[q_idx * D + d];
                float val_do = do_ptr[q_idx * D + d];
                float val_o = o_ptr[q_idx * D + d];
                
                sQ[tx][d] = val_q;
                sdO[tx][d] = val_do;
                dot += val_do * val_o;
            }
            sD[tx] = dot;
        }
        __syncthreads();

        // 3.2 Compute Attention & Gradients
        if (kv_idx < N) {
            #pragma unroll
            for (int m = 0; m < BLOCK_M; ++m) {
                int global_q = i + m;
                if (global_q >= N) continue;
                if (is_causal && global_q < kv_idx) continue;

                // Compute Score S_ij
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += sQ[m][d] * k_reg[d];
                }
                score *= scale;

                // P_ij = exp(S_ij - L_i)
                // Note: L stored is LSE = m + log(sum)
                // So exp(S - LSE) = exp(S) / exp(LSE) = exp(S) / sum(exp(S)) = P
                // Wait, L stored in forward was m + log(l). Correct.
                // But we need to access L for the specific query m.
                // We didn't store L in shared memory for all m? 
                // Ah, we need L for each query in the block.
                // Let's reload L properly.
                // We only loaded L for 'tx'. We need L for 'm'.
                // Optimization: Store L in shared memory too.
            }
        }
        // ... (Refining the loop structure)
    }
}

// Re-writing Backward Kernel for correctness and simplicity
// We will use a simpler structure where we load L into shared memory.
__global__ void flash_attn_bwd_kernel_v2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ L, // LogSumExp
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int N, int D, float scale, bool is_causal)
{
    int bx = blockIdx.x; // KV block
    int by = blockIdx.y; // Batch * Head
    int tx = threadIdx.x;

    int64_t offset = (int64_t)by * N * D;
    int64_t offset_l = (int64_t)by * N;

    // Registers for K, V, dK, dV
    int kv_idx = bx * BLOCK_N + tx;
    float k_reg[HEAD_DIM], v_reg[HEAD_DIM];
    float dk_acc[HEAD_DIM] = {0.0f}, dv_acc[HEAD_DIM] = {0.0f};

    if (kv_idx < N) {
        #pragma unroll
        for(int d=0; d<HEAD_DIM; ++d) {
            k_reg[d] = K[offset + kv_idx*D + d];
            v_reg[d] = V[offset + kv_idx*D + d];
        }
    }

    // Shared Memory
    __shared__ float sQ[BLOCK_M][HEAD_DIM];
    __shared__ float sdO[BLOCK_M][HEAD_DIM];
    __shared__ float sL[BLOCK_M];
    __shared__ float sD[BLOCK_M]; // Delta term

    // Loop over Q blocks
    int q_start = is_causal ? bx * BLOCK_N : 0;
    q_start = (q_start / BLOCK_M) * BLOCK_M;

    for (int i = q_start; i < N; i += BLOCK_M) {
        // Load Q, dO, L, D
        int q_idx = i + tx;
        if (q_idx < N) {
            float dot = 0.0f;
            // We need O to compute D. O is not passed? 
            // Ah, we need to pass O or recompute it? 
            // Standard FA passes O.
            // Let's assume we can read O from global.
            // Wait, the signature above didn't have O. Let's add it.
            // Actually, let's assume D is precomputed or computed on the fly.
            // Computing D on the fly requires O.
            // Let's assume we pass O.
            // Wait, I missed O in the kernel signature above.
            // Let's fix that in the dispatch.
        }
        __syncthreads();
        
        // ... This is getting complicated to write inline.
        // Let's use the "Parallelize over Q" for dQ and "Parallelize over KV" for dK/dV approach?
        // No, single kernel is better.
    }
}

// Let's stick to the Forward Kernel first and a simplified Backward Kernel.
// For the backward kernel, I will implement the "Parallelize over KV" strategy properly.
// I need to pass O to it.

__global__ void flash_attn_bwd_kernel_final(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int N, int D, float scale, bool is_causal)
{
    int bx = blockIdx.x; // KV block
    int by = blockIdx.y; // Batch * Head
    int tx = threadIdx.x;

    int64_t offset = (int64_t)by * N * D;
    int64_t offset_l = (int64_t)by * N;

    int kv_idx = bx * BLOCK_N + tx;
    float k_reg[HEAD_DIM], v_reg[HEAD_DIM];
    float dk_acc[HEAD_DIM] = {0.0f}, dv_acc[HEAD_DIM] = {0.0f};

    if (kv_idx < N) {
        #pragma unroll
        for(int d=0; d<HEAD_DIM; ++d) {
            k_reg[d] = K[offset + kv_idx*D + d];
            v_reg[d] = V[offset + kv_idx*D + d];
        }
    }

    __shared__ float sQ[BLOCK_M][HEAD_DIM];
    __shared__ float sdO[BLOCK_M][HEAD_DIM];
    __shared__ float sL[BLOCK_M];
    __shared__ float sD[BLOCK_M];

    int q_start = is_causal ? bx * BLOCK_N : 0;
    q_start = (q_start / BLOCK_M) * BLOCK_M;

    for (int i = q_start; i < N; i += BLOCK_M) {
        int q_idx = i + tx;
        
        // Load Q, dO, L, compute D
        if (q_idx < N) {
            float val_l = L[offset_l + q_idx];
            sL[tx] = val_l;
            
            float dot = 0.0f;
            #pragma unroll
            for(int d=0; d<HEAD_DIM; ++d) {
                float val_q = Q[offset + q_idx*D + d];
                float val_do = dO[offset + q_idx*D + d];
                float val_o = O[offset + q_idx*D + d];
                sQ[tx][d] = val_q;
                sdO[tx][d] = val_do;
                dot += val_do * val_o;
            }
            sD[tx] = dot;
        }
        __syncthreads();

        // Compute gradients
        if (kv_idx < N) {
            #pragma unroll
            for (int m = 0; m < BLOCK_M; ++m) {
                int global_q = i + m;
                if (global_q >= N) continue;
                if (is_causal && global_q < kv_idx) continue;

                // S = Q * K
                float score = 0.0f;
                #pragma unroll
                for(int d=0; d<HEAD_DIM; ++d) {
                    score += sQ[m][d] * k_reg[d];
                }
                score *= scale;

                // P = exp(S - L)
                float p = expf(score - sL[m]);

                // dS = P * (dP - D)
                // dP = dO * V
                float dp = 0.0f;
                #pragma unroll
                for(int d=0; d<HEAD_DIM; ++d) {
                    dp += sdO[m][d] * v_reg[d];
                }
                
                float ds = p * (dp - sD[m]);

                // Accumulate dK, dV
                #pragma unroll
                for(int d=0; d<HEAD_DIM; ++d) {
                    dk_acc[d] += ds * sQ[m][d] * scale;
                    dv_acc[d] += p * sdO[m][d];
                }

                // Atomic Add to dQ
                // dQ = dS * K
                #pragma unroll
                for(int d=0; d<HEAD_DIM; ++d) {
                    atomicAdd_float(&dQ[offset + global_q*D + d], ds * k_reg[d] * scale);
                }
            }
        }
        __syncthreads();
    }

    // Write dK, dV
    if (kv_idx < N) {
        #pragma unroll
        for(int d=0; d<HEAD_DIM; ++d) {
            dK[offset + kv_idx*D + d] = dk_acc[d];
            dV[offset + kv_idx*D + d] = dv_acc[d];
        }
    }
}

// =============================================================================
// Dispatch
// =============================================================================

void flash_attention_cuda_dispatch(
    const Tensor& q, const Tensor& k, const Tensor& v,
    Tensor& output, Tensor& lse,
    float scale, bool is_causal,
    float dropout_p, bool training, Tensor* dropout_mask_out)
{
    int B = q.shape()[0];
    int H = q.shape()[1];
    int N = q.shape()[2];
    int D = q.shape()[3];

    if (D != HEAD_DIM) {
        throw std::runtime_error("CUDA FlashAttention currently only supports head_dim=64");
    }

    dim3 grid( (N + BLOCK_M - 1) / BLOCK_M, B * H );
    dim3 block(BLOCK_M);
    
    // Stream
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());

    flash_attn_fwd_kernel<<<grid, block, 0, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        output.data_ptr<float>(), lse.data_ptr<float>(),
        N, D, scale, is_causal
    );
}

void flash_attention_backward_cuda_dispatch(
    const Tensor& grad_output,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& output, const Tensor& lse,
    Tensor& grad_q, Tensor& grad_k, Tensor& grad_v,
    float scale, bool is_causal,
    float dropout_p, const Tensor* dropout_mask)
{
    int B = q.shape()[0];
    int H = q.shape()[1];
    int N = q.shape()[2];
    int D = q.shape()[3];

    if (D != HEAD_DIM) {
        throw std::runtime_error("CUDA FlashAttention currently only supports head_dim=64");
    }

    // Zero dQ first because we use atomicAdd
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
    cudaMemsetAsync(grad_q.data_ptr<float>(), 0, grad_q.numel() * sizeof(float), stream);

    dim3 grid( (N + BLOCK_N - 1) / BLOCK_N, B * H );
    dim3 block(BLOCK_N);

    flash_attn_bwd_kernel_final<<<grid, block, 0, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        output.data_ptr<float>(), grad_output.data_ptr<float>(), lse.data_ptr<float>(),
        grad_q.data_ptr<float>(), grad_k.data_ptr<float>(), grad_v.data_ptr<float>(),
        N, D, scale, is_causal
    );
}

} // namespace vesper::ops
