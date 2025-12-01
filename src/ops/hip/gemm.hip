#include <vesper/ops/gemm.h>
#include <vesper/core/tensor.h>
#include <vesper/core/stream.h>
#include <vesper/core/vectorization.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace vesper::ops {

// =============================================================================
// Auto-Tuning GEMM Configuration
// =============================================================================
// Different tile sizes optimized for different matrix shapes

// Config 0: Small tiles (32x32) - Best for small matrices and low parallelism
constexpr int BM_SMALL = 32;
constexpr int BN_SMALL = 32;
constexpr int BK_SMALL = 16;
constexpr int TM_SMALL = 4;
constexpr int TN_SMALL = 4;
// Threads = (32/4) * (32/4) = 8 * 8 = 64

// Config 1: Medium tiles (64x64) - Good balance for moderate matrices
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
// Threads = (64/4) * (64/4) = 16 * 16 = 256

// Config 2: Large tiles (128x128) - Best for large matrices
constexpr int BM_LARGE = 128;
constexpr int BN_LARGE = 128;
constexpr int BK_LARGE = 16;
constexpr int TM_LARGE = 8;
constexpr int TN_LARGE = 8;
// Threads = (128/8) * (128/8) = 16 * 16 = 256

// Tiling parameters for FP16 (larger tiles for better throughput)
constexpr int BM_FP16 = 64;
constexpr int BN_FP16 = 64;
constexpr int BK_FP16 = 16;
constexpr int TM_FP16 = 4;
constexpr int TN_FP16 = 4;

// =============================================================================
// Small Tile GEMM Kernel (32x32) - Optimized for small/skinny matrices
// =============================================================================
__global__ __launch_bounds__(64, 4)
void gemm_small_tile_kernel(
    const float* __restrict__ A, int64_t stride_a_row,
    const float* __restrict__ B, int64_t stride_b_row,
    float* __restrict__ C, int64_t stride_c_row,
    int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int ty = tid / (BN_SMALL / TN_SMALL);  // 0..7
    const int tx = tid % (BN_SMALL / TN_SMALL);  // 0..7
    
    const int block_row = blockIdx.y * BM_SMALL;
    const int block_col = blockIdx.x * BN_SMALL;
    
    __shared__ float sA[BK_SMALL][BM_SMALL + 1];  // Transposed, +1 to avoid bank conflicts
    __shared__ float sB[BK_SMALL][BN_SMALL + 1];
    
    float rC[TM_SMALL][TN_SMALL] = {0.0f};
    
    // Load indices: 64 threads, 32*16=512 elements each for A and B
    // Each thread loads 8 elements
    const int loads_per_thread = (BM_SMALL * BK_SMALL) / 64;  // 8
    
    for (int k = 0; k < K; k += BK_SMALL) {
        // Cooperative load of A (transposed into shared memory)
        #pragma unroll
        for (int i = 0; i < loads_per_thread; ++i) {
            int elem_idx = tid + i * 64;
            int row = elem_idx % BM_SMALL;
            int col = elem_idx / BM_SMALL;
            int global_row = block_row + row;
            int global_col = k + col;
            
            if (global_row < M && global_col < K) {
                sA[col][row] = A[global_row * stride_a_row + global_col];
            } else {
                sA[col][row] = 0.0f;
            }
        }
        
        // Cooperative load of B
        #pragma unroll
        for (int i = 0; i < loads_per_thread; ++i) {
            int elem_idx = tid + i * 64;
            int row = elem_idx / BN_SMALL;
            int col = elem_idx % BN_SMALL;
            int global_row = k + row;
            int global_col = block_col + col;
            
            if (global_row < K && global_col < N) {
                sB[row][col] = B[global_row * stride_b_row + global_col];
            } else {
                sB[row][col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int sub_k = 0; sub_k < BK_SMALL; ++sub_k) {
            float rA[TM_SMALL], rB[TN_SMALL];
            
            #pragma unroll
            for (int i = 0; i < TM_SMALL; ++i) {
                rA[i] = sA[sub_k][ty * TM_SMALL + i];
            }
            #pragma unroll
            for (int j = 0; j < TN_SMALL; ++j) {
                rB[j] = sB[sub_k][tx * TN_SMALL + j];
            }
            
            #pragma unroll
            for (int i = 0; i < TM_SMALL; ++i) {
                #pragma unroll
                for (int j = 0; j < TN_SMALL; ++j) {
                    rC[i][j] += rA[i] * rB[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    const int c_row_base = block_row + ty * TM_SMALL;
    const int c_col_base = block_col + tx * TN_SMALL;
    
    #pragma unroll
    for (int i = 0; i < TM_SMALL; ++i) {
        int row = c_row_base + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN_SMALL; ++j) {
                int col = c_col_base + j;
                if (col < N) {
                    C[row * stride_c_row + col] = rC[i][j];
                }
            }
        }
    }
}

// =============================================================================
// Optimized FP32 GEMM Kernel - Large Tiles with Double Buffering
// =============================================================================

__global__ __launch_bounds__(256, 2)
void gemm_large_tile_kernel(
    const float* __restrict__ A, int64_t stride_a_row,
    const float* __restrict__ B, int64_t stride_b_row,
    float* __restrict__ C, int64_t stride_c_row,
    int M, int N, int K)
{
    // Thread indexing
    const int tid = threadIdx.x;
    const int ty = tid / (BN_LARGE / TN_LARGE);  // 0..15
    const int tx = tid % (BN_LARGE / TN_LARGE);  // 0..15
    
    // Block position
    const int block_row = blockIdx.y * BM_LARGE;
    const int block_col = blockIdx.x * BN_LARGE;
    
    // Double-buffered shared memory
    // +4 padding to avoid bank conflicts (128-bit boundary)
    __shared__ float sA[2][BK_LARGE][BM_LARGE + 4];  // Transposed for coalesced access
    __shared__ float sB[2][BK_LARGE][BN_LARGE + 4];
    
    // Register tile for output accumulation
    float rC[TM_LARGE][TN_LARGE] = {0.0f};
    
    // Register storage for A and B fragments
    float rA[TM_LARGE];
    float rB[TN_LARGE];
    
    // Indices for loading A and B cooperatively
    // A tile: [BM_LARGE, BK_LARGE] = [128, 16] = 2048 elements
    // B tile: [BK_LARGE, BN_LARGE] = [16, 128] = 2048 elements
    // 256 threads, each loads 8 elements from A and 8 from B
    
    const int a_loads_per_thread = (BM_LARGE * BK_LARGE) / 256;  // 8
    const int b_loads_per_thread = (BK_LARGE * BN_LARGE) / 256;  // 8
    
    // Calculate load positions
    // For A: load in column-major to store transposed
    const int a_thread_col = tid % BK_LARGE;       // 0..15
    const int a_thread_row_base = tid / BK_LARGE;  // 0..15, stride by 16
    
    // For B: load in row-major
    const int b_thread_row = tid / (BN_LARGE / 4);  // 0..15 for loading 4 elements per thread row-wise
    const int b_thread_col_base = (tid % (BN_LARGE / 4)) * 4;  // 0, 4, 8, ... 124
    
    int write_stage = 0;
    
    // Prefetch first tile
    #pragma unroll
    for (int i = 0; i < a_loads_per_thread; ++i) {
        int row = a_thread_row_base + i * 16;
        int global_row = block_row + row;
        int global_col = a_thread_col;
        
        if (global_row < M && global_col < K) {
            sA[0][a_thread_col][row] = A[global_row * stride_a_row + global_col];
        } else {
            sA[0][a_thread_col][row] = 0.0f;
        }
    }
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {  // Load 2 elements at a time per iteration (total 8)
        int row = b_thread_row;
        int col = b_thread_col_base + i;
        int global_row = row;
        int global_col = block_col + col;
        
        if (global_row < K && global_col < N) {
            sB[0][row][col] = B[global_row * stride_b_row + global_col];
        } else {
            sB[0][row][col] = 0.0f;
        }
    }
    // Second set of B loads for remaining rows
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = b_thread_row;
        int col = b_thread_col_base + i + 64;  // Second half
        if (col < BN_LARGE) {
            int global_row = row;
            int global_col = block_col + col;
            
            if (global_row < K && global_col < N) {
                sB[0][row][col] = B[global_row * stride_b_row + global_col];
            } else {
                sB[0][row][col] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    // Main loop over K dimension
    for (int k_tile = BK_LARGE; k_tile < K; k_tile += BK_LARGE) {
        int read_stage = write_stage;
        write_stage = 1 - write_stage;
        
        // Prefetch next tile
        #pragma unroll
        for (int i = 0; i < a_loads_per_thread; ++i) {
            int row = a_thread_row_base + i * 16;
            int global_row = block_row + row;
            int global_col = k_tile + a_thread_col;
            
            if (global_row < M && global_col < K) {
                sA[write_stage][a_thread_col][row] = A[global_row * stride_a_row + global_col];
            } else {
                sA[write_stage][a_thread_col][row] = 0.0f;
            }
        }
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = b_thread_row;
            int col = b_thread_col_base + i;
            int global_row = k_tile + row;
            int global_col = block_col + col;
            
            if (global_row < K && global_col < N) {
                sB[write_stage][row][col] = B[global_row * stride_b_row + global_col];
            } else {
                sB[write_stage][row][col] = 0.0f;
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = b_thread_row;
            int col = b_thread_col_base + i + 64;
            if (col < BN_LARGE) {
                int global_row = k_tile + row;
                int global_col = block_col + col;
                
                if (global_row < K && global_col < N) {
                    sB[write_stage][row][col] = B[global_row * stride_b_row + global_col];
                } else {
                    sB[write_stage][row][col] = 0.0f;
                }
            }
        }
        
        // Compute on current tile
        #pragma unroll
        for (int sub_k = 0; sub_k < BK_LARGE; ++sub_k) {
            // Load A fragment from shared memory (transposed layout)
            #pragma unroll
            for (int i = 0; i < TM_LARGE; ++i) {
                rA[i] = sA[read_stage][sub_k][ty * TM_LARGE + i];
            }
            
            // Load B fragment
            #pragma unroll
            for (int j = 0; j < TN_LARGE; ++j) {
                rB[j] = sB[read_stage][sub_k][tx * TN_LARGE + j];
            }
            
            // Outer product accumulation
            #pragma unroll
            for (int i = 0; i < TM_LARGE; ++i) {
                #pragma unroll
                for (int j = 0; j < TN_LARGE; ++j) {
                    rC[i][j] += rA[i] * rB[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Process last tile
    #pragma unroll
    for (int sub_k = 0; sub_k < BK_LARGE; ++sub_k) {
        #pragma unroll
        for (int i = 0; i < TM_LARGE; ++i) {
            rA[i] = sA[write_stage][sub_k][ty * TM_LARGE + i];
        }
        
        #pragma unroll
        for (int j = 0; j < TN_LARGE; ++j) {
            rB[j] = sB[write_stage][sub_k][tx * TN_LARGE + j];
        }
        
        #pragma unroll
        for (int i = 0; i < TM_LARGE; ++i) {
            #pragma unroll
            for (int j = 0; j < TN_LARGE; ++j) {
                rC[i][j] += rA[i] * rB[j];
            }
        }
    }
    
    // Store results
    const int c_row_base = block_row + ty * TM_LARGE;
    const int c_col_base = block_col + tx * TN_LARGE;
    
    #pragma unroll
    for (int i = 0; i < TM_LARGE; ++i) {
        const int row = c_row_base + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN_LARGE; ++j) {
                const int col = c_col_base + j;
                if (col < N) {
                    C[row * stride_c_row + col] = rC[i][j];
                }
            }
        }
    }
}

__global__ void gemm_fp16_kernel(
    const __half* __restrict__ A, int64_t stride_a_row,
    const __half* __restrict__ B, int64_t stride_b_row,
    __half* __restrict__ C, int64_t stride_c_row,
    int M, int N, int K)
{
    int tid = threadIdx.x;
    int ty = tid / (BN_FP16 / TN_FP16);  // 0..15
    int tx = tid % (BN_FP16 / TN_FP16);  // 0..15
    
    // Shared memory for tiles - using __half
    __shared__ __half sA[BM_FP16][BK_FP16 + 1];  // [64][17] +1 to avoid bank conflicts
    __shared__ __half sB[BK_FP16][BN_FP16 + 1];  // [16][65]
    
    // Registers for output tile - using float for accumulation
    float rC[TM_FP16][TN_FP16] = {0.0f};
    
    int block_row = blockIdx.y * BM_FP16;
    int block_col = blockIdx.x * BN_FP16;
    
    // Each thread loads multiple elements to fill the shared memory tiles
    // sA is [BM_FP16, BK_FP16] = [64, 16] = 1024 elements
    // sB is [BK_FP16, BN_FP16] = [16, 64] = 1024 elements
    // With 256 threads, each thread loads 4 elements from A and 4 from B
    
    for (int k = 0; k < K; k += BK_FP16) {
        // Load A tile cooperatively
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int elem_idx = tid + i * 256;  // 0..1023
            int row_a = elem_idx / BK_FP16;  // 0..63
            int col_a = elem_idx % BK_FP16;  // 0..15
            int global_row_a = block_row + row_a;
            int global_col_a = k + col_a;
            
            if (global_row_a < M && global_col_a < K) {
                sA[row_a][col_a] = A[global_row_a * stride_a_row + global_col_a];
            } else {
                sA[row_a][col_a] = __float2half(0.0f);
            }
        }
        
        // Load B tile cooperatively
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int elem_idx = tid + i * 256;
            int row_b = elem_idx / BN_FP16;  // 0..15
            int col_b = elem_idx % BN_FP16;  // 0..63
            int global_row_b = k + row_b;
            int global_col_b = block_col + col_b;
            
            if (global_row_b < K && global_col_b < N) {
                sB[row_b][col_b] = B[global_row_b * stride_b_row + global_col_b];
            } else {
                sB[row_b][col_b] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // Compute register tile
        #pragma unroll
        for (int sub_k = 0; sub_k < BK_FP16; ++sub_k) {
            // Load A values into registers
            float rA[TM_FP16];
            #pragma unroll
            for (int i = 0; i < TM_FP16; ++i) {
                rA[i] = __half2float(sA[ty * TM_FP16 + i][sub_k]);
            }
            
            // Load B values and compute
            #pragma unroll
            for (int j = 0; j < TN_FP16; ++j) {
                float rB = __half2float(sB[sub_k][tx * TN_FP16 + j]);
                #pragma unroll
                for (int i = 0; i < TM_FP16; ++i) {
                    rC[i][j] += rA[i] * rB;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results back to C (convert from FP32 accumulator to FP16)
    int global_c_base_row = block_row + ty * TM_FP16;
    int global_c_base_col = block_col + tx * TN_FP16;
    
    #pragma unroll
    for (int i = 0; i < TM_FP16; ++i) {
        int row = global_c_base_row + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN_FP16; ++j) {
                int col = global_c_base_col + j;
                if (col < N) {
                    C[row * stride_c_row + col] = __float2half(rC[i][j]);
                }
            }
        }
    }
}

// FP16 GEMM with FP32 output (useful for loss computation)
__global__ void gemm_fp16_fp32out_kernel(
    const __half* __restrict__ A, int64_t stride_a_row,
    const __half* __restrict__ B, int64_t stride_b_row,
    float* __restrict__ C, int64_t stride_c_row,
    int M, int N, int K)
{
    int tid = threadIdx.x;
    int ty = tid / (BN_FP16 / TN_FP16);  // 0..15
    int tx = tid % (BN_FP16 / TN_FP16);  // 0..15
    
    __shared__ __half sA[BM_FP16][BK_FP16 + 1];  // [64][17]
    __shared__ __half sB[BK_FP16][BN_FP16 + 1];  // [16][65]
    
    float rC[TM_FP16][TN_FP16] = {0.0f};
    
    int block_row = blockIdx.y * BM_FP16;
    int block_col = blockIdx.x * BN_FP16;
    
    for (int k = 0; k < K; k += BK_FP16) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int elem_idx = tid + i * 256;  // 0..1023
            int row_a = elem_idx / BK_FP16;  // 0..63
            int col_a = elem_idx % BK_FP16;  // 0..15
            int global_row_a = block_row + row_a;
            int global_col_a = k + col_a;
            
            if (global_row_a < M && global_col_a < K) {
                sA[row_a][col_a] = A[global_row_a * stride_a_row + global_col_a];
            } else {
                sA[row_a][col_a] = __float2half(0.0f);
            }
        }
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int elem_idx = tid + i * 256;
            int row_b = elem_idx / BN_FP16;  // 0..15
            int col_b = elem_idx % BN_FP16;  // 0..63
            int global_row_b = k + row_b;
            int global_col_b = block_col + col_b;
            
            if (global_row_b < K && global_col_b < N) {
                sB[row_b][col_b] = B[global_row_b * stride_b_row + global_col_b];
            } else {
                sB[row_b][col_b] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int sub_k = 0; sub_k < BK_FP16; ++sub_k) {
            float rA[TM_FP16];
            #pragma unroll
            for (int i = 0; i < TM_FP16; ++i) {
                rA[i] = __half2float(sA[ty * TM_FP16 + i][sub_k]);
            }
            
            #pragma unroll
            for (int j = 0; j < TN_FP16; ++j) {
                float rB = __half2float(sB[sub_k][tx * TN_FP16 + j]);
                #pragma unroll
                for (int i = 0; i < TM_FP16; ++i) {
                    rC[i][j] += rA[i] * rB;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store FP32 output directly
    int global_c_base_row = block_row + ty * TM_FP16;
    int global_c_base_col = block_col + tx * TN_FP16;
    
    #pragma unroll
    for (int i = 0; i < TM_FP16; ++i) {
        int row = global_c_base_row + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN_FP16; ++j) {
                int col = global_c_base_col + j;
                if (col < N) {
                    C[row * stride_c_row + col] = rC[i][j];
                }
            }
        }
    }
}

// Batched FP16 GEMM kernel
__global__ void gemm_fp16_batch_kernel(
    const __half* __restrict__ A, int64_t stride_a_batch, int64_t stride_a_row,
    const __half* __restrict__ B, int64_t stride_b_batch, int64_t stride_b_row,
    __half* __restrict__ C, int64_t stride_c_batch, int64_t stride_c_row,
    int M, int N, int K, int batch_count)
{
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_count) return;
    
    const __half* A_curr = A + batch_idx * stride_a_batch;
    const __half* B_curr = B + batch_idx * stride_b_batch;
    __half* C_curr = C + batch_idx * stride_c_batch;
    
    int tid = threadIdx.x;
    int ty = tid / (BN_FP16 / TN_FP16);  // 0..15
    int tx = tid % (BN_FP16 / TN_FP16);  // 0..15
    
    __shared__ __half sA[BM_FP16][BK_FP16 + 1];  // [64][17]
    __shared__ __half sB[BK_FP16][BN_FP16 + 1];  // [16][65]
    
    float rC[TM_FP16][TN_FP16] = {0.0f};
    
    int block_row = blockIdx.y * BM_FP16;
    int block_col = blockIdx.x * BN_FP16;
    
    for (int k = 0; k < K; k += BK_FP16) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int elem_idx = tid + i * 256;  // 0..1023
            int row_a = elem_idx / BK_FP16;  // 0..63
            int col_a = elem_idx % BK_FP16;  // 0..15
            int global_row_a = block_row + row_a;
            int global_col_a = k + col_a;
            
            if (global_row_a < M && global_col_a < K) {
                sA[row_a][col_a] = A_curr[global_row_a * stride_a_row + global_col_a];
            } else {
                sA[row_a][col_a] = __float2half(0.0f);
            }
        }
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int elem_idx = tid + i * 256;
            int row_b = elem_idx / BN_FP16;  // 0..15
            int col_b = elem_idx % BN_FP16;  // 0..63
            int global_row_b = k + row_b;
            int global_col_b = block_col + col_b;
            
            if (global_row_b < K && global_col_b < N) {
                sB[row_b][col_b] = B_curr[global_row_b * stride_b_row + global_col_b];
            } else {
                sB[row_b][col_b] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int sub_k = 0; sub_k < BK_FP16; ++sub_k) {
            float rA[TM_FP16];
            #pragma unroll
            for (int i = 0; i < TM_FP16; ++i) {
                rA[i] = __half2float(sA[ty * TM_FP16 + i][sub_k]);
            }
            
            #pragma unroll
            for (int j = 0; j < TN_FP16; ++j) {
                float rB = __half2float(sB[sub_k][tx * TN_FP16 + j]);
                #pragma unroll
                for (int i = 0; i < TM_FP16; ++i) {
                    rC[i][j] += rA[i] * rB;
                }
            }
        }
        
        __syncthreads();
    }
    
    int global_c_base_row = block_row + ty * TM_FP16;
    int global_c_base_col = block_col + tx * TN_FP16;
    
    #pragma unroll
    for (int i = 0; i < TM_FP16; ++i) {
        int row = global_c_base_row + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN_FP16; ++j) {
                int col = global_c_base_col + j;
                if (col < N) {
                    C_curr[row * stride_c_row + col] = __float2half(rC[i][j]);
                }
            }
        }
    }
}

// Dispatch function for FP16 GEMM
void gemm_fp16_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB) {
    int M = transA ? a.shape()[1] : a.shape()[0];
    int K = transA ? a.shape()[0] : a.shape()[1];
    int N = transB ? b.shape()[0] : b.shape()[1];
    
    int64_t stride_a_row = transA ? a.strides()[1] : a.strides()[0];
    int64_t stride_b_row = transB ? b.strides()[1] : b.strides()[0];
    int64_t stride_c_row = c.strides()[0];
    
    dim3 threads(256);
    dim3 blocks((N + BN_FP16 - 1) / BN_FP16, (M + BM_FP16 - 1) / BM_FP16);
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    if (c.dtype() == DType::Float32) {
        // FP16 compute with FP32 output
        hipLaunchKernelGGL(gemm_fp16_fp32out_kernel, blocks, threads, 0, stream,
            reinterpret_cast<const __half*>(a.data_ptr<void>()), stride_a_row,
            reinterpret_cast<const __half*>(b.data_ptr<void>()), stride_b_row,
            c.data_ptr<float>(), stride_c_row,
            M, N, K);
    } else {
        // FP16 compute with FP16 output
        hipLaunchKernelGGL(gemm_fp16_kernel, blocks, threads, 0, stream,
            reinterpret_cast<const __half*>(a.data_ptr<void>()), stride_a_row,
            reinterpret_cast<const __half*>(b.data_ptr<void>()), stride_b_row,
            reinterpret_cast<__half*>(c.data_ptr<void>()), stride_c_row,
            M, N, K);
    }
}

void gemm_fp16_batch_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& c,
                                   int64_t batch_count, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                                   bool transA, bool transB) {
    int a_rank = a.ndim();
    int b_rank = b.ndim();
    
    int64_t M = transA ? a.shape()[a_rank-1] : a.shape()[a_rank-2];
    int64_t K = transA ? a.shape()[a_rank-2] : a.shape()[a_rank-1];
    int64_t N = transB ? b.shape()[b_rank-2] : b.shape()[b_rank-1];
    
    int64_t stride_a_row = transA ? a.strides()[a_rank-1] : a.strides()[a_rank-2];
    int64_t stride_b_row = transB ? b.strides()[b_rank-1] : b.strides()[b_rank-2];
    int64_t stride_c_row = c.strides()[c.ndim()-2];
    
    dim3 threads(256);
    dim3 blocks((N + BN_FP16 - 1) / BN_FP16, (M + BM_FP16 - 1) / BM_FP16, batch_count);
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    hipLaunchKernelGGL(gemm_fp16_batch_kernel, blocks, threads, 0, stream,
        reinterpret_cast<const __half*>(a.data_ptr<void>()), stride_a, stride_a_row,
        reinterpret_cast<const __half*>(b.data_ptr<void>()), stride_b, stride_b_row,
        reinterpret_cast<__half*>(c.data_ptr<void>()), stride_c, stride_c_row,
        M, N, K, batch_count);
}

// Vectorized GEMM Kernel (float4)
// Assumes stride_col == 1 for A, B, C (RowMajor)
// Assumes K % 4 == 0, N % 4 == 0
// Assumes pointers are aligned to 16 bytes
__global__ void gemm_vectorized_kernel(
    const float* __restrict__ A, int64_t stride_a_row,
    const float* __restrict__ B, int64_t stride_b_row,
    float* __restrict__ C, int64_t stride_c_row,
    int M, int N, int K)
{
    int tid = threadIdx.x;
    int ty = tid / (BN / TN); // 0..15
    int tx = tid % (BN / TN); // 0..15

    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float rC[TM][TN] = {0.0f};
    float rA[TM];
    float rB[TN];

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Vectorized load indices
    // A: [BM, BK]. Load 1 float4 per thread.
    // tid 0..255.
    // row_a = tid / 4; col_a = (tid % 4) * 4;
    int row_a = tid >> 2;
    int col_a = (tid & 3) << 2;

    // B: [BK, BN]. Load 1 float4 per thread.
    // row_b = tid / 16; col_b = (tid % 16) * 4;
    int row_b = tid >> 4;
    int col_b = (tid & 15) << 2;

    for (int k = 0; k < K; k += BK) {
        // 1. Load A tile (Vectorized)
        int global_row_a = block_row + row_a;
        int global_col_a = k + col_a;
        
        if (global_row_a < M && global_col_a < K) { // K is multiple of 4, so global_col_a < K check is safe for float4
             float4 loaded_a = vesper::core::load_float4(&A[global_row_a * stride_a_row + global_col_a]);
             // Store to shared memory as float4
             *reinterpret_cast<float4*>(&sA[row_a][col_a]) = loaded_a;
        } else {
             *reinterpret_cast<float4*>(&sA[row_a][col_a]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // 2. Load B tile (Vectorized)
        int global_row_b = k + row_b;
        int global_col_b = block_col + col_b;
        
        if (global_row_b < K && global_col_b < N) { // N is multiple of 4
             float4 loaded_b = vesper::core::load_float4(&B[global_row_b * stride_b_row + global_col_b]);
             *reinterpret_cast<float4*>(&sB[row_b][col_b]) = loaded_b;
        } else {
             *reinterpret_cast<float4*>(&sB[row_b][col_b]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        __syncthreads();

        // 3. Compute
        #pragma unroll
        for (int sub_k = 0; sub_k < BK; ++sub_k) {
            for (int i = 0; i < TM; ++i) {
                rA[i] = sA[ty * TM + i][sub_k];
            }
            for (int j = 0; j < TN; ++j) {
                rB[j] = sB[sub_k][tx * TN + j];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    rC[i][j] += rA[i] * rB[j];
                }
            }
        }

        __syncthreads();
    }

    // 4. Store C (Vectorized)
    int global_c_base_row = block_row + ty * TM;
    int global_c_base_col = block_col + tx * TN; // tx*TN is multiple of 4 (TN=4)

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = global_c_base_row + i;
        int col = global_c_base_col; // Start of vector
        
        if (row < M && col < N) {
            // rC[i] is float[4]. We can cast to float4.
            float4 res = *reinterpret_cast<float4*>(&rC[i][0]);
            vesper::core::store_float4(&C[row * stride_c_row + col], res);
        }
    }
}

template <typename T>
__global__ void gemm_register_tiled_kernel(
    const T* A, int64_t stride_a_row, int64_t stride_a_col,
    const T* B, int64_t stride_b_row, int64_t stride_b_col,
    T* C, int64_t stride_c_row, int64_t stride_c_col,
    int M, int N, int K) 
{
    int tid = threadIdx.x;
    int ty = tid / (BN / TN);
    int tx = tid % (BN / TN);

    __shared__ T sA[BM][BK];
    __shared__ T sB[BK][BN];

    T rC[TM][TN] = {0.0f};
    T rA[TM];
    T rB[TN];

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    for (int k = 0; k < K; k += BK) {
        // 1. Load A tile
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid + i * 256;
            int row_a = load_idx / BK;
            int col_a = load_idx % BK;
            int global_row_a = block_row + row_a;
            int global_col_a = k + col_a;
            
            if (global_row_a < M && global_col_a < K) {
                sA[row_a][col_a] = A[global_row_a * stride_a_row + global_col_a * stride_a_col];
            } else {
                sA[row_a][col_a] = 0.0f;
            }
        }

        // 2. Load B tile
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid + i * 256;
            int row_b = load_idx / BN;
            int col_b = load_idx % BN;
            int global_row_b = k + row_b;
            int global_col_b = block_col + col_b;
            
            if (global_row_b < K && global_col_b < N) {
                sB[row_b][col_b] = B[global_row_b * stride_b_row + global_col_b * stride_b_col];
            } else {
                sB[row_b][col_b] = 0.0f;
            }
        }

        __syncthreads();

        // 3. Compute
        #pragma unroll
        for (int sub_k = 0; sub_k < BK; ++sub_k) {
            for (int i = 0; i < TM; ++i) {
                rA[i] = sA[ty * TM + i][sub_k];
            }
            for (int j = 0; j < TN; ++j) {
                rB[j] = sB[sub_k][tx * TN + j];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    rC[i][j] += rA[i] * rB[j];
                }
            }
        }

        __syncthreads();
    }

    // 4. Store C
    int global_c_base_row = block_row + ty * TM;
    int global_c_base_col = block_col + tx * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int row = global_c_base_row + i;
            int col = global_c_base_col + j;
            
            if (row < M && col < N) {
                C[row * stride_c_row + col * stride_c_col] = rC[i][j];
            }
        }
    }
}

void gemm_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB) {
    // Dispatch based on dtype
    if (a.dtype() == DType::Float16) {
        gemm_fp16_hip_dispatch(a, b, c, transA, transB);
        return;
    }
    
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("GEMM supports Float32 and Float16.");
    }
    
    int M = transA ? a.shape()[1] : a.shape()[0];
    int K = transA ? a.shape()[0] : a.shape()[1];
    int N = transB ? b.shape()[0] : b.shape()[1];

    int64_t stride_a_row = transA ? a.strides()[1] : a.strides()[0];
    int64_t stride_a_col = transA ? a.strides()[0] : a.strides()[1];
    
    int64_t stride_b_row = transB ? b.strides()[1] : b.strides()[0];
    int64_t stride_b_col = transB ? b.strides()[0] : b.strides()[1];
    
    int64_t stride_c_row = c.strides()[0];
    int64_t stride_c_col = c.strides()[1];

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    // Check if contiguous (required for optimized kernels)
    bool is_contiguous = (stride_a_col == 1) && (stride_b_col == 1) && (stride_c_col == 1);
    
    // ==========================================================================
    // Auto-tuning Kernel Selection based on matrix dimensions
    // ==========================================================================
    // The goal is to maximize GPU occupancy and compute efficiency.
    //
    // Key insight: Different tile sizes are optimal for different regimes:
    // - Small matrices (M,N < 128): Use small tiles (32x32) for more blocks
    // - Medium matrices (128 <= M,N < 512): Use medium tiles (64x64) 
    // - Large matrices (M,N >= 512): Use large tiles (128x128) for compute intensity
    // - Wide matrices (N >> M): Prioritize N dimension tiling
    //
    // Compute intensity = FLOPs / Bytes = 2MNK / (4*(MK + KN + MN))
    // Higher is better for GPU efficiency
    
    if (is_contiguous) {
        // Calculate compute intensity to decide kernel
        int64_t flops = 2LL * M * N * K;
        int64_t bytes = 4LL * (M * K + K * N + M * N);
        float compute_intensity = static_cast<float>(flops) / bytes;
        
        // Number of output elements (determines parallelism)
        int64_t output_size = static_cast<int64_t>(M) * N;
        
        // LARGE TILES: For high compute intensity or very wide matrices
        // Best when: lots of K iterations, large output, or vocabulary projection
        bool use_large = (M >= 512 && N >= 512 && K >= 256) ||
                         (N >= 8192 && M >= 128 && K >= 128) ||  // Wide (vocab projection)
                         (compute_intensity > 64.0f && output_size >= 65536);
        
        if (use_large) {
            dim3 threads(256);
            dim3 blocks((N + BN_LARGE - 1) / BN_LARGE, (M + BM_LARGE - 1) / BM_LARGE);
            
            hipLaunchKernelGGL(gemm_large_tile_kernel, blocks, threads, 0, stream,
                a.data_ptr<const float>(), stride_a_row,
                b.data_ptr<const float>(), stride_b_row,
                c.data_ptr<float>(), stride_c_row,
                M, N, K);
            return;
        }
        
        // SMALL TILES: For small matrices with low parallelism
        // More blocks = better occupancy on small problems
        // 32x32 tiles with 64 threads allows 4x more blocks than 64x64 with 256 threads
        // BUT: need minimum K for efficiency (each thread needs enough work)
        bool use_small = (M <= 64 && N <= 64) ||   // Very small output
                         (output_size < 4096 && K >= 64);  // Tiny output but reasonable K
        
        if (use_small) {
            dim3 threads(64);
            dim3 blocks((N + BN_SMALL - 1) / BN_SMALL, (M + BM_SMALL - 1) / BM_SMALL);
            
            hipLaunchKernelGGL(gemm_small_tile_kernel, blocks, threads, 0, stream,
                a.data_ptr<const float>(), stride_a_row,
                b.data_ptr<const float>(), stride_b_row,
                c.data_ptr<float>(), stride_c_row,
                M, N, K);
            return;
        }
        
        // MEDIUM TILES (default): Good balance for moderate matrices
        // Try vectorized loads if possible
        bool can_vectorize = (K % 4 == 0) && (N % 4 == 0) &&
                             ((uintptr_t)a.data_ptr<float>() % 16 == 0) &&
                             ((uintptr_t)b.data_ptr<float>() % 16 == 0) &&
                             ((uintptr_t)c.data_ptr<float>() % 16 == 0);

        dim3 threads(256);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
        
        if (can_vectorize) {
            hipLaunchKernelGGL(gemm_vectorized_kernel, blocks, threads, 0, stream,
                a.data_ptr<const float>(), stride_a_row,
                b.data_ptr<const float>(), stride_b_row,
                c.data_ptr<float>(), stride_c_row,
                M, N, K);
            return;
        }
        
        // Medium tiles without vectorization
        hipLaunchKernelGGL(gemm_register_tiled_kernel<float>, blocks, threads, 0, stream,
            a.data_ptr<const float>(), stride_a_row, stride_a_col,
            b.data_ptr<const float>(), stride_b_row, stride_b_col,
            c.data_ptr<float>(), stride_c_row, stride_c_col,
            M, N, K);
        return;
    }
    
    // Non-contiguous fallback: use generic register-tiled kernel
    dim3 threads(256);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    hipLaunchKernelGGL(gemm_register_tiled_kernel<float>, blocks, threads, 0, stream,
        a.data_ptr<const float>(), stride_a_row, stride_a_col,
        b.data_ptr<const float>(), stride_b_row, stride_b_col,
        c.data_ptr<float>(), stride_c_row, stride_c_col,
        M, N, K);
}

template <typename T>
__global__ void gemm_batch_tiled_kernel(
    const T* A, int64_t stride_a_batch, int64_t stride_a_row, int64_t stride_a_col,
    const T* B, int64_t stride_b_batch, int64_t stride_b_row, int64_t stride_b_col,
    T* C, int64_t stride_c_batch, int64_t stride_c_row, int64_t stride_c_col,
    int M, int N, int K, int batch_count) 
{
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_count) return;

    const T* A_curr = A + batch_idx * stride_a_batch;
    const T* B_curr = B + batch_idx * stride_b_batch;
    T* C_curr = C + batch_idx * stride_c_batch;

    int tid = threadIdx.x;
    int ty = tid / (BN / TN);
    int tx = tid % (BN / TN);

    __shared__ T sA[BM][BK];
    __shared__ T sB[BK][BN];

    T rC[TM][TN] = {0.0f};
    T rA[TM];
    T rB[TN];

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid + i * 256;
            int row_a = load_idx / BK;
            int col_a = load_idx % BK;
            int global_row_a = block_row + row_a;
            int global_col_a = k + col_a;
            
            if (global_row_a < M && global_col_a < K) {
                sA[row_a][col_a] = A_curr[global_row_a * stride_a_row + global_col_a * stride_a_col];
            } else {
                sA[row_a][col_a] = 0.0f;
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid + i * 256;
            int row_b = load_idx / BN;
            int col_b = load_idx % BN;
            int global_row_b = k + row_b;
            int global_col_b = block_col + col_b;
            
            if (global_row_b < K && global_col_b < N) {
                sB[row_b][col_b] = B_curr[global_row_b * stride_b_row + global_col_b * stride_b_col];
            } else {
                sB[row_b][col_b] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int sub_k = 0; sub_k < BK; ++sub_k) {
            for (int i = 0; i < TM; ++i) {
                rA[i] = sA[ty * TM + i][sub_k];
            }
            for (int j = 0; j < TN; ++j) {
                rB[j] = sB[sub_k][tx * TN + j];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    rC[i][j] += rA[i] * rB[j];
                }
            }
        }

        __syncthreads();
    }

    int global_c_base_row = block_row + ty * TM;
    int global_c_base_col = block_col + tx * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int row = global_c_base_row + i;
            int col = global_c_base_col + j;
            
            if (row < M && col < N) {
                C_curr[row * stride_c_row + col * stride_c_col] = rC[i][j];
            }
        }
    }
}

void gemm_batch_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& c, 
                             int64_t batch_count, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                             bool transA, bool transB) 
{
    // Dispatch based on dtype
    if (a.dtype() == DType::Float16) {
        gemm_fp16_batch_hip_dispatch(a, b, c, batch_count, stride_a, stride_b, stride_c, transA, transB);
        return;
    }
    
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("GEMM supports Float32 and Float16.");
    }

    int a_rank = a.ndim();
    int b_rank = b.ndim();
    int c_rank = c.ndim();
    
    int64_t M = transA ? a.shape()[a_rank-1] : a.shape()[a_rank-2];
    int64_t K = transA ? a.shape()[a_rank-2] : a.shape()[a_rank-1];
    int64_t N = transB ? b.shape()[b_rank-2] : b.shape()[b_rank-1];
    
    int64_t stride_a_row = transA ? a.strides()[a_rank-1] : a.strides()[a_rank-2];
    int64_t stride_a_col = transA ? a.strides()[a_rank-2] : a.strides()[a_rank-1];
    
    int64_t stride_b_row = transB ? b.strides()[b_rank-1] : b.strides()[b_rank-2];
    int64_t stride_b_col = transB ? b.strides()[b_rank-2] : b.strides()[b_rank-1];
    
    int64_t stride_c_row = c.strides()[c_rank-2];
    int64_t stride_c_col = c.strides()[c_rank-1];

    dim3 threads(256);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM, batch_count);

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    hipLaunchKernelGGL(
        gemm_batch_tiled_kernel<float>,
        blocks,
        threads,
        0, stream,
        a.data_ptr<const float>(), stride_a, stride_a_row, stride_a_col,
        b.data_ptr<const float>(), stride_b, stride_b_row, stride_b_col,
        c.data_ptr<float>(), stride_c, stride_c_row, stride_c_col,
        M, N, K, batch_count
    );
}

} // namespace vesper::ops