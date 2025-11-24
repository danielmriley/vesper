#include <vesper/ops/gemm.h>
#include <vesper/core/tensor.h>
#include <vesper/core/stream.h>
#include <cuda_runtime.h>
#include <vector>

namespace vesper::ops {

// Tiling parameters
// Block tile size (M, N, K)
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

// Thread tile size (M, N)
// Each thread computes a TM x TN patch of C
constexpr int TM = 4;
constexpr int TN = 4;

template <typename T>
__global__ void gemm_register_tiled_kernel(
    const T* A, int64_t stride_a_row, int64_t stride_a_col,
    const T* B, int64_t stride_b_row, int64_t stride_b_col,
    T* C, int64_t stride_c_row, int64_t stride_c_col,
    int M, int N, int K) 
{
    // Thread indices
    // We map 1D threadIdx.x to a 2D grid of threads (16x16)
    // blockDim.x must be 256
    int tid = threadIdx.x;
    int ty = tid / (BN / TN); // 0..15
    int tx = tid % (BN / TN); // 0..15

    // Shared memory for A and B tiles
    __shared__ T sA[BM][BK];
    __shared__ T sB[BK][BN];

    // Register file for C (accumulation)
    T rC[TM][TN] = {0.0f};

    // Registers for A and B fragments (loading from shared)
    T rA[TM];
    T rB[TN];

    // Global memory pointers for this block
    // We shift A and B pointers to the block's starting position
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Loop over K dimension in chunks of BK
    for (int k = 0; k < K; k += BK) {
        // 1. Load A tile from Global -> Shared
        // Each thread loads specific elements. 
        // We need to load BM x BK elements using 256 threads.
        // BM*BK = 64*16 = 1024 elements. 256 threads -> 4 elements per thread.
        // Tiled loading pattern:
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid + i * 256; // Linear index in tile
            int row_a = load_idx / BK;    // row in sA (0..63)
            int col_a = load_idx % BK;    // col in sA (0..15)
            
            // Global row/col
            int global_row_a = block_row + row_a;
            int global_col_a = k + col_a;
            
            if (global_row_a < M && global_col_a < K) {
                sA[row_a][col_a] = A[global_row_a * stride_a_row + global_col_a * stride_a_col];
            } else {
                sA[row_a][col_a] = 0.0f;
            }
        }

        // 2. Load B tile from Global -> Shared
        // BK*BN = 16*64 = 1024 elements. 4 elements per thread.
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid + i * 256;
            int row_b = load_idx / BN; // 0..15
            int col_b = load_idx % BN; // 0..63
            
            int global_row_b = k + row_b;
            int global_col_b = block_col + col_b;
            
            if (global_row_b < K && global_col_b < N) {
                sB[row_b][col_b] = B[global_row_b * stride_b_row + global_col_b * stride_b_col];
            } else {
                sB[row_b][col_b] = 0.0f;
            }
        }

        __syncthreads();

        // 3. Compute on Shared -> Register
        // Iterate over the BK dimension
        #pragma unroll
        for (int sub_k = 0; sub_k < BK; ++sub_k) {
            // Load column slice of A (TM x 1) and row slice of B (1 x TN) into regs
            for (int i = 0; i < TM; ++i) {
                rA[i] = sA[ty * TM + i][sub_k];
            }
            for (int j = 0; j < TN; ++j) {
                rB[j] = sB[sub_k][tx * TN + j];
            }

            // Outer product update
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    rC[i][j] += rA[i] * rB[j];
                }
            }
        }

        __syncthreads();
    }

    // 4. Store C from Register -> Global
    // Each thread writes its TM x TN block
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

void gemm_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB) {
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("GEMM only supports Float32 for now.");
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

    dim3 threads(256); // Fixed block size due to tiling params
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());

    gemm_register_tiled_kernel<float><<<blocks, threads, 0, stream>>>(
        a.data_ptr<const float>(), stride_a_row, stride_a_col,
        b.data_ptr<const float>(), stride_b_row, stride_b_col,
        c.data_ptr<float>(), stride_c_row, stride_c_col,
        M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
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

    // Thread indices
    int tid = threadIdx.x;
    int ty = tid / (BN / TN); // 0..15
    int tx = tid % (BN / TN); // 0..15

    // Shared memory for A and B tiles
    __shared__ T sA[BM][BK];
    __shared__ T sB[BK][BN];

    // Register file for C (accumulation)
    T rC[TM][TN] = {0.0f};

    // Registers for A and B fragments
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
                sA[row_a][col_a] = A_curr[global_row_a * stride_a_row + global_col_a * stride_a_col];
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
                sB[row_b][col_b] = B_curr[global_row_b * stride_b_row + global_col_b * stride_b_col];
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
                C_curr[row * stride_c_row + col * stride_c_col] = rC[i][j];
            }
        }
    }
}

void gemm_batch_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& c, 
                              int64_t batch_count, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                              bool transA, bool transB) 
{
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("GEMM only supports Float32 for now.");
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

    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());

    gemm_batch_tiled_kernel<float><<<blocks, threads, 0, stream>>>(
        a.data_ptr<const float>(), stride_a, stride_a_row, stride_a_col,
        b.data_ptr<const float>(), stride_b, stride_b_row, stride_b_col,
        c.data_ptr<float>(), stride_c, stride_c_row, stride_c_col,
        M, N, K, batch_count
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA batch kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace vesper::ops