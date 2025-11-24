#include <vesper/ops/gemm.h>
#include <vesper/core/tensor.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace vesper::ops {

// Tiling parameters
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;

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

    dim3 threads(256);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    hipLaunchKernelGGL(
        gemm_register_tiled_kernel<float>,
        blocks,
        threads,
        0, stream,
        a.data_ptr<const float>(), stride_a_row, stride_a_col,
        b.data_ptr<const float>(), stride_b_row, stride_b_col,
        c.data_ptr<float>(), stride_c_row, stride_c_col,
        M, N, K
    );
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