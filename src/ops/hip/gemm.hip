#include <vesper/ops/gemm.h>
#include <vesper/core/tensor.h>
#include <hip/hip_runtime.h>

namespace vesper::ops {

// Define the size of the tile. This must be a compile-time constant.
// 16 or 32 are common values. Must match the thread block size.
constexpr int TILE_WIDTH = 16;

// Tiled GEMM Kernel for C = op(A) * op(B)
// M, N, K are dimensions of the operation: C is MxN, op(A) is MxK, op(B) is KxN
template <typename T>
__global__ void gemm_tiled_kernel(const T* A, const T* B, T* C, int M, int N, int K, bool transA, bool transB) {
    // 1. Thread and block identification
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // 2. Allocate tiles in shared memory
    __shared__ T tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ T tileB[TILE_WIDTH][TILE_WIDTH];

    // Accumulator for the partial sum computed by this thread
    T partial_sum = 0.0;

    // 3. Loop over tiles
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // 4. Load one tile of A and one tile of B into shared memory
        // Each thread in the block loads one element of each tile.
        
        // Load A
        int a_row = blockIdx.y * TILE_WIDTH + threadIdx.y; // Logical row in op(A)
        int a_col = t * TILE_WIDTH + threadIdx.x;          // Logical col in op(A)
        
        if (a_row < M && a_col < K) {
            if (!transA) {
                // A is [M, K], index = a_row * K + a_col
                tileA[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
            } else {
                // A is [K, M], index = a_col * M + a_row
                tileA[threadIdx.y][threadIdx.x] = A[a_col * M + a_row];
            }
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Load B
        int b_row = t * TILE_WIDTH + threadIdx.y;          // Logical row in op(B)
        int b_col = blockIdx.x * TILE_WIDTH + threadIdx.x; // Logical col in op(B)
        
        if (b_row < K && b_col < N) {
            if (!transB) {
                // B is [K, N], index = b_row * N + b_col
                tileB[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
            } else {
                // B is [N, K], index = b_col * K + b_row
                tileB[threadIdx.y][threadIdx.x] = B[b_col * K + b_row];
            }
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }

        // 5. Synchronize to ensure all threads have finished loading
        __syncthreads();

        // 6. Multiply the tiles from shared memory and accumulate results
        for (int k = 0; k < TILE_WIDTH; ++k) {
            partial_sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // 7. Synchronize before loading the next tile
        __syncthreads();
    }

    // 8. Write the final result to global memory
    if (row < M && col < N) {
        C[row * N + col] = partial_sum;
    }
}


// The C++ dispatch function that launches the kernel
void gemm_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB) {
    if (a.dtype() != DType::Float32) {
        throw std::runtime_error("GEMM only supports Float32 for now.");
    }
    
    // Determine logical dimensions M, N, K
    // op(A) is [M, K]
    // op(B) is [K, N]
    // C is [M, N]
    
    int M = transA ? a.shape()[1] : a.shape()[0];
    int K = transA ? a.shape()[0] : a.shape()[1];
    int N = transB ? b.shape()[0] : b.shape()[1];

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    auto kernel = gemm_tiled_kernel<float>;
    hipLaunchKernelGGL(
        kernel,
        blocks,
        threads,
        0, 0,
        a.data_ptr<const float>(),
        b.data_ptr<const float>(),
        c.data_ptr<float>(),
        M, N, K,
        transA, transB
    );
}

} // namespace vesper::ops
