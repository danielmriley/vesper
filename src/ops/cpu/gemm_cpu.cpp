#include <vesper/ops/gemm.h>
#include <vesper/core/tensor.h>
#include <vector>
#include <thread>
#include <algorithm>
#include <stdexcept>

namespace vesper::ops {

// A tiled, multi-threaded GEMM implementation with transpose support.
constexpr int TILE_SIZE = 32;

void gemm_cpu_tiled_worker(const float* A, const float* B, float* C, 
                           int M, int N, int K,
                           bool transA, bool transB,
                           int64_t lda, int64_t ldb, int64_t ldc,
                           int start_row, int end_row) {
    
    // Initialize C to zero for the assigned rows
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] = 0.0f;
        }
    }

    for (int i0 = start_row; i0 < end_row; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {
            for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                
                int i_max = std::min(i0 + TILE_SIZE, end_row);
                int j_max = std::min(j0 + TILE_SIZE, N);
                int k_max = std::min(k0 + TILE_SIZE, K);

                for (int i = i0; i < i_max; ++i) {
                    for (int j = j0; j < j_max; ++j) {
                        float sum = 0.0f;
                        for (int k = k0; k < k_max; ++k) {
                            float val_a = !transA ? A[i * lda + k] : A[k * lda + i];
                            float val_b = !transB ? B[k * ldb + j] : B[j * ldb + k];
                            sum += val_a * val_b;
                        }
                        C[i * ldc + j] += sum;
                    }
                }
            }
        }
    }
}

void gemm_cpu_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB) {
    // Determine logical dimensions M, N, K
    int64_t M = c.shape()[0];
    int64_t N = c.shape()[1];
    int64_t K = transA ? a.shape()[0] : a.shape()[1];

    int64_t lda = a.shape()[1];
    int64_t ldb = b.shape()[1];
    int64_t ldc = c.shape()[1];

    const float* a_ptr = a.data_ptr<const float>();
    const float* b_ptr = b.data_ptr<const float>();
    float* c_ptr = c.data_ptr<float>();

    // 1. Determine number of threads
    const unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // 2. Divide the work
    int rows_per_thread = (M + num_threads - 1) / num_threads;

    // 3. Launch threads
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = std::min(static_cast<int64_t>(start_row + rows_per_thread), M);
        
        if (start_row < end_row) {
            threads.emplace_back(gemm_cpu_tiled_worker, a_ptr, b_ptr, c_ptr, 
                                 static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), 
                                 transA, transB, lda, ldb, ldc, 
                                 start_row, end_row);
        }
    }

    // 4. Wait for all threads to finish
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

} // namespace vesper::ops
