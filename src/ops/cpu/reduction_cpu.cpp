#include <vesper/ops/reduction.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <stdexcept>
#include <thread>
#include <vector>
#include <algorithm>

namespace vesper::ops {

Tensor sum_rows_cpu(const Tensor& input) {
    if (input.device() != Device::CPU) throw std::runtime_error("sum_rows_cpu only supports CPU");
    
    int64_t M = input.shape()[0];
    int64_t N = input.shape()[1];
    
    Tensor out = zeros({N}, input.dtype(), input.device());
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    // Parallelize over columns for sum_rows
    unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    int64_t cols_per_thread = (N + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        int64_t start_col = t * cols_per_thread;
        int64_t end_col = std::min(start_col + cols_per_thread, N);
        
        if (start_col < end_col) {
            threads.emplace_back([=]() {
                for (int64_t j = start_col; j < end_col; ++j) {
                    float sum = 0.0f;
                    for (int64_t i = 0; i < M; ++i) {
                        sum += in_ptr[i * N + j];
                    }
                    out_ptr[j] = sum;
                }
            });
        }
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return out;
}

Tensor sum_cols_cpu(const Tensor& input) {
    if (input.device() != Device::CPU) throw std::runtime_error("sum_cols_cpu only supports CPU");
    
    int64_t M = input.shape()[0];
    int64_t N = input.shape()[1];
    
    Tensor out = zeros({M, 1}, input.dtype(), input.device());
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    // Parallelize over rows for sum_cols
    unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    int64_t rows_per_thread = (M + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        int64_t start_row = t * rows_per_thread;
        int64_t end_row = std::min(start_row + rows_per_thread, M);
        
        if (start_row < end_row) {
            threads.emplace_back([=]() {
                for (int64_t i = start_row; i < end_row; ++i) {
                    float sum = 0.0f;
                    for (int64_t j = 0; j < N; ++j) {
                        sum += in_ptr[i * N + j];
                    }
                    out_ptr[i] = sum;
                }
            });
        }
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return out;
}

} // namespace vesper::ops
