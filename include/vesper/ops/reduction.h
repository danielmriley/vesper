#pragma once

#include <cstdint>

namespace vesper {

class Tensor; // Forward declaration

namespace ops {
    // The full, multi-block sum operation
    Tensor sum(const Tensor& input);

    // Sum over a specific dimension
    Tensor sum(const Tensor& input, int64_t dim, bool keepdim = false);
    
    // Mean reduction
    Tensor mean(const Tensor& input);

    // Max reduction (full)
    Tensor max(const Tensor& input);
    
    // Min reduction (full)
    Tensor min(const Tensor& input);

    // The backend-specific dispatch for the full operation
    void sum_hip_dispatch(const Tensor& input, Tensor& output);
    void sum_cuda_dispatch(const Tensor& input, Tensor& output);

    // Backend-specific dispatch for summing rows (reducing [M, N] -> [N])
    void sum_rows_hip_dispatch(const Tensor& input, Tensor& output);
    void sum_rows_cuda_dispatch(const Tensor& input, Tensor& output);

    // Backend-specific dispatch for summing cols (reducing [M, N] -> [M, 1])
    void sum_cols_hip_dispatch(const Tensor& input, Tensor& output);
    void sum_cols_cuda_dispatch(const Tensor& input, Tensor& output);
}
}
