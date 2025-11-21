#pragma once

namespace vesper {

class Tensor; // Forward declaration

namespace ops {
    // The full, multi-block sum operation
    Tensor sum(const Tensor& input);
    
    // Mean reduction
    Tensor mean(const Tensor& input);

    // The backend-specific dispatch for the full operation
    void sum_hip_dispatch(const Tensor& input, Tensor& output);

    // Backend-specific dispatch for summing rows (reducing [M, N] -> [N])
    void sum_rows_hip_dispatch(const Tensor& input, Tensor& output);
}
}
