#pragma once

#include <cstdint>

namespace vesper {

class Tensor; // Forward declaration

namespace ops {
    // The public-facing function for matrix multiplication
    Tensor matmul(const Tensor& a, const Tensor& b);

    // General Matrix Multiplication with transpose support
    Tensor gemm(const Tensor& a, const Tensor& b, bool transA, bool transB);

    // Backend-specific dispatch function (to be implemented in gemm.hip)
    void gemm_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB);
    void gemm_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB);

    // Batch GEMM dispatch
    void gemm_batch_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& c, 
                                 int64_t batch_count, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                                 bool transA, bool transB);
    void gemm_batch_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& c, 
                                  int64_t batch_count, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                                  bool transA, bool transB);
}
}
