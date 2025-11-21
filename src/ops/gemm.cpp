#include <vesper/ops/gemm.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <stdexcept>

namespace vesper::ops {

Tensor matmul(const Tensor& a, const Tensor& b) {
    // --- 1. Pre-condition Checks ---
    if (a.device() != b.device()) {
        throw std::runtime_error("Matmul requires tensors to be on the same device.");
    }
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::runtime_error("Matmul currently only supports 2D tensors.");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("Inner dimensions of matrices do not match for matmul.");
    }
    if (!a.is_contiguous() || !b.is_contiguous()) {
        throw std::runtime_error("Matmul currently requires contiguous tensors.");
    }

    // --- 2. Prepare Output Tensor ---
    const int M = a.shape()[0];
    const int K = a.shape()[1]; // a.k.a. b.shape()[0]
    const int N = b.shape()[1];

    Tensor c = empty({M, N}, a.dtype(), a.device());

    // --- 3. Dispatch to Backend-Specific Implementation ---
    switch (a.device()) {
        case Device::HIP:
#if USE_HIP_BACKEND
            gemm_hip_dispatch(a, b, c);
#else
            throw std::runtime_error("HIP backend not enabled during build, but required for matmul.");
#endif
            break;
        
        case Device::CPU: {
            const float* a_ptr = a.data_ptr<float>();
            const float* b_ptr = b.data_ptr<float>();
            float* c_ptr = c.data_ptr<float>();
            
            // Naive CPU GEMM
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += a_ptr[i * K + k] * b_ptr[k * N + j];
                    }
                    c_ptr[i * N + j] = sum;
                }
            }
            break;
        }
        case Device::CUDA:
            throw std::runtime_error("CUDA backend for matmul is not yet implemented.");

        default:
            throw std::runtime_error("Unsupported device for matmul.");
    }

    return c;
}

} // namespace vesper::ops
