#include <vesper/core/reference_ops.h>
#include <stdexcept>

namespace vesper::reference {

void add(const Tensor& a, const Tensor& b, Tensor& out) {
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    size_t n = a.numel();
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}

void sub(const Tensor& a, const Tensor& b, Tensor& out) {
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    size_t n = a.numel();
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] - b_ptr[i];
    }
}

void mul(const Tensor& a, const Tensor& b, Tensor& out) {
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    size_t n = a.numel();
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] * b_ptr[i];
    }
}

void sum(const Tensor& input, Tensor& out) {
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    float sum_val = 0.0f;
    size_t n = input.numel();
    for (size_t i = 0; i < n; ++i) {
        sum_val += in_ptr[i];
    }
    *out_ptr = sum_val;
}

void gemm(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB) {
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* c_ptr = c.data_ptr<float>();

    // Determine logical dimensions
    // op(A) is [M, K]
    // op(B) is [K, N]
    // C is [M, N]
    
    int64_t M = transA ? a.shape()[1] : a.shape()[0];
    int64_t K = transA ? a.shape()[0] : a.shape()[1];
    int64_t N = transB ? b.shape()[0] : b.shape()[1];

    // Physical dimensions
    int64_t rowsA = a.shape()[0];
    int64_t colsA = a.shape()[1];
    int64_t rowsB = b.shape()[0];
    int64_t colsB = b.shape()[1];

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // Access A[i, k]
                float val_a;
                if (!transA) {
                    // A is [M, K], index [i, k] -> i * K + k
                    val_a = a_ptr[i * colsA + k];
                } else {
                    // A is [K, M], logical [i, k] maps to physical [k, i] -> k * M + i
                    val_a = a_ptr[k * colsA + i];
                }

                // Access B[k, j]
                float val_b;
                if (!transB) {
                    // B is [K, N], index [k, j] -> k * N + j
                    val_b = b_ptr[k * colsB + j];
                } else {
                    // B is [N, K], logical [k, j] maps to physical [j, k] -> j * K + k
                    val_b = b_ptr[j * colsB + k];
                }

                sum += val_a * val_b;
            }
            c_ptr[i * N + j] = sum;
        }
    }
}

} // namespace vesper::reference
