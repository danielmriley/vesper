#include <vesper/ops/reduction.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <stdexcept>

namespace vesper::ops {

Tensor sum_rows_cpu(const Tensor& input) {
    if (input.device() != Device::CPU) throw std::runtime_error("sum_rows_cpu only supports CPU");
    
    int64_t M = input.shape()[0];
    int64_t N = input.shape()[1];
    
    Tensor out = zeros({N}, input.dtype(), input.device());
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            out_ptr[j] += in_ptr[i * N + j];
        }
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
    
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += in_ptr[i * N + j];
        }
        out_ptr[i] = sum;
    }
    return out;
}

} // namespace vesper::ops
