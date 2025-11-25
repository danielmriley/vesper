#pragma once
#include <vesper/core/tensor.h>

namespace vesper::ops {

// Softmax
Tensor softmax(const Tensor& input, int64_t dim);

// Layer Normalization
// y = (x - mean) / sqrt(var + eps) * gamma + beta
Tensor layer_norm(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                  const Tensor& weight, const Tensor& bias, float eps);

// RMS Normalization
// y = x / RMS(x) * gamma
Tensor rms_norm(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                const Tensor& weight, float eps);

// Backend dispatchers
void softmax_cpu_dispatch(const Tensor& input, int64_t dim, Tensor& output);
void layer_norm_cpu_dispatch(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                             const Tensor& weight, const Tensor& bias, float eps, Tensor& output);
void rms_norm_cpu_dispatch(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                           const Tensor& weight, float eps, Tensor& output);

// CUDA/HIP dispatchers (declarations)
void softmax_cuda_dispatch(const Tensor& input, int64_t dim, Tensor& output);
void layer_norm_cuda_dispatch(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                              const Tensor& weight, const Tensor& bias, float eps, Tensor& output);
void rms_norm_cuda_dispatch(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                            const Tensor& weight, float eps, Tensor& output);

// Backward dispatchers for LayerNorm (GPU)
void layer_norm_backward_cuda_dispatch(const Tensor& grad_output, const Tensor& input, 
                                       const Tensor& weight, const std::vector<int64_t>& normalized_shape,
                                       float eps, Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias);
void layer_norm_backward_hip_dispatch(const Tensor& grad_output, const Tensor& input, 
                                      const Tensor& weight, const std::vector<int64_t>& normalized_shape,
                                      float eps, Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias);

// Backward dispatchers for RMSNorm (GPU)
void rms_norm_backward_cuda_dispatch(const Tensor& grad_output, const Tensor& input, 
                                     const Tensor& weight, const std::vector<int64_t>& normalized_shape,
                                     float eps, Tensor& grad_input, Tensor& grad_weight);
void rms_norm_backward_hip_dispatch(const Tensor& grad_output, const Tensor& input, 
                                    const Tensor& weight, const std::vector<int64_t>& normalized_shape,
                                    float eps, Tensor& grad_input, Tensor& grad_weight);

} // namespace vesper::ops
