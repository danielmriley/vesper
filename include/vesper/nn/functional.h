#pragma once
#include <vesper/core/tensor.h>

namespace vesper::nn::functional {

Tensor sigmoid(const Tensor& input);
void sigmoid_hip_dispatch(const Tensor& input, Tensor& output);
void sigmoid_cuda_dispatch(const Tensor& input, Tensor& output);

Tensor relu(const Tensor& input);
void relu_hip_dispatch(const Tensor& input, Tensor& output);
void relu_cuda_dispatch(const Tensor& input, Tensor& output);

Tensor gelu(const Tensor& input);
Tensor gelu_tanh(const Tensor& input);
Tensor gelu_erf(const Tensor& input);

Tensor dropout(const Tensor& input, double p = 0.5, bool training = true);
Tensor softmax(const Tensor& input, int64_t dim);
Tensor log_softmax(const Tensor& input, int64_t dim);

Tensor layer_norm(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                  const Tensor& weight = {}, const Tensor& bias = {}, float eps = 1e-5);
Tensor rms_norm(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                const Tensor& weight, float eps = 1e-5);

Tensor mse_loss(const Tensor& y_pred, const Tensor& y_true);

Tensor scaled_dot_product_attention(const Tensor& query, 
                                    const Tensor& key, 
                                    const Tensor& value, 
                                    bool is_causal = false, 
                                    double dropout_p = 0.0);

// RoPE
Tensor compute_rope_frequencies(int seq_len, int head_dim, int start_pos = 0, float theta = 10000.0f, Device device = Device::CPU);
Tensor apply_rotary_emb(const Tensor& x, const Tensor& freqs);

}
