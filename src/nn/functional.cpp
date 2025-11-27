#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>
#include <vesper/core/macros.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/comparison.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/normalization.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/random.h>
#include <vesper/ops/cast.h>
#include <vesper/ops/stack.h>
#include <vesper/autograd/guard.h>
#include <cmath>
#include <limits>

namespace vesper::nn::functional {

void sigmoid_cpu_dispatch(const Tensor& input, Tensor& output) {
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    size_t n = input.numel();
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = 1.0f / (1.0f + std::exp(-in_ptr[i]));
    }
}

void relu_cpu_dispatch(const Tensor& input, Tensor& output) {
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    size_t n = input.numel();
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = in_ptr[i] > 0.0f ? in_ptr[i] : 0.0f;
    }
}

Tensor sigmoid(const Tensor& input) {
    bool result_requires_grad = input.requires_grad() && autograd::grad_mode_enabled;
    Tensor result = empty(input.shape(), input.dtype(), input.device(), result_requires_grad);
    
    if (input.device() == Device::CPU) {
        sigmoid_cpu_dispatch(input, result);
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        sigmoid_hip_dispatch(input, result);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        sigmoid_cuda_dispatch(input, result);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for sigmoid.");
    }

    if (result_requires_grad) {
        result.grad_node = std::make_shared<autograd::Node>();
        result.grad_node->next_edges.push_back({input.grad_node});

        // Capture a non-const copy of input to allow accumulate_grad
        Tensor input_nc = input;

        result.grad_node->backward_fn = [input_nc, weak_res=result.weak()]() mutable {
            auto result = weak_res.lock();
            if (!result) return;

            // Sigmoid gradient: grad_output * (output * (1 - output))
            // We need `ones` factory.
            // Since `ones` is not in factories.h (I checked earlier), I use `full` with 1.0f.
            auto ones = vesper::full(result->shape(), result->dtype(), result->device(), 1.0f);
            auto term2 = ops::sub(ones, *result);
            auto local_grad = ops::mul(*result, term2);
            auto final_grad = ops::mul(result->grad(), local_grad);
            input_nc.accumulate_grad(final_grad);
        };
    }
    return result;
}

Tensor relu(const Tensor& input) {
    bool result_requires_grad = input.requires_grad() && autograd::grad_mode_enabled;
    Tensor result = empty(input.shape(), input.dtype(), input.device(), result_requires_grad);
    
    if (input.device() == Device::CPU) {
        relu_cpu_dispatch(input, result);
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        relu_hip_dispatch(input, result);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        relu_cuda_dispatch(input, result);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for relu.");
    }

    if (result_requires_grad) {
        result.grad_node = std::make_shared<autograd::Node>();
        result.grad_node->next_edges.push_back({input.grad_node});
        
        // Capture non-const copy
        Tensor input_nc = input;

        result.grad_node->backward_fn = [input_nc, weak_res=result.weak()]() mutable {
            auto result = weak_res.lock();
            if (!result) return;

            // ReLU gradient: grad_input = grad_output * (input > 0)
            // 1. Create the mask from the original input
            auto mask = ops::greater_than(input_nc, 0.0f);

            // 2. Multiply the upstream gradient by the mask
            auto final_grad = ops::mul(result->grad(), mask);
            input_nc.accumulate_grad(final_grad);
        };
    }
    return result;
}

Tensor mse_loss(const Tensor& y_pred, const Tensor& y_true) {
    // MSE = mean((y_pred - y_true)^2)
    Tensor diff = ops::sub(y_pred, y_true);
    Tensor sq_diff = ops::mul(diff, diff);
    return ops::mean(sq_diff);
}

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    // Cross-entropy loss = -mean(log_softmax[target_class])
    // logits: [N, C] - raw scores
    // targets: [N] - integer class indices (Int32 or Int64)
    
    VESPER_CHECK(logits.ndim() == 2, "logits must be 2D [N, C]");
    VESPER_CHECK(targets.ndim() == 1, "targets must be 1D [N]");
    VESPER_CHECK(logits.shape()[0] == targets.shape()[0], "batch size mismatch");
    
    int64_t N = logits.shape()[0];
    int64_t C = logits.shape()[1];
    
    // Compute log_softmax
    Tensor log_probs = log_softmax(logits, -1);  // [N, C]
    
    // Gather the log probabilities at target indices
    // Manual implementation since we don't have gather op
    Tensor log_probs_cpu = log_probs.to(Device::CPU);
    Tensor targets_cpu = targets.to(Device::CPU);
    
    const float* lp_ptr = log_probs_cpu.data_ptr<float>();
    
    // Handle both Int32 and Int64 targets
    std::vector<float> selected(N);
    if (targets.dtype() == DType::Int32) {
        const int32_t* t_ptr = targets_cpu.data_ptr<int32_t>();
        for (int64_t i = 0; i < N; ++i) {
            int64_t idx = static_cast<int64_t>(t_ptr[i]);
            VESPER_CHECK(idx >= 0 && idx < C, "target index out of range");
            selected[i] = lp_ptr[i * C + idx];
        }
    } else if (targets.dtype() == DType::Int64) {
        const int64_t* t_ptr = targets_cpu.data_ptr<int64_t>();
        for (int64_t i = 0; i < N; ++i) {
            int64_t idx = t_ptr[i];
            VESPER_CHECK(idx >= 0 && idx < C, "target index out of range");
            selected[i] = lp_ptr[i * C + idx];
        }
    } else {
        VESPER_CHECK(false, "targets must be Int32 or Int64");
    }
    
    // Create tensor from selected values
    Tensor selected_log_probs = empty({N}, DType::Float32, logits.device(), false);
    selected_log_probs.copy_from_host(selected.data());
    
    // Negate and compute mean (cross-entropy = -mean(log_probs[target]))
    // neg(x) = mul(x, -1.0f)
    Tensor neg_log_probs = ops::mul(selected_log_probs, -1.0f);
    
    // Return mean with autograd support
    // We need to track gradients through log_softmax
    Tensor loss = ops::mean(neg_log_probs);
    
    // For autograd: d(loss)/d(logits) = (softmax(logits) - one_hot(targets)) / N
    // This is a well-known result for cross-entropy + softmax
    if (logits.requires_grad()) {
        // Store copies for backward
        auto logits_nc = logits;
        auto targets_nc = targets;
        
        loss.set_requires_grad(true);
        auto node = std::make_shared<autograd::Node>();
        node->next_edges.push_back({logits.grad_node});
        
        node->backward_fn = [logits_nc, targets_nc, N, C, weak_loss=loss.weak()]() mutable {
            auto loss_ptr = weak_loss.lock();
            if (!loss_ptr) return;
            
            // Get upstream gradient (for scalar loss, usually 1.0)
            Tensor grad_output = loss_ptr->grad();
            float upstream = grad_output.defined() ? grad_output.item<float>() : 1.0f;
            
            // Compute softmax from logits
            Tensor softmax_probs = ops::softmax(logits_nc, -1);  // [N, C]
            
            // Create one-hot from targets
            Tensor targets_cpu = targets_nc.to(Device::CPU);
            std::vector<float> one_hot_data(N * C, 0.0f);
            
            if (targets_nc.dtype() == DType::Int32) {
                const int32_t* t_ptr = targets_cpu.data_ptr<int32_t>();
                for (int64_t i = 0; i < N; ++i) {
                    one_hot_data[i * C + t_ptr[i]] = 1.0f;
                }
            } else {
                const int64_t* t_ptr = targets_cpu.data_ptr<int64_t>();
                for (int64_t i = 0; i < N; ++i) {
                    one_hot_data[i * C + t_ptr[i]] = 1.0f;
                }
            }
            
            Tensor one_hot = empty({N, C}, DType::Float32, softmax_probs.device(), false);
            one_hot.copy_from_host(one_hot_data.data());
            
            // grad_logits = (softmax - one_hot) * upstream / N
            Tensor grad_logits = ops::sub(softmax_probs, one_hot);
            grad_logits = ops::mul(grad_logits, upstream / static_cast<float>(N));
            
            // Accumulate gradient to logits
            logits_nc.accumulate_grad(grad_logits);
        };
        loss.grad_node = node;
    }
    
    return loss;
}

Tensor gelu(const Tensor& input) {
    // Default to tanh approximation as per GPT-2
    return ops::gelu(input);
}

Tensor gelu_tanh(const Tensor& input) {
    return ops::gelu(input);
}

Tensor gelu_erf(const Tensor& input) {
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    return ops::gelu_erf(input);
}

Tensor silu(const Tensor& input) {
    // SiLU (Swish): silu(x) = x * sigmoid(x)
    return ops::silu(input);
}

void silu_(Tensor& input) {
    // In-place SiLU
    ops::silu_(input);
}

Tensor dropout(const Tensor& input, double p, bool training) {
    if (p == 0.0 || !training) {
        return input;
    }
    
    // Generate noise
    Tensor noise = vesper::empty(input.shape(), input.dtype(), input.device(), false);
    ops::uniform_(noise, 0.0f, 1.0f);
    
    // Create mask: keep if noise > p
    // greater_than returns 1.0f or 0.0f (Float32)
    Tensor mask = ops::greater_than(noise, static_cast<float>(p));
    
    // Scale factor
    float scale = 1.0f / (1.0f - static_cast<float>(p));
    
    // Apply mask and scale
    Tensor output = ops::mul(input, mask);
    output = ops::mul(output, scale);
    
    return output;
}

Tensor softmax(const Tensor& input, int64_t dim) {
    return ops::softmax(input, dim);
}

Tensor log_softmax(const Tensor& input, int64_t dim) {
    // log_softmax(x) = x - log(sum(exp(x)))
    // Stable: x - M - log(sum(exp(x - M)))
    // But softmax(x) = exp(x-M) / sum(exp(x-M))
    // log_softmax(x) = (x-M) - log(sum(exp(x-M)))
    
    // We can reuse softmax implementation if it returns log probs? 
    // ops::softmax returns probs.
    
    // Implementation using existing ops:
    // 1. Max
    Tensor max_val = ops::max(input, dim, true);
    
    // 2. Subtract max
    Tensor shifted = ops::sub(input, max_val);
    
    // 3. Exp
    Tensor exp_val = ops::exp(shifted);
    
    // 4. Sum
    Tensor sum_exp = ops::sum(exp_val, dim, true);
    
    // 5. Log
    Tensor log_sum = ops::log(sum_exp);
    
    // 6. Result = shifted - log_sum
    return ops::sub(shifted, log_sum);
}

Tensor layer_norm(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                  const Tensor& weight, const Tensor& bias, float eps) {
    // If weight/bias are empty, we should probably create ones/zeros or handle it in op.
    // The op expects valid tensors if we pass them.
    // Let's assume for now the caller handles defaults or we pass empty tensors and op handles it.
    // My op implementation assumed valid pointers.
    // So I should handle defaults here.
    
    Tensor w = weight;
    Tensor b = bias;
    
    if (!w.defined()) {
        w = vesper::full(normalized_shape, input.dtype(), input.device(), 1.0f);
    }
    if (!b.defined()) {
        b = vesper::full(normalized_shape, input.dtype(), input.device(), 0.0f);
    }
    
    return ops::layer_norm(input, normalized_shape, w, b, eps);
}

Tensor rms_norm(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                const Tensor& weight, float eps) {
    Tensor w = weight;
    if (!w.defined()) {
        w = vesper::full(normalized_shape, input.dtype(), input.device(), 1.0f);
    }
    return ops::rms_norm(input, normalized_shape, w, eps);
}

Tensor scaled_dot_product_attention(const Tensor& query, 
                                    const Tensor& key, 
                                    const Tensor& value, 
                                    bool is_causal, 
                                    double dropout_p) {
    // query: [Batch, Heads, SeqLen, HeadDim]
    // key:   [Batch, Heads, SeqLen, HeadDim]
    // value: [Batch, Heads, SeqLen, HeadDim]
    
    int64_t d_k = query.shape().back();
    
    // 1. Scores = Q @ K^T
    // K^T: [Batch, Heads, HeadDim, SeqLen]
    // We transpose the last two dimensions of K.
    Tensor key_t = key.transpose(-2, -1);
    Tensor scores = ops::matmul(query, key_t); // [Batch, Heads, SeqLen, SeqLen]
    
    // 2. Scale
    scores = ops::div(scores, std::sqrt(static_cast<float>(d_k)));
    
    // 3. Masking
    if (is_causal) {
        int64_t S = query.shape()[2]; // SeqLen
        
        // Create mask on CPU
        // We want a mask where future positions are -inf.
        // Since we add the mask, we want 0 for keep, -inf for mask.
        std::vector<float> mask_data(S * S);
        float neg_inf = -std::numeric_limits<float>::infinity();
        
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j) {
                if (j > i) {
                    mask_data[i * S + j] = neg_inf;
                } else {
                    mask_data[i * S + j] = 0.0f;
                }
            }
        }
        
        Tensor mask = vesper::empty({S, S}, DType::Float32, Device::CPU);
        mask.copy_from_host(mask_data.data());
        mask = mask.to(query.device());
        
        // Broadcast add: [Batch, Heads, S, S] + [S, S]
        scores = ops::add(scores, mask);
    }
    
    // 4. Softmax
    Tensor probs = ops::softmax(scores, -1);
    
    // 5. Dropout (Placeholder)
    if (dropout_p > 0.0) {
        probs = dropout(probs, dropout_p, true);
    }
    
    // 6. Output = probs @ V
    Tensor output = ops::matmul(probs, value); // [Batch, Heads, SeqLen, HeadDim]
    
    return output;
}

Tensor compute_rope_frequencies(int seq_len, int head_dim, int start_pos, float theta, Device device) {
    int dim = head_dim;
    int half_dim = dim / 2;
    
    std::vector<float> freqs_data(seq_len * half_dim);
    
    for (int m = 0; m < seq_len; ++m) {
        int pos = start_pos + m;
        for (int i = 0; i < half_dim; ++i) {
            float freq = std::pow(theta, -2.0f * i / dim);
            freqs_data[m * half_dim + i] = pos * freq;
        }
    }
    
    Tensor freqs = vesper::empty({seq_len, half_dim}, DType::Float32, Device::CPU);
    freqs.copy_from_host(freqs_data.data());
    return freqs.to(device);
}

Tensor apply_rotary_emb(const Tensor& x, const Tensor& freqs) {
    // x: [B, H, S, D]
    // freqs: [S, D/2]
    
    auto shape = x.shape();
    int B = shape[0];
    int H = shape[1];
    int S = shape[2];
    int D = shape[3];
    
    // Reshape to separate pairs: [B, H, S, D/2, 2]
    Tensor x_pairs = x.view({B, H, S, D/2, 2});
    
    // Split into real (even) and imag (odd) parts
    // We use slicing. 
    // Slice(start, end) is exclusive of end.
    std::vector<int64_t> half_shape = {B, H, S, D/2};
    Tensor x_r = x_pairs.index({Slice(), Slice(), Slice(), Slice(), Slice(0, 1)}).view(half_shape);
    Tensor x_i = x_pairs.index({Slice(), Slice(), Slice(), Slice(), Slice(1, 2)}).view(half_shape);
    
    // Compute cos and sin of frequencies
    Tensor cos_freq = ops::cos(freqs);
    Tensor sin_freq = ops::sin(freqs);
    
    // Apply rotation
    // x_new_r = x_r * cos - x_i * sin
    // x_new_i = x_r * sin + x_i * cos
    // Broadcasting handles [B, H, S, D/2] vs [S, D/2]
    
    Tensor term1 = ops::mul(x_r, cos_freq);
    Tensor term2 = ops::mul(x_i, sin_freq);
    Tensor out_r = ops::sub(term1, term2);
    
    Tensor term3 = ops::mul(x_r, sin_freq);
    Tensor term4 = ops::mul(x_i, cos_freq);
    Tensor out_i = ops::add(term3, term4);
    
    // Stack and reshape back
    // stack currently only supports dim=0.
    // We stack at dim 0 -> [2, B, H, S, D/2]
    Tensor stacked = ops::stack({out_r, out_i}, 0);
    
    // Permute to [B, H, S, D/2, 2]
    // Original dims: 0 (2), 1 (B), 2 (H), 3 (S), 4 (D/2)
    // Target: B, H, S, D/2, 2 -> 1, 2, 3, 4, 0
    Tensor permuted = stacked.permute({1, 2, 3, 4, 0});
    
    // We need contiguous tensor to view as [B, H, S, D]
    Tensor out = permuted.contiguous().view({B, H, S, D});
    
    return out;
}

} // namespace vesper::nn::functional
