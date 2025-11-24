#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/comparison.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/normalization.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/random.h>
#include <vesper/ops/cast.h>
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

        result.grad_node->backward_fn = [input_nc, result]() mutable {
            // Sigmoid gradient: grad_output * (output * (1 - output))
            // We need `ones` factory.
            // Since `ones` is not in factories.h (I checked earlier), I use `full` with 1.0f.
            auto ones = vesper::full(result.shape(), result.dtype(), result.device(), 1.0f);
            auto term2 = ops::sub(ones, result);
            auto local_grad = ops::mul(result, term2);
            auto final_grad = ops::mul(result.grad(), local_grad);
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

        result.grad_node->backward_fn = [input_nc, result]() mutable {
            // ReLU gradient: grad_input = grad_output * (input > 0)
            // 1. Create the mask from the original input
            auto mask = ops::greater_than(input_nc, 0.0f);

            // 2. Multiply the upstream gradient by the mask
            auto final_grad = ops::mul(result.grad(), mask);
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

Tensor gelu(const Tensor& input) {
    return ops::gelu(input);
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
        float neg_inf = -1e9f; // Use a large negative number instead of -inf to avoid NaNs in some cases
        
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

} // namespace vesper::nn::functional
