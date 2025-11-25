#include <vesper/ops/embedding.h>
#include <vesper/core/factories.h>
#include <vesper/core/stream.h>
#include <iostream>
#include <cstring>
#include <cmath>

namespace vesper::ops {

Tensor embedding(const Tensor& input, const Tensor& weight, int64_t padding_idx, bool scale_grad_by_freq, bool sparse, float max_norm, float norm_type) {
    // Checks
    if (weight.shape().size() != 2) {
        throw std::runtime_error("embedding weight must be 2D");
    }
    // Input can be any shape. Output will be [*input.shape, embedding_dim]
    std::vector<int64_t> out_shape = input.shape();
    out_shape.push_back(weight.shape()[1]);
    
    Tensor out = empty(out_shape, weight.dtype(), weight.device(), input.requires_grad() || weight.requires_grad()); // Actually depends on weight.requires_grad primarily
    // Input (indices) usually doesn't require grad.
    // If weight requires grad, output requires grad.
    
    if (input.device() == Device::CPU) {
        embedding_cpu_dispatch(input, weight, padding_idx, max_norm, norm_type, out);
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        embedding_cuda_dispatch(input, weight, padding_idx, max_norm, norm_type, out);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        embedding_hip_dispatch(input, weight, padding_idx, max_norm, norm_type, out);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    }

    if (weight.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        node->next_edges.push_back({weight.grad_node});
        
        node->backward_fn = [input, weight_copy=weight, padding_idx, scale_grad_by_freq, out_weak=out.weak()]() mutable {
            auto out_ptr = out_weak.lock();
            if (!out_ptr) return;
            
            if (weight_copy.requires_grad()) {
                // Backward: Scatter Add
                // grad_weight = zeros_like(weight)
                // for i in input: grad_weight[i] += grad_output[...]
                
                Tensor grad_weight = zeros(weight_copy.shape(), weight_copy.dtype(), weight_copy.device());
                Tensor grad_output = out_ptr->grad();
                
                if (grad_output.device() == Device::CPU) {
                    embedding_backward_cpu_dispatch(grad_output, input, weight_copy.shape()[0], padding_idx, scale_grad_by_freq, grad_weight);
                } else if (grad_output.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
                    embedding_backward_cuda_dispatch(grad_output, input, weight_copy.shape()[0], padding_idx, scale_grad_by_freq, grad_weight);
#endif
                } else if (grad_output.device() == Device::HIP) {
#if USE_HIP_BACKEND
                    embedding_backward_hip_dispatch(grad_output, input, weight_copy.shape()[0], padding_idx, scale_grad_by_freq, grad_weight);
#endif
                }
                
                weight_copy.accumulate_grad(grad_weight);
            }
        };
        out.grad_node = node;
    }
    
    return out;
}

// --- CPU Implementation ---

void embedding_cpu_dispatch(const Tensor& input, const Tensor& weight, int64_t padding_idx, float max_norm, float norm_type, Tensor& out) {
    // input: indices (Int32 or Int64)
    // weight: [NumEmbed, Dim]
    // out: [*InputShape, Dim]
    
    int64_t num_indices = input.numel();
    int64_t embedding_dim = weight.shape()[1];
    int64_t num_embeddings = weight.shape()[0];
    
    // Assumes contiguous for simplicity (or implement strided)
    // Input and Out might be strided. Weight likely contiguous.
    // For MVP CPU, assume everything contiguous or use iterators if robust.
    // Or ensure contiguous.
    // We'll assume input/out contiguous logic or linear indexing.
    
    // Support Float32 weights only for now
    // We use const_cast because max_norm requires in-place modification of weight.
    float* w_ptr = const_cast<float*>(weight.data_ptr<float>());
    float* out_ptr = out.data_ptr<float>();
    
    // Indices can be int32 or int64 (long) or float (if casted).
    // We should check dtype.
    // For simplicity, let's assume Int32 or Int64.
    // Or use a template helper.
    
    auto impl = [&](auto* indices) {
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = static_cast<int64_t>(indices[i]);
            
            if (idx < 0 || idx >= num_embeddings) {
                // In real pytorch, this throws.
                // Or wraps if negative? usually throws index out of bounds.
                // We'll throw.
                throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx));
            }
            
            float* src_row = w_ptr + idx * embedding_dim;

            // Max Norm Renormalization
            if (max_norm > 0.0f) {
                 float norm = 0.0f;
                 for (int64_t d = 0; d < embedding_dim; ++d) {
                     norm += std::pow(std::abs(src_row[d]), norm_type);
                 }
                 norm = std::pow(norm, 1.0f / norm_type);
                 if (norm > max_norm) {
                     float scale = max_norm / (norm + 1e-7f);
                     for (int64_t d = 0; d < embedding_dim; ++d) {
                         src_row[d] *= scale;
                     }
                 }
            }

            float* dst_row = out_ptr + i * embedding_dim;
            
            if (idx == padding_idx) {
                std::memset(dst_row, 0, embedding_dim * sizeof(float));
            } else {
                std::memcpy(dst_row, src_row, embedding_dim * sizeof(float));
            }
        }
    };

    if (input.dtype() == DType::Int32) {
        impl(input.data_ptr<int32_t>());
    } else if (input.dtype() == DType::Float32) {
        // Cast float to int (dangerous but support for now if user passes float tensor)
        impl(input.data_ptr<float>());
    } else {
        // Assume Int64 if we support it, or throw.
        // Vesper DType::Int32 is common.
        // We don't have DType::Int64 exposed in factories.h/dtype.h explicitly?
        // Check dtype.h. If not there, assume Int32.
        throw std::runtime_error("Embedding indices must be Int32 or Float32 (casted).");
    }
}

void embedding_backward_cpu_dispatch(const Tensor& grad_output, const Tensor& input, int64_t num_embeddings, int64_t padding_idx, bool scale_grad_by_freq, Tensor& grad_weight) {
    // grad_output: [*InputShape, Dim]
    // input: [*InputShape]
    // grad_weight: [NumEmbed, Dim] (Dense accumulator)
    
    int64_t num_indices = input.numel();
    int64_t embedding_dim = grad_weight.shape()[1];
    
    float* gw_ptr = grad_weight.data_ptr<float>();
    const float* go_ptr = grad_output.data_ptr<float>();
    
    auto impl = [&](auto* indices) {
        std::vector<int64_t> counts;
        if (scale_grad_by_freq) {
            counts.resize(num_embeddings, 0);
            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t idx = static_cast<int64_t>(indices[i]);
                if (idx >= 0 && idx < num_embeddings && idx != padding_idx) {
                    counts[idx]++;
                }
            }
        }

        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = static_cast<int64_t>(indices[i]);
            if (idx == padding_idx) continue;
            if (idx < 0 || idx >= num_embeddings) continue; // Error or ignore? CPU forward threw error.
            
            // Scatter add
            float* dst_row = gw_ptr + idx * embedding_dim;
            const float* src_row = go_ptr + i * embedding_dim;
            
            float scale = 1.0f;
            if (scale_grad_by_freq && counts[idx] > 0) {
                scale = 1.0f / static_cast<float>(counts[idx]);
            }

            for (int64_t d = 0; d < embedding_dim; ++d) {
                dst_row[d] += src_row[d] * scale;
            }
        }
    };
    
    if (input.dtype() == DType::Int32) {
        impl(input.data_ptr<int32_t>());
    } else {
        impl(input.data_ptr<float>());
    }
}

} // namespace vesper::ops
