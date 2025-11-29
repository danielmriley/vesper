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
    
    // IMPORTANT: Do not modify weights in-place during forward pass!
    // max_norm renormalization should be done on the output, not the weights.
    // This matches PyTorch's behavior for the output and avoids corrupting model weights.
    const float* w_ptr = weight.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    auto impl = [&](auto* indices) {
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = static_cast<int64_t>(indices[i]);
            
            if (idx < 0 || idx >= num_embeddings) {
                throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx));
            }
            
            const float* src_row = w_ptr + idx * embedding_dim;
            float* dst_row = out_ptr + i * embedding_dim;
            
            if (idx == padding_idx) {
                std::memset(dst_row, 0, embedding_dim * sizeof(float));
            } else {
                // Copy embedding to output
                std::memcpy(dst_row, src_row, embedding_dim * sizeof(float));
                
                // Apply max_norm renormalization to the OUTPUT (not the weights)
                // This ensures the forward pass doesn't have side effects on model parameters
                if (max_norm > 0.0f) {
                    float norm = 0.0f;
                    for (int64_t d = 0; d < embedding_dim; ++d) {
                        norm += std::pow(std::abs(dst_row[d]), norm_type);
                    }
                    norm = std::pow(norm, 1.0f / norm_type);
                    if (norm > max_norm) {
                        float scale = max_norm / (norm + 1e-7f);
                        for (int64_t d = 0; d < embedding_dim; ++d) {
                            dst_row[d] *= scale;
                        }
                    }
                }
            }
        }
    };

    if (input.dtype() == DType::Int32) {
        impl(input.data_ptr<int32_t>());
    } else if (input.dtype() == DType::Float32) {
        // Float indices are unsafe - they hide bugs in user code
        // Emit a warning but still support for backward compatibility
        static bool warned = false;
        if (!warned) {
            std::cerr << "Warning: Embedding received Float32 indices. "
                      << "This is unsafe and may be removed in future versions. "
                      << "Please use Int32 indices." << std::endl;
            warned = true;
        }
        impl(input.data_ptr<float>());
    } else {
        throw std::runtime_error("Embedding indices must be Int32. Received dtype that is not Int32 or Float32.");
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
