#include <vesper/nn/utils.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/random.h>
#include <vesper/core/factories.h>
#include <cmath>
#include <string>

namespace vesper::nn::utils {

float clip_grad_norm_(std::vector<Tensor>& parameters, float max_norm, float norm_type) {
    if (parameters.empty()) return 0.0f;
    
    // Currently only L2 norm is supported (most common case)
    // Other norm types would require abs/pow ops
    if (norm_type != 2.0f) {
        // Fallback to simple L2 implementation
        norm_type = 2.0f;
    }
    
    // Find device from first parameter with gradient
    Device device = Device::CPU;
    bool found_grad = false;
    for (auto& p : parameters) {
        if (p.grad().defined()) {
            device = p.grad().device();
            found_grad = true;
            break;
        }
    }
    
    if (!found_grad) return 0.0f;
    
    // Compute total L2 norm on GPU (single sync at end)
    // L2 norm: sqrt(sum of squared norms)
    Tensor total_norm_sq;
    bool first = true;
    
    for (auto& p : parameters) {
        if (p.grad().defined()) {
            Tensor g = p.grad();
            Tensor norm_sq = ops::sum(ops::mul(g, g));
            if (first) {
                total_norm_sq = norm_sq;
                first = false;
            } else {
                total_norm_sq = ops::add(total_norm_sq, norm_sq);
            }
        }
    }
    
    if (first) return 0.0f;  // No gradients found
    
    // Take sqrt for L2 norm
    Tensor total_norm_tensor = ops::sqrt(total_norm_sq);
    
    // Single GPU->CPU transfer to get the norm value
    float total_norm = total_norm_tensor.item<float>();
    
    // Clip gradients if needed (all operations on GPU)
    if (total_norm > max_norm) {
        float clip_coef = max_norm / (total_norm + 1e-6f);
        for (auto& p : parameters) {
            if (p.grad().defined()) {
                Tensor clipped = ops::mul(p.grad(), clip_coef);
                p.grad().copy_from(clipped);
            }
        }
    }
    
    return total_norm;
}

void gpt2_init_(Tensor& tensor, float std, int n_layers, bool is_residual_proj) {
    float actual_std = std;
    
    // Scale residual projections by 1/sqrt(2*n_layers) to prevent variance growth
    if (is_residual_proj && n_layers > 0) {
        actual_std = std / std::sqrt(2.0f * n_layers);
    }
    
    ops::normal_(tensor, 0.0f, actual_std);
}

void init_transformer_weights(Module& module, int n_layers, float base_std) {
    // Get all named parameters from the module - use pointers to modify in-place!
    auto named_params = module.named_parameters_ptrs();
    
    float residual_std = base_std / std::sqrt(2.0f * n_layers);
    
    for (auto& [name, param_ptr] : named_params) {
        // Skip if parameter is not defined
        if (!param_ptr || !param_ptr->defined()) continue;
        
        Tensor& param = *param_ptr;  // Get reference to actual parameter
        
        // Check parameter dimensionality
        auto& shape = param.shape();
        
        // LayerNorm parameters
        if (name.find("ln") != std::string::npos || 
            name.find("norm") != std::string::npos) {
            if (shape.size() == 1) {
                // Could be weight or bias
                if (name.find("weight") != std::string::npos) {
                    // LayerNorm weight -> 1.0
                    Tensor ones = vesper::ones(shape, param.dtype(), param.device());
                    param.copy_from(ones);
                } else {
                    // LayerNorm bias -> 0.0
                    Tensor zeros_t = vesper::zeros(shape, param.dtype(), param.device());
                    param.copy_from(zeros_t);
                }
                continue;
            }
        }
        
        // Bias parameters -> zero
        if (name.find("bias") != std::string::npos && shape.size() == 1) {
            Tensor zeros_t = vesper::zeros(shape, param.dtype(), param.device());
            param.copy_from(zeros_t);
            continue;
        }
        
        // Output projections (c_proj in attention/MLP, out_proj, down_proj) get scaled init
        // These are the residual path outputs
        if ((name.find("c_proj") != std::string::npos ||
             name.find("out_proj") != std::string::npos ||
             name.find("down_proj") != std::string::npos) && 
            name.find("weight") != std::string::npos) {
            ops::normal_(param, 0.0f, residual_std);
            continue;
        }
        
        // All other weight matrices get standard init
        if (shape.size() >= 2) {
            ops::normal_(param, 0.0f, base_std);
            continue;
        }
        
        // 1D parameters that aren't bias/layernorm (rare)
        ops::normal_(param, 0.0f, base_std);
    }
}

void tie_weights(nn::Embedding& embedding, nn::Linear& linear) {
    // Verify shapes are compatible
    // Embedding: [vocab_size, embed_dim]
    // Linear (lm_head): [embed_dim, vocab_size] with output [batch, vocab_size]
    // Linear stores weight as [out_features, in_features]
    
    auto& emb_shape = embedding.weight.shape();
    auto& lin_shape = linear.weight.shape();
    
    // emb_shape = [vocab_size, embed_dim]
    // lin_shape should be [vocab_size, embed_dim] for lm_head
    // (Linear weight is [out_features, in_features])
    
    if (emb_shape.size() != 2 || lin_shape.size() != 2) {
        throw std::runtime_error("tie_weights: embedding and linear weight must be 2D");
    }
    
    // For lm_head: out_features = vocab_size, in_features = embed_dim
    // So linear.weight shape is [vocab_size, embed_dim], same as embedding!
    if (emb_shape[0] != lin_shape[0] || emb_shape[1] != lin_shape[1]) {
        throw std::runtime_error(
            "tie_weights: shape mismatch. Embedding [" + 
            std::to_string(emb_shape[0]) + ", " + std::to_string(emb_shape[1]) + 
            "] vs Linear [" + std::to_string(lin_shape[0]) + ", " + 
            std::to_string(lin_shape[1]) + "]"
        );
    }
    
    // Make the linear weight reference the embedding weight's storage
    // This creates a view/alias - both point to same underlying data
    linear.weight = embedding.weight;
}

} // namespace vesper::nn::utils
