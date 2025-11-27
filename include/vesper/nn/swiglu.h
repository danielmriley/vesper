#pragma once
#include <vesper/nn/module.h>
#include <vesper/nn/linear.h>
#include <vesper/core/tensor.h>

namespace vesper::nn {

/**
 * @brief SwiGLU MLP - Gated Linear Unit with Swish activation
 * 
 * Used in Llama, Mistral, and other modern LLMs.
 * 
 * Architecture:
 *   output = W_down(SiLU(W_gate(x)) * W_up(x))
 * 
 * Where:
 *   - W_gate: [d_model, hidden_dim] - Gate projection
 *   - W_up:   [d_model, hidden_dim] - Up projection
 *   - W_down: [hidden_dim, d_model] - Down projection
 *   - SiLU(x) = x * sigmoid(x)
 * 
 * Parameter count: 3 * d_model * hidden_dim (no bias by default)
 * 
 * For equivalent parameter count to 4x MLP:
 *   hidden_dim = 8/3 * d_model ≈ 2.67 * d_model
 */
class SwiGLUMLP : public Module {
public:
    /**
     * @brief Construct SwiGLU MLP
     * @param d_model Input/output dimension
     * @param hidden_dim Hidden dimension (typically 8/3 * d_model)
     * @param bias Whether to use bias in linear layers (default: false for Llama-style)
     */
    SwiGLUMLP(int64_t d_model, int64_t hidden_dim, bool bias = false);
    
    Tensor forward(const Tensor& x) override;
    
    /**
     * @brief Calculate recommended hidden dimension for parameter parity with 4x MLP
     * @param d_model Model dimension
     * @param multiple_of Round to nearest multiple (for hardware efficiency)
     * @return Recommended hidden dimension
     */
    static int64_t compute_hidden_dim(int64_t d_model, int64_t multiple_of = 256);
    
    // Accessors
    int64_t d_model() const { return d_model_; }
    int64_t hidden_dim() const { return hidden_dim_; }
    
private:
    Linear gate_proj_;  // W_gate: [d_model, hidden_dim]
    Linear up_proj_;    // W_up:   [d_model, hidden_dim]
    Linear down_proj_;  // W_down: [hidden_dim, d_model]
    int64_t d_model_;
    int64_t hidden_dim_;
};

/**
 * @brief Fused SwiGLU MLP - Optimized version with combined gate+up projection
 * 
 * Combines gate and up projections into a single GEMM for better efficiency.
 * Output is equivalent to SwiGLUMLP.
 */
class SwiGLUMLPFused : public Module {
public:
    /**
     * @brief Construct Fused SwiGLU MLP
     * @param d_model Input/output dimension
     * @param hidden_dim Hidden dimension
     * @param bias Whether to use bias
     */
    SwiGLUMLPFused(int64_t d_model, int64_t hidden_dim, bool bias = false);
    
    Tensor forward(const Tensor& x) override;
    
    int64_t d_model() const { return d_model_; }
    int64_t hidden_dim() const { return hidden_dim_; }
    
private:
    Linear gate_up_proj_;  // Combined: [d_model, 2 * hidden_dim]
    Linear down_proj_;     // [hidden_dim, d_model]
    int64_t d_model_;
    int64_t hidden_dim_;
};

/**
 * @brief GeGLU MLP - GELU-gated variant (used in Gemma, some other models)
 */
class GeGLUMLP : public Module {
public:
    GeGLUMLP(int64_t d_model, int64_t hidden_dim, bool bias = false);
    Tensor forward(const Tensor& x) override;
    
private:
    Linear gate_proj_;
    Linear up_proj_;
    Linear down_proj_;
    int64_t d_model_;
    int64_t hidden_dim_;
};

/**
 * @brief ReGLU MLP - ReLU-gated variant (simpler, used in some research)
 */
class ReGLUMLP : public Module {
public:
    ReGLUMLP(int64_t d_model, int64_t hidden_dim, bool bias = false);
    Tensor forward(const Tensor& x) override;
    
private:
    Linear gate_proj_;
    Linear up_proj_;
    Linear down_proj_;
    int64_t d_model_;
    int64_t hidden_dim_;
};

} // namespace vesper::nn
