/**
 * @file swiglu.cpp
 * @brief Implementation of SwiGLU and other GLU-variant MLPs
 * 
 * Chapter 33.4: SwiGLU and FFN Variants
 */

#include <vesper/nn/swiglu.h>
#include <vesper/nn/functional.h>
#include <vesper/ops/elementwise.h>
#include <stdexcept>
#include <cmath>

namespace vesper::nn {

// ============================================================================
// SwiGLUMLP Implementation
// ============================================================================

SwiGLUMLP::SwiGLUMLP(int64_t d_model, int64_t hidden_dim, bool bias)
    : d_model_(d_model), hidden_dim_(hidden_dim),
      gate_proj_(d_model, hidden_dim, bias),
      up_proj_(d_model, hidden_dim, bias),
      down_proj_(hidden_dim, d_model, bias)
{
    if (d_model <= 0 || hidden_dim <= 0) {
        throw std::invalid_argument("SwiGLUMLP: d_model and hidden_dim must be positive");
    }
    
    // Register submodules using pointer to member variables
    register_module("gate_proj", &gate_proj_);
    register_module("up_proj", &up_proj_);
    register_module("down_proj", &down_proj_);
}

Tensor SwiGLUMLP::forward(const Tensor& x) {
    // x: [Batch, SeqLen, D] or [Batch, D]
    
    // 1. Gate and Up projections
    Tensor gate = gate_proj_.forward(x);   // [B, S, Hidden] or [B, Hidden]
    Tensor up = up_proj_.forward(x);       // [B, S, Hidden] or [B, Hidden]
    
    // 2. Fused SiLU * multiply: silu(gate) * up in single kernel
    Tensor hidden = ops::silu_mul(gate, up);
    
    // 3. Down projection
    return down_proj_.forward(hidden);     // [B, S, D] or [B, D]
}

int64_t SwiGLUMLP::compute_hidden_dim(int64_t d_model, int64_t multiple_of) {
    // For parameter parity with 4x MLP (2 * d * 4d = 8d^2):
    // GLU has 3 matrices: 3 * d * h = 8d^2 => h = 8d/3
    int64_t hidden = (8 * d_model) / 3;
    
    // Round up to nearest multiple for hardware efficiency
    if (multiple_of > 1) {
        hidden = ((hidden + multiple_of - 1) / multiple_of) * multiple_of;
    }
    
    return hidden;
}

// ============================================================================
// SwiGLUMLPFused Implementation
// ============================================================================

SwiGLUMLPFused::SwiGLUMLPFused(int64_t d_model, int64_t hidden_dim, bool bias)
    : d_model_(d_model), hidden_dim_(hidden_dim),
      gate_up_proj_(d_model, 2 * hidden_dim, bias),
      down_proj_(hidden_dim, d_model, bias)
{
    if (d_model <= 0 || hidden_dim <= 0) {
        throw std::invalid_argument("SwiGLUMLPFused: d_model and hidden_dim must be positive");
    }
    
    register_module("gate_up_proj", &gate_up_proj_);
    register_module("down_proj", &down_proj_);
}

Tensor SwiGLUMLPFused::forward(const Tensor& x) {
    // x: [Batch, SeqLen, D] or [Batch, D]
    
    // 1. Fused gate+up projection
    Tensor gate_up = gate_up_proj_.forward(x);  // [B, S, 2*Hidden] or [B, 2*Hidden]
    
    // 2. Split along last dimension using narrow (more efficient than index)
    int64_t last_dim = gate_up.ndim() - 1;
    int64_t half = hidden_dim_;  // We know hidden_dim_ = shape[last_dim] / 2
    
    // narrow(dim, start, length) - splits without building selectors
    Tensor gate = gate_up.narrow(last_dim, 0, half).contiguous();     // [B, S, Hidden]
    Tensor up = gate_up.narrow(last_dim, half, half).contiguous();    // [B, S, Hidden]
    
    // 3. Fused SiLU * multiply: silu(gate) * up in single kernel
    Tensor hidden = ops::silu_mul(gate, up);
    
    // 4. Down projection
    return down_proj_.forward(hidden);
}

// ============================================================================
// GeGLUMLP Implementation (GELU-gated)
// ============================================================================

GeGLUMLP::GeGLUMLP(int64_t d_model, int64_t hidden_dim, bool bias)
    : d_model_(d_model), hidden_dim_(hidden_dim),
      gate_proj_(d_model, hidden_dim, bias),
      up_proj_(d_model, hidden_dim, bias),
      down_proj_(hidden_dim, d_model, bias)
{
    if (d_model <= 0 || hidden_dim <= 0) {
        throw std::invalid_argument("GeGLUMLP: d_model and hidden_dim must be positive");
    }
    
    register_module("gate_proj", &gate_proj_);
    register_module("up_proj", &up_proj_);
    register_module("down_proj", &down_proj_);
}

Tensor GeGLUMLP::forward(const Tensor& x) {
    // GeGLU: GELU(gate) * up
    Tensor gate = gate_proj_.forward(x);
    Tensor up = up_proj_.forward(x);
    
    // GELU activation on gate
    gate = functional::gelu(gate);
    
    // Gating
    Tensor hidden = ops::mul(gate, up);
    
    return down_proj_.forward(hidden);
}

// ============================================================================
// ReGLUMLP Implementation (ReLU-gated)
// ============================================================================

ReGLUMLP::ReGLUMLP(int64_t d_model, int64_t hidden_dim, bool bias)
    : d_model_(d_model), hidden_dim_(hidden_dim),
      gate_proj_(d_model, hidden_dim, bias),
      up_proj_(d_model, hidden_dim, bias),
      down_proj_(hidden_dim, d_model, bias)
{
    if (d_model <= 0 || hidden_dim <= 0) {
        throw std::invalid_argument("ReGLUMLP: d_model and hidden_dim must be positive");
    }
    
    register_module("gate_proj", &gate_proj_);
    register_module("up_proj", &up_proj_);
    register_module("down_proj", &down_proj_);
}

Tensor ReGLUMLP::forward(const Tensor& x) {
    // ReGLU: ReLU(gate) * up
    Tensor gate = gate_proj_.forward(x);
    Tensor up = up_proj_.forward(x);
    
    // ReLU activation on gate
    gate = functional::relu(gate);
    
    // Gating
    Tensor hidden = ops::mul(gate, up);
    
    return down_proj_.forward(hidden);
}

} // namespace vesper::nn
