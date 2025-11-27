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
    
    // 2. SiLU activation on gate: silu(x) = x * sigmoid(x)
    gate = functional::silu(gate);
    
    // 3. Element-wise multiply (gating)
    Tensor hidden = ops::mul(gate, up);    // [B, S, Hidden] or [B, Hidden]
    
    // 4. Down projection
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
    
    // 2. Split along last dimension using index selectors
    // Get the shape and compute split points
    auto shape = gate_up.shape();
    int64_t last_dim = static_cast<int64_t>(shape.size()) - 1;
    int64_t half = shape[last_dim] / 2;
    
    // Build index selectors for each half
    // For gate: [..., 0:half]
    // For up: [..., half:2*half]
    std::vector<IndexSelector> gate_selectors, up_selectors;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        gate_selectors.push_back(Slice());  // Select all
        up_selectors.push_back(Slice());
    }
    gate_selectors.push_back(Slice(0, half));
    up_selectors.push_back(Slice(half, 2 * half));
    
    Tensor gate = gate_up.index(gate_selectors).contiguous();  // [B, S, Hidden]
    Tensor up = gate_up.index(up_selectors).contiguous();      // [B, S, Hidden]
    
    // 3. SiLU on gate, multiply with up
    gate = functional::silu(gate);
    Tensor hidden = ops::mul(gate, up);
    
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
