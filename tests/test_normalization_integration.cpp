/**
 * @file test_normalization_integration.cpp
 * @brief Integration tests for Chapter 30: Normalization and Stability
 * 
 * Tests:
 * - Deep transformer block stability (12 layers)
 * - Gradient flow through many layers (24 layers)
 * - Combined LayerNorm + Attention + GELU
 */

#include <vesper/nn/normalization.h>
#include <vesper/nn/functional.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/module_list.h>
#include <vesper/core/factories.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/random.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/gemm.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>

using namespace vesper;

// ============================================================================
// Test Utilities
// ============================================================================

bool contains_nan_or_inf(const Tensor& t) {
    std::vector<float> data(t.numel());
    t.contiguous().copy_to_host(data.data());
    for (float v : data) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
}

// ============================================================================
// Simplified Transformer Components for Testing
// ============================================================================

// Simple MLP: Linear -> GELU -> Linear
class SimpleMLP {
public:
    nn::Linear fc1, fc2;
    
    SimpleMLP(int hidden, int mlp_dim)
        : fc1(hidden, mlp_dim), fc2(mlp_dim, hidden) {}
    
    void to(Device device) {
        fc1.weight = fc1.weight.to(device);
        fc1.bias = fc1.bias.to(device);
        fc2.weight = fc2.weight.to(device);
        fc2.bias = fc2.bias.to(device);
    }
    
    Tensor forward(const Tensor& x) {
        Tensor h = fc1.forward(x);
        h = nn::functional::gelu(h);
        return fc2.forward(h);
    }
};

// Pre-norm block: LN -> Attention-like -> residual, LN -> MLP -> residual
class PreNormBlock {
public:
    nn::LayerNorm ln1, ln2;
    SimpleMLP mlp;
    nn::Linear proj;  // Simplified "attention" as linear projection
    
    PreNormBlock(int hidden)
        : ln1({hidden}), ln2({hidden}), mlp(hidden, hidden * 4), proj(hidden, hidden) {}
    
    void to(Device device) {
        ln1.weight = ln1.weight.to(device);
        ln1.bias = ln1.bias.to(device);
        ln2.weight = ln2.weight.to(device);
        ln2.bias = ln2.bias.to(device);
        mlp.to(device);
        proj.weight = proj.weight.to(device);
        proj.bias = proj.bias.to(device);
    }
    
    Tensor forward(const Tensor& x) {
        // Pre-norm attention
        Tensor h = ln1.forward(x);
        h = proj.forward(h);
        h = ops::add(x, h);
        
        // Pre-norm MLP
        Tensor h2 = ln2.forward(h);
        h2 = mlp.forward(h2);
        return ops::add(h, h2);
    }
};

// ============================================================================
// Integration Tests
// ============================================================================

void test_transformer_block_stability(Device device) {
    std::string dev_str = (device == Device::CPU) ? "CPU" : 
                          (device == Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing Transformer Block Stability (12 layers) on " << dev_str << "..." << std::endl;
    
    int hidden = 64;
    int num_layers = 12;
    int B = 2, S = 8;
    
    // Create 12-layer stack. PreNormBlock aggregates non-movable Modules, so it is
    // held behind unique_ptr to keep element addresses stable.
    std::vector<std::unique_ptr<PreNormBlock>> layers;
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(std::make_unique<PreNormBlock>(hidden));
        if (device != Device::CPU) {
            layers.back()->to(device);
        }
    }
    
    // Random input
    Tensor x = randn({B, S, hidden}, DType::Float32, device);
    
    // Forward through all layers
    Tensor out = x;
    for (int i = 0; i < num_layers; ++i) {
        out = layers[i]->forward(out);

        // Check no NaN/Inf at each layer
        assert(!contains_nan_or_inf(out) && "NaN/Inf detected in layer output");
    }
    
    // Check output stats are reasonable (not exploded)
    std::vector<float> out_data(B * S * hidden);
    out.contiguous().copy_to_host(out_data.data());
    
    float max_val = 0.0f;
    for (float v : out_data) {
        max_val = std::max(max_val, std::abs(v));
    }
    
    // After 12 layers, values should still be bounded (normalization keeps things stable)
    assert(max_val < 1000.0f && "Output values exploded after 12 layers");
    
    std::cout << "  Max output value after 12 layers: " << max_val << std::endl;
    std::cout << "Transformer Block Stability passed!" << std::endl;
}

void test_gradient_flow_deep(Device device) {
    std::string dev_str = (device == Device::CPU) ? "CPU" : 
                          (device == Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing Gradient Flow (24 layers) on " << dev_str << "..." << std::endl;
    
    int hidden = 32;
    int num_layers = 24;
    int B = 1, S = 4;
    
    // Create layers - using simple LN + Linear + GELU chain.
    // ModuleList stores each module behind unique_ptr (stable addresses, no copies).
    nn::ModuleList<nn::LayerNorm> norms;
    nn::ModuleList<nn::Linear> linears;

    for (int i = 0; i < num_layers; ++i) {
        norms.emplace_back(std::vector<int64_t>{hidden});
        linears.emplace_back(hidden, hidden);
        
        if (device != Device::CPU) {
            norms.back().weight = norms.back().weight.to(device);
            norms.back().bias = norms.back().bias.to(device);
            linears.back().weight = linears.back().weight.to(device);
            linears.back().bias = linears.back().bias.to(device);
        }
    }
    
    // Input with requires_grad
    Tensor x = randn({B, S, hidden}, DType::Float32, device);
    x.set_requires_grad(true);
    
    // Forward through all layers with residual connections
    Tensor out = x;
    for (int i = 0; i < num_layers; ++i) {
        Tensor h = norms[i].forward(out);
        h = linears[i].forward(h);
        h = nn::functional::gelu(h);
        out = ops::add(out, h);  // Residual
    }
    
    // Compute loss and backward
    Tensor loss = ops::mean(out);
    loss.backward();
    
    // Check gradients exist and are finite
    assert(x.grad().defined() && "Input gradient not computed");
    assert(!contains_nan_or_inf(x.grad()) && "Input gradient contains NaN/Inf");
    
    // Check gradient magnitude is reasonable (not vanished or exploded)
    std::vector<float> grad_data(B * S * hidden);
    x.grad().contiguous().copy_to_host(grad_data.data());
    
    float grad_norm = 0.0f;
    for (float g : grad_data) {
        grad_norm += g * g;
    }
    grad_norm = std::sqrt(grad_norm);
    
    std::cout << "  Gradient norm after 24 layers: " << grad_norm << std::endl;
    
    // Gradient should not be zero (vanishing) or huge (exploding)
    assert(grad_norm > 1e-10f && "Gradient vanished");
    assert(grad_norm < 1e10f && "Gradient exploded");
    
    std::cout << "Gradient Flow Deep passed!" << std::endl;
}

void test_layer_norm_gelu_composition(Device device) {
    std::string dev_str = (device == Device::CPU) ? "CPU" : 
                          (device == Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing LayerNorm + GELU Composition on " << dev_str << "..." << std::endl;
    
    int hidden = 64;
    
    auto ln = nn::LayerNorm({hidden});
    if (device != Device::CPU) {
        ln.weight = ln.weight.to(device);
        ln.bias = ln.bias.to(device);
    }
    
    // Test that LayerNorm output into GELU works correctly
    Tensor x = randn({4, 16, hidden}, DType::Float32, device);
    x.set_requires_grad(true);
    
    Tensor normalized = ln.forward(x);
    Tensor activated = nn::functional::gelu(normalized);
    
    // Check output is valid
    assert(!contains_nan_or_inf(activated));
    
    // Backward should work
    Tensor loss = ops::sum(activated);
    loss.backward();
    
    assert(x.grad().defined());
    assert(!contains_nan_or_inf(x.grad()));
    
    std::cout << "LayerNorm + GELU Composition passed!" << std::endl;
}

void test_rms_norm_gelu_composition(Device device) {
    std::string dev_str = (device == Device::CPU) ? "CPU" : 
                          (device == Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing RMSNorm + GELU Composition on " << dev_str << "..." << std::endl;
    
    int hidden = 64;
    
    auto rms = nn::RMSNorm({hidden});
    if (device != Device::CPU) {
        rms.weight = rms.weight.to(device);
    }
    
    Tensor x = randn({4, 16, hidden}, DType::Float32, device);
    x.set_requires_grad(true);
    
    Tensor normalized = rms.forward(x);
    Tensor activated = nn::functional::gelu(normalized);
    
    assert(!contains_nan_or_inf(activated));
    
    Tensor loss = ops::sum(activated);
    loss.backward();
    
    assert(x.grad().defined());
    assert(!contains_nan_or_inf(x.grad()));
    
    std::cout << "RMSNorm + GELU Composition passed!" << std::endl;
}

void test_softmax_in_attention_context(Device device) {
    std::string dev_str = (device == Device::CPU) ? "CPU" : 
                          (device == Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing Softmax in Attention Context on " << dev_str << "..." << std::endl;
    
    // Simulate attention scores: [Batch, Heads, SeqLen, SeqLen]
    int B = 2, H = 4, S = 32;
    
    Tensor scores = randn({B, H, S, S}, DType::Float32, device);
    scores.set_requires_grad(true);
    
    // Scale (typical attention scaling)
    float scale = 1.0f / std::sqrt(static_cast<float>(64));  // head_dim=64
    scores = ops::mul(scores, scale);
    
    // Softmax over last dimension
    Tensor probs = nn::functional::softmax(scores, -1);
    
    // Check softmax properties
    assert(!contains_nan_or_inf(probs));
    
    // Backward
    Tensor loss = ops::sum(probs);
    loss.backward();
    
    assert(scores.grad().defined());
    assert(!contains_nan_or_inf(scores.grad()));
    
    std::cout << "Softmax in Attention Context passed!" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Normalization Integration Tests (Chapter 30) ===" << std::endl;
    
    // CPU tests
    test_transformer_block_stability(Device::CPU);
    test_gradient_flow_deep(Device::CPU);
    test_layer_norm_gelu_composition(Device::CPU);
    test_rms_norm_gelu_composition(Device::CPU);
    test_softmax_in_attention_context(Device::CPU);
    
#ifdef USE_CUDA_BACKEND
    std::cout << "\n--- CUDA Tests ---" << std::endl;
    try {
        test_transformer_block_stability(Device::CUDA);
        test_gradient_flow_deep(Device::CUDA);
        test_layer_norm_gelu_composition(Device::CUDA);
        test_rms_norm_gelu_composition(Device::CUDA);
        test_softmax_in_attention_context(Device::CUDA);
    } catch (const std::exception& e) {
        std::cerr << "CUDA test failed: " << e.what() << std::endl;
        return 1;
    }
#endif

#ifdef USE_HIP_BACKEND
    std::cout << "\n--- HIP Tests ---" << std::endl;
    try {
        test_transformer_block_stability(Device::HIP);
        test_gradient_flow_deep(Device::HIP);
        test_layer_norm_gelu_composition(Device::HIP);
        test_rms_norm_gelu_composition(Device::HIP);
        test_softmax_in_attention_context(Device::HIP);
    } catch (const std::exception& e) {
        std::cerr << "HIP test failed: " << e.what() << std::endl;
        return 1;
    }
#endif
    
    std::cout << "\n=== All Normalization Integration Tests Passed! ===" << std::endl;
    return 0;
}
