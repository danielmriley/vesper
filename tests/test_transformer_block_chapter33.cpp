/**
 * @file test_transformer_block_chapter33.cpp
 * @brief Chapter 33.1 Comprehensive Tests for Transformer Block
 * 
 * Tests from Chapter 33.1 testing strategy matrix:
 * - Parameter count verification for multiple configurations
 * - Forward pass no NaN/Inf verification
 * - Gradient finite verification
 * - Gradient magnitude analysis
 * - Overfitting convergence tests
 * - Stacked blocks tests
 * - Full model sketch test
 */

#include <vesper/nn/transformer.h>
#include <vesper/nn/embedding.h>
#include <vesper/nn/functional.h>
#include <vesper/nn/loss.h>
#include <vesper/core/factories.h>
#include <vesper/ops/random.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/elementwise.h>
#include <vesper/optim/sgd.h>
#include <vesper/optim/adam.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <chrono>

using namespace vesper;

// ============================================================================
// Utilities
// ============================================================================

bool contains_nan_or_inf(const Tensor& t) {
    std::vector<float> data(t.numel());
    t.contiguous().copy_to_host(data.data());
    for (float v : data) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
}

size_t count_parameters(nn::Module& module) {
    size_t total = 0;
    for (const auto& p : module.parameters()) {
        total += p.numel();
    }
    return total;
}

float max_abs_diff(const Tensor& t1, const Tensor& t2) {
    std::vector<float> d1(t1.numel()), d2(t2.numel());
    t1.contiguous().copy_to_host(d1.data());
    t2.contiguous().copy_to_host(d2.data());
    float max_diff = 0.0f;
    for (size_t i = 0; i < d1.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(d1[i] - d2[i]));
    }
    return max_diff;
}

// ============================================================================
// Parameter Count Tests
// ============================================================================

void test_param_count_formula() {
    std::cout << "Testing parameter count formula..." << std::endl;
    
    // Formula: 12E² + 13E
    // c_attn: E × 3E + 3E = 3E² + 3E
    // c_proj (attn): E × E + E = E² + E
    // c_fc: E × 4E + 4E = 4E² + 4E
    // c_proj (mlp): 4E × E + E = 4E² + E
    // ln1: 2E (weight + bias)
    // ln2: 2E (weight + bias)
    // Total: 12E² + 13E
    
    auto compute_expected = [](int E) -> size_t {
        return 12 * E * E + 13 * E;
    };
    
    // Test various configurations
    struct Config {
        int E, H;
        const char* name;
    };
    
    std::vector<Config> configs = {
        {64, 4, "Tiny"},
        {128, 4, "Small"},
        {256, 8, "Medium"},
        {512, 8, "Large"},
        {768, 12, "GPT-2 Small"},
        {1024, 16, "GPT-2 Medium"},
    };
    
    for (const auto& cfg : configs) {
        nn::TransformerBlock block(cfg.E, cfg.H);
        size_t expected = compute_expected(cfg.E);
        size_t actual = count_parameters(block);
        
        std::cout << "  " << cfg.name << " (E=" << cfg.E << "): expected=" 
                  << expected << ", actual=" << actual << std::endl;
        
        assert(actual == expected && "Parameter count mismatch");
    }
    
    std::cout << "Parameter count formula passed!" << std::endl;
}

void test_param_count_gpt2_medium() {
    std::cout << "Testing GPT-2 Medium parameter count..." << std::endl;
    
    int E = 1024, H = 16;
    nn::TransformerBlock block(E, H);
    
    // 12 × 1024² + 13 × 1024 = 12,582,912 + 13,312 = 12,596,224
    size_t expected = 12 * 1024 * 1024 + 13 * 1024;
    size_t actual = count_parameters(block);
    
    std::cout << "  Expected: " << expected << ", Actual: " << actual << std::endl;
    assert(actual == expected);
    
    std::cout << "GPT-2 Medium parameter count passed!" << std::endl;
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

void test_forward_no_nan_inf() {
    std::cout << "Testing forward pass produces no NaN/Inf..." << std::endl;
    
    int E = 64, H = 4;
    nn::TransformerBlock block(E, H);
    
    // Test with various input ranges
    struct TestCase {
        float mean, std;
        const char* name;
    };
    
    std::vector<TestCase> cases = {
        {0.0f, 0.1f, "Small std"},
        {0.0f, 1.0f, "Unit std"},
        {0.0f, 10.0f, "Large std"},
        {100.0f, 1.0f, "Large mean"},
        {-100.0f, 1.0f, "Negative mean"},
    };
    
    for (const auto& tc : cases) {
        Tensor x = empty({2, 16, E}, DType::Float32, Device::CPU);
        ops::normal_(x, tc.mean, tc.std);
        
        Tensor out = block.forward(x, true);
        
        assert(!contains_nan_or_inf(out) && "Forward produced NaN/Inf");
        std::cout << "  " << tc.name << ": passed" << std::endl;
    }
    
    std::cout << "Forward no NaN/Inf passed!" << std::endl;
}

void test_forward_output_range() {
    std::cout << "Testing forward output is in reasonable range..." << std::endl;
    
    int E = 64, H = 4;
    nn::TransformerBlock block(E, H);
    
    Tensor x = empty({4, 32, E}, DType::Float32, Device::CPU);
    ops::normal_(x, 0.0f, 1.0f);
    
    Tensor out = block.forward(x, true);
    
    std::vector<float> out_data(out.numel());
    out.copy_to_host(out_data.data());
    
    float min_val = *std::min_element(out_data.begin(), out_data.end());
    float max_val = *std::max_element(out_data.begin(), out_data.end());
    float mean = 0.0f;
    for (float v : out_data) mean += v;
    mean /= out_data.size();
    
    std::cout << "  Output range: [" << min_val << ", " << max_val << "], mean=" << mean << std::endl;
    
    // Output should be roughly in a reasonable range (not exploding)
    assert(std::abs(min_val) < 1000.0f && "Output exploded (min)");
    assert(std::abs(max_val) < 1000.0f && "Output exploded (max)");
    
    std::cout << "Forward output range passed!" << std::endl;
}

// ============================================================================
// Gradient Tests
// ============================================================================

void test_gradient_finite() {
    std::cout << "Testing all gradients are finite..." << std::endl;
    
    int E = 32, H = 4;
    nn::TransformerBlock block(E, H);
    
    Tensor x = empty({2, 8, E}, DType::Float32, Device::CPU);
    ops::normal_(x, 0.0f, 0.5f);
    x.set_requires_grad(true);
    
    Tensor out = block.forward(x, true);
    Tensor loss = ops::sum(out);
    loss.backward();
    
    // Check input gradient
    assert(x.grad().defined() && !contains_nan_or_inf(x.grad()) && "Input grad NaN/Inf");
    
    // Check all parameter gradients
    for (const auto& [name, param] : block.named_parameters()) {
        assert(param.grad().defined() && "Gradient not defined for " + name);
        assert(!contains_nan_or_inf(param.grad()) && "NaN/Inf in gradient for " + name);
    }
    
    std::cout << "All gradients finite passed!" << std::endl;
}

void test_gradient_magnitude() {
    std::cout << "Testing gradient magnitude is reasonable..." << std::endl;
    
    int E = 64, H = 4;
    nn::TransformerBlock block(E, H);
    
    Tensor x = empty({2, 16, E}, DType::Float32, Device::CPU);
    ops::normal_(x, 0.0f, 0.5f);
    x.set_requires_grad(true);
    
    Tensor out = block.forward(x, true);
    Tensor loss = ops::mean(out);  // Use mean for normalized gradient
    loss.backward();
    
    std::cout << "  Gradient statistics:" << std::endl;
    
    // Get parameters (non-const)
    auto params = block.parameters();
    auto named = block.named_parameters();
    
    for (auto& [name, param_const] : named) {
        // Need to get a non-const version to call grad()
        Tensor param = param_const;  // Copy to non-const
        
        std::vector<float> grad_data(param.grad().numel());
        param.grad().copy_to_host(grad_data.data());
        
        float grad_min = *std::min_element(grad_data.begin(), grad_data.end());
        float grad_max = *std::max_element(grad_data.begin(), grad_data.end());
        float grad_mean = 0.0f;
        float grad_sq_sum = 0.0f;
        for (float g : grad_data) {
            grad_mean += g;
            grad_sq_sum += g * g;
        }
        grad_mean /= grad_data.size();
        float grad_rms = std::sqrt(grad_sq_sum / grad_data.size());
        
        // Gradient magnitude should be reasonable (not vanishing or exploding)
        assert(grad_rms < 100.0f && "Gradient exploding");
        
        // At least some gradients should be non-zero
        bool has_nonzero = false;
        for (float g : grad_data) {
            if (std::abs(g) > 1e-10f) {
                has_nonzero = true;
                break;
            }
        }
        assert(has_nonzero && "All gradients are zero");
        
        std::cout << "    " << name << ": range=[" << grad_min << ", " << grad_max 
                  << "], rms=" << grad_rms << std::endl;
    }
    
    std::cout << "Gradient magnitude passed!" << std::endl;
}

// ============================================================================
// Overfitting Tests
// ============================================================================

void test_overfit_convergence() {
    std::cout << "Testing overfitting convergence on single sample..." << std::endl;
    
    int E = 32, H = 4;
    nn::TransformerBlock block(E, H, 0.0f);  // No dropout for overfitting
    
    // Single sample to overfit
    Tensor x = empty({1, 4, E}, DType::Float32, Device::CPU);
    ops::normal_(x, 0.0f, 0.5f);
    
    // Target: simple transformation of input
    Tensor target = ops::mul(x, 0.5f);  // Target is half of input
    
    optim::Adam optimizer(block.parameters(), 0.01f);
    
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    
    int num_steps = 200;
    for (int step = 0; step < num_steps; ++step) {
        optimizer.zero_grad();
        
        Tensor out = block.forward(x, false);
        
        // MSE loss
        Tensor diff = ops::sub(out, target);
        Tensor sq = ops::mul(diff, diff);
        Tensor loss = ops::mean(sq);
        
        std::vector<float> loss_val(1);
        loss.copy_to_host(loss_val.data());
        
        if (step == 0) initial_loss = loss_val[0];
        if (step == num_steps - 1) final_loss = loss_val[0];
        
        if (step % 50 == 0 || step == num_steps - 1) {
            std::cout << "  Step " << step << ": loss=" << loss_val[0] << std::endl;
        }
        
        loss.backward();
        optimizer.step();
    }
    
    std::cout << "  Initial loss: " << initial_loss << ", Final loss: " << final_loss << std::endl;
    
    // Loss should decrease significantly
    assert(final_loss < initial_loss * 0.1f && "Failed to overfit (loss didn't decrease enough)");
    assert(final_loss < 0.1f && "Failed to converge to low loss");
    
    std::cout << "Overfitting convergence passed!" << std::endl;
}

// ============================================================================
// Stacked Blocks Tests
// ============================================================================

void test_stacked_blocks_forward() {
    std::cout << "Testing stacked transformer blocks..." << std::endl;
    
    int E = 64, H = 4;
    int num_layers = 4;
    
    std::vector<nn::TransformerBlock> blocks;
    blocks.reserve(num_layers);  // Pre-reserve to avoid reallocation issues with pointer-based registration
    for (int i = 0; i < num_layers; ++i) {
        blocks.emplace_back(E, H, 0.0f);
    }
    
    Tensor x = empty({2, 16, E}, DType::Float32, Device::CPU);
    ops::normal_(x, 0.0f, 0.5f);
    
    Tensor h = x;
    for (int i = 0; i < num_layers; ++i) {
        h = blocks[i].forward(h, true);
        
        assert(!contains_nan_or_inf(h) && "NaN/Inf in layer output");
        assert(h.shape()[0] == 2 && h.shape()[1] == 16 && h.shape()[2] == E);
    }
    
    std::cout << "  " << num_layers << " stacked blocks: output valid" << std::endl;
    std::cout << "Stacked blocks forward passed!" << std::endl;
}

void test_stacked_blocks_gradient_flow() {
    std::cout << "Testing gradient flow through stacked blocks..." << std::endl;
    
    int E = 32, H = 4;
    int num_layers = 6;  // Deep stack
    
    std::vector<nn::TransformerBlock> blocks;
    blocks.reserve(num_layers);  // Pre-reserve to avoid reallocation issues with pointer-based registration
    for (int i = 0; i < num_layers; ++i) {
        blocks.emplace_back(E, H, 0.0f);
    }
    
    Tensor x = empty({1, 8, E}, DType::Float32, Device::CPU);
    ops::normal_(x, 0.0f, 0.5f);
    x.set_requires_grad(true);
    
    Tensor h = x;
    for (auto& block : blocks) {
        h = block.forward(h, true);
    }
    
    Tensor loss = ops::sum(h);
    loss.backward();
    
    // Check input gradient exists and is non-zero
    assert(x.grad().defined() && !contains_nan_or_inf(x.grad()));
    
    std::vector<float> grad_data(x.grad().numel());
    x.grad().copy_to_host(grad_data.data());
    
    float grad_norm = 0.0f;
    for (float g : grad_data) grad_norm += g * g;
    grad_norm = std::sqrt(grad_norm);
    
    std::cout << "  Input gradient norm through " << num_layers << " layers: " << grad_norm << std::endl;
    
    // Gradient should not vanish
    assert(grad_norm > 1e-6f && "Gradient vanished through deep stack");
    
    // Check gradients in first and last block
    std::vector<float> first_grad(blocks[0].attn.c_attn.weight.grad().numel());
    std::vector<float> last_grad(blocks[num_layers-1].attn.c_attn.weight.grad().numel());
    
    blocks[0].attn.c_attn.weight.grad().copy_to_host(first_grad.data());
    blocks[num_layers-1].attn.c_attn.weight.grad().copy_to_host(last_grad.data());
    
    float first_norm = 0.0f, last_norm = 0.0f;
    for (float g : first_grad) first_norm += g * g;
    for (float g : last_grad) last_norm += g * g;
    first_norm = std::sqrt(first_norm);
    last_norm = std::sqrt(last_norm);
    
    std::cout << "  First block c_attn gradient norm: " << first_norm << std::endl;
    std::cout << "  Last block c_attn gradient norm: " << last_norm << std::endl;
    
    // Gradient shouldn't vanish even in first block
    assert(first_norm > 1e-6f && "Gradient vanished in first block");
    
    std::cout << "Stacked blocks gradient flow passed!" << std::endl;
}

// ============================================================================
// MLP Component Tests
// ============================================================================

void test_mlp_4x_expansion() {
    std::cout << "Testing MLP 4x expansion..." << std::endl;
    
    int E = 64;
    nn::MLP mlp(E, 0.0f);
    
    // c_fc should be E -> 4E
    assert(mlp.c_fc.weight.shape()[0] == 4 * E);
    assert(mlp.c_fc.weight.shape()[1] == E);
    
    // c_proj should be 4E -> E
    assert(mlp.c_proj.weight.shape()[0] == E);
    assert(mlp.c_proj.weight.shape()[1] == 4 * E);
    
    // Forward should preserve shape
    Tensor x = empty({2, 8, E}, DType::Float32, Device::CPU);
    ops::normal_(x, 0.0f, 0.5f);
    
    Tensor out = mlp.forward(x);
    
    assert(out.shape()[0] == 2);
    assert(out.shape()[1] == 8);
    assert(out.shape()[2] == E);
    
    std::cout << "MLP 4x expansion passed!" << std::endl;
}

void test_mlp_gelu_activation() {
    std::cout << "Testing MLP uses GELU activation..." << std::endl;
    
    int E = 4;
    nn::MLP mlp(E, 0.0f);
    
    // Set weights to identity-like for debugging
    // c_fc: just pass through scaled
    std::vector<float> fc_w(E * 4 * E, 0.0f);
    for (int i = 0; i < E; ++i) {
        fc_w[i * E + i] = 1.0f;  // First E outputs are identity
    }
    mlp.c_fc.weight.copy_from_host(fc_w.data());
    
    std::vector<float> fc_b(4 * E, 0.0f);
    mlp.c_fc.bias.copy_from_host(fc_b.data());
    
    // c_proj: sum the first E values
    std::vector<float> proj_w(E * 4 * E, 0.0f);
    for (int i = 0; i < E; ++i) {
        proj_w[i * 4 * E + i] = 1.0f;
    }
    mlp.c_proj.weight.copy_from_host(proj_w.data());
    
    std::vector<float> proj_b(E, 0.0f);
    mlp.c_proj.bias.copy_from_host(proj_b.data());
    
    // Test input
    Tensor x = empty({1, 1, E}, DType::Float32, Device::CPU);
    std::vector<float> x_data = {-2.0f, -1.0f, 0.0f, 1.0f};
    x.copy_from_host(x_data.data());
    
    Tensor out = mlp.forward(x);
    
    std::vector<float> out_data(E);
    out.copy_to_host(out_data.data());
    
    // GELU(-2) ≈ -0.0454, GELU(-1) ≈ -0.159, GELU(0) = 0, GELU(1) ≈ 0.841
    // These should be approximately what we see
    std::cout << "  Input: [" << x_data[0] << ", " << x_data[1] << ", " 
              << x_data[2] << ", " << x_data[3] << "]" << std::endl;
    std::cout << "  Output: [" << out_data[0] << ", " << out_data[1] << ", " 
              << out_data[2] << ", " << out_data[3] << "]" << std::endl;
    
    // GELU(x) < x for x < ~0.5, and GELU(x) ≈ x for large positive x
    // Just verify the pattern
    assert(out_data[0] < 0.0f && out_data[0] > -0.5f);  // GELU(-2) small negative
    assert(out_data[2] < 1e-5f && out_data[2] > -1e-5f);  // GELU(0) ≈ 0
    assert(out_data[3] > 0.5f && out_data[3] < 1.0f);  // GELU(1) ≈ 0.84
    
    std::cout << "MLP GELU activation passed!" << std::endl;
}

// ============================================================================
// Backend Consistency
// ============================================================================

void test_transformer_block_cpu_vs_hip() {
    std::cout << "Testing TransformerBlock CPU vs HIP..." << std::endl;

#ifndef USE_HIP_BACKEND
    std::cout << "  HIP not available, skipping." << std::endl;
    return;
#else
    int E = 64, H = 4;
    nn::TransformerBlock block_cpu(E, H, 0.0f);
    
    Tensor x_cpu = empty({2, 16, E}, DType::Float32, Device::CPU);
    ops::uniform_(x_cpu, -0.5f, 0.5f);
    
    Tensor out_cpu = block_cpu.forward(x_cpu, true);
    
    // Create HIP block with same weights
    nn::TransformerBlock block_hip(E, H, 0.0f);
    
    // Copy weights
    block_hip.ln1.weight = block_cpu.ln1.weight.to(Device::HIP);
    block_hip.ln1.bias = block_cpu.ln1.bias.to(Device::HIP);
    block_hip.ln2.weight = block_cpu.ln2.weight.to(Device::HIP);
    block_hip.ln2.bias = block_cpu.ln2.bias.to(Device::HIP);
    block_hip.attn.c_attn.weight = block_cpu.attn.c_attn.weight.to(Device::HIP);
    block_hip.attn.c_attn.bias = block_cpu.attn.c_attn.bias.to(Device::HIP);
    block_hip.attn.c_proj.weight = block_cpu.attn.c_proj.weight.to(Device::HIP);
    block_hip.attn.c_proj.bias = block_cpu.attn.c_proj.bias.to(Device::HIP);
    block_hip.mlp.c_fc.weight = block_cpu.mlp.c_fc.weight.to(Device::HIP);
    block_hip.mlp.c_fc.bias = block_cpu.mlp.c_fc.bias.to(Device::HIP);
    block_hip.mlp.c_proj.weight = block_cpu.mlp.c_proj.weight.to(Device::HIP);
    block_hip.mlp.c_proj.bias = block_cpu.mlp.c_proj.bias.to(Device::HIP);
    
    Tensor x_hip = x_cpu.to(Device::HIP);
    Tensor out_hip = block_hip.forward(x_hip, true);
    
    float diff = max_abs_diff(out_cpu, out_hip.to(Device::CPU));
    std::cout << "  CPU vs HIP diff: " << diff << std::endl;
    
    assert(diff < 1e-3f && "CPU vs HIP mismatch");
    
    std::cout << "TransformerBlock CPU vs HIP passed!" << std::endl;
#endif
}

// ============================================================================
// Performance Test
// ============================================================================

void test_transformer_block_performance() {
    std::cout << "Testing TransformerBlock performance..." << std::endl;
    
    int E = 256, H = 8;
    int B = 4, S = 128;
    
    nn::TransformerBlock block(E, H, 0.0f);
    
    Tensor x = empty({B, S, E}, DType::Float32, Device::CPU);
    ops::normal_(x, 0.0f, 0.5f);
    
    // Warmup
    block.forward(x, true);
    
    int num_iters = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iters; ++i) {
        block.forward(x, true);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float ms_per_iter = duration.count() / 1000.0f / num_iters;
    std::cout << "  CPU (B=" << B << ", S=" << S << ", E=" << E << "): " 
              << ms_per_iter << " ms/iter" << std::endl;

#ifdef USE_HIP_BACKEND
    nn::TransformerBlock block_hip(E, H, 0.0f);
    
    // Move to HIP
    block_hip.ln1.weight = block.ln1.weight.to(Device::HIP);
    block_hip.ln1.bias = block.ln1.bias.to(Device::HIP);
    block_hip.ln2.weight = block.ln2.weight.to(Device::HIP);
    block_hip.ln2.bias = block.ln2.bias.to(Device::HIP);
    block_hip.attn.c_attn.weight = block.attn.c_attn.weight.to(Device::HIP);
    block_hip.attn.c_attn.bias = block.attn.c_attn.bias.to(Device::HIP);
    block_hip.attn.c_proj.weight = block.attn.c_proj.weight.to(Device::HIP);
    block_hip.attn.c_proj.bias = block.attn.c_proj.bias.to(Device::HIP);
    block_hip.mlp.c_fc.weight = block.mlp.c_fc.weight.to(Device::HIP);
    block_hip.mlp.c_fc.bias = block.mlp.c_fc.bias.to(Device::HIP);
    block_hip.mlp.c_proj.weight = block.mlp.c_proj.weight.to(Device::HIP);
    block_hip.mlp.c_proj.bias = block.mlp.c_proj.bias.to(Device::HIP);
    
    Tensor x_hip = x.to(Device::HIP);
    
    // Warmup
    block_hip.forward(x_hip, true);
    
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iters; ++i) {
        block_hip.forward(x_hip, true);
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    ms_per_iter = duration.count() / 1000.0f / num_iters;
    std::cout << "  HIP (B=" << B << ", S=" << S << ", E=" << E << "): " 
              << ms_per_iter << " ms/iter" << std::endl;
#endif
    
    std::cout << "TransformerBlock performance test done!" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Chapter 33.1 Transformer Block Tests ===" << std::endl;
    
    // Parameter count tests
    test_param_count_formula();
    test_param_count_gpt2_medium();
    
    // Forward pass tests
    test_forward_no_nan_inf();
    test_forward_output_range();
    
    // Gradient tests
    test_gradient_finite();
    test_gradient_magnitude();
    
    // Overfitting tests
    test_overfit_convergence();
    
    // Stacked blocks tests
    test_stacked_blocks_forward();
    test_stacked_blocks_gradient_flow();
    
    // MLP tests
    test_mlp_4x_expansion();
    test_mlp_gelu_activation();
    
    // Backend consistency
    test_transformer_block_cpu_vs_hip();
    
    // Performance
    test_transformer_block_performance();
    
    std::cout << "\n=== All Chapter 33.1 Tests Passed! ===" << std::endl;
    return 0;
}
