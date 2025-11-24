#include <vesper/serialization.h>
#include <vesper/nn/linear.h>
#include <vesper/optim/adam.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstdio>

// Helper to compare tensor values
bool all_close(const vesper::Tensor& a, const vesper::Tensor& b, float tol = 1e-5) {
    if (a.numel() != b.numel()) return false;
    std::vector<float> va(a.numel()), vb(b.numel());
    a.copy_to_host(va.data());
    b.copy_to_host(vb.data());
    for (size_t i = 0; i < va.size(); ++i) {
        if (std::abs(va[i] - vb[i]) > tol) return false;
    }
    return true;
}

void test_shape_mismatch() {
    std::cout << "Testing shape mismatch..." << std::endl;
    auto model = vesper::nn::Linear(10, 5); // Weight: [5, 10], Bias: [5]
    
    vesper::StateDict bad_dict;
    // Create mismatching weight: [5, 5] (numel 25 vs 50)
    bad_dict["weight"] = vesper::full({5, 5}, vesper::DType::Float32, vesper::Device::CPU, 0.0f);
    bad_dict["bias"] = vesper::full({5}, vesper::DType::Float32, vesper::Device::CPU, 0.0f);
    
    bool caught = false;
    try {
        model.load_state_dict(bad_dict);
    } catch (const std::runtime_error& e) {
        // "copy_from: element count mismatch" expected
        std::cout << "Caught expected error: " << e.what() << std::endl;
        caught = true;
    }
    assert(caught);
    std::cout << "Shape mismatch passed!" << std::endl;
}

void test_extra_and_missing_keys() {
    std::cout << "Testing extra/missing keys..." << std::endl;
    auto model = vesper::nn::Linear(2, 2); 
    // weights init to random.
    
    // 1. Missing key (partial load)
    vesper::StateDict partial_dict;
    partial_dict["bias"] = vesper::full({2}, vesper::DType::Float32, vesper::Device::CPU, 100.0f);
    
    model.load_state_dict(partial_dict);
    
    // Check bias updated
    std::vector<float> b_data(2);
    model.bias.copy_to_host(b_data.data());
    assert(b_data[0] == 100.0f);
    
    // Check weight NOT updated (still random/kaiming)
    // We assume it's not 100.0f
    std::vector<float> w_data(4);
    model.weight.copy_to_host(w_data.data());
    assert(std::abs(w_data[0] - 100.0f) > 1.0f); // Unlikely to be 100
    
    // 2. Extra keys (should ignore)
    vesper::StateDict extra_dict;
    extra_dict["weight"] = vesper::full({2, 2}, vesper::DType::Float32, vesper::Device::CPU, 200.0f);
    extra_dict["unknown_layer.weight"] = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 0.0f);
    
    try {
        model.load_state_dict(extra_dict);
    } catch (...) {
        assert(false && "Should not throw on extra keys");
    }
    
    // Check weight updated
    model.weight.copy_to_host(w_data.data());
    assert(w_data[0] == 200.0f);
    
    std::cout << "Extra/Missing keys passed!" << std::endl;
}

void test_optimizer_resume_exact() {
    std::cout << "Testing exact optimizer resume..." << std::endl;
    
    // Problem: x^2. Grad 2x. Min at 0.
    // Start at 1.0
    
    // Case A: Run 20 steps continuous
    auto x_a = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 1.0f, true);
    auto opt_a = vesper::optim::Adam({x_a}, 0.1f);
    
    for (int i = 0; i < 20; ++i) {
        opt_a.zero_grad();
        auto loss = x_a * x_a;
        loss.backward();
        opt_a.step();
    }
    
    // Case B: Run 10 steps, save, load, run 10 steps
    auto x_b = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 1.0f, true);
    auto opt_b = vesper::optim::Adam({x_b}, 0.1f);
    
    for (int i = 0; i < 10; ++i) {
        opt_b.zero_grad();
        auto loss = x_b * x_b;
        loss.backward();
        opt_b.step();
    }
    
    // Save
    std::string ckpt = "opt_resume.vsp";
    vesper::save(opt_b.state_dict(), ckpt);
    
    // Load into new optimizer (simulating restart)
    // We reuse x_b as the parameter, but reset optimizer
    // Or better, create new optimizer instance binding to x_b
    auto opt_c = vesper::optim::Adam({x_b}, 0.1f); // Fresh state (t=0)
    opt_c.load_state_dict(vesper::load(ckpt));
    
    // Continue 10 steps
    for (int i = 0; i < 10; ++i) {
        opt_c.zero_grad();
        auto loss = x_b * x_b;
        loss.backward();
        opt_c.step();
    }
    
    // Compare x_a and x_b
    if (!all_close(x_a, x_b)) {
        std::vector<float> va(1), vb(1);
        x_a.copy_to_host(va.data());
        x_b.copy_to_host(vb.data());
        std::cerr << "Resume mismatch: Continuous " << va[0] << " vs Resumed " << vb[0] << std::endl;
        assert(false);
    }
    
    std::remove(ckpt.c_str());
    std::cout << "Optimizer resume exact passed!" << std::endl;
}

int main() {
    test_shape_mismatch();
    test_extra_and_missing_keys();
    test_optimizer_resume_exact();
    return 0;
}
