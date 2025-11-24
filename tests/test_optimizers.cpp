#include <vesper/optim/adam.h>
#include <vesper/optim/lion.h>
#include <vesper/core/factories.h>
#include <vesper/nn/linear.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#if defined(USE_HIP_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CUDA;
#elif defined(USE_CPU_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CPU;
#else
    #error "No backend enabled for testing"
#endif

// Simple quadratic problem: min (x - target)^2
// Gradient: 2 * (x - target)
// Optimal x = target.

void test_adam_convergence() {
    std::cout << "Testing Adam Convergence..." << std::endl;
    
    float target_val = 5.0f;
    auto x = vesper::full({1}, vesper::DType::Float32, TEST_DEVICE, 0.0f, true);
    auto target = vesper::full({1}, vesper::DType::Float32, TEST_DEVICE, target_val, false);
    
    vesper::optim::Adam optimizer({x}, 0.1f); // lr = 0.1
    
    for (int i = 0; i < 100; ++i) {
        optimizer.zero_grad();
        
        auto diff = x - target;
        auto loss = diff * diff; // (x-5)^2
        
        loss.backward();
        optimizer.step();
    }
    
    float val = x.item<float>();
    std::cout << "Adam Final x: " << val << " (Target: 5.0)" << std::endl;
    assert(std::abs(val - target_val) < 0.1f);
    std::cout << "Adam Passed!" << std::endl;
}

void test_lion_convergence() {
    std::cout << "Testing Lion Convergence..." << std::endl;
    
    float target_val = -3.0f;
    auto x = vesper::full({1}, vesper::DType::Float32, TEST_DEVICE, 0.0f, true);
    auto target = vesper::full({1}, vesper::DType::Float32, TEST_DEVICE, target_val, false);
    
    // Lion usually needs smaller LR than Adam for similar problems or different tuning.
    vesper::optim::Lion optimizer({x}, 0.05f); 
    
    for (int i = 0; i < 100; ++i) {
        optimizer.zero_grad();
        
        auto diff = x - target;
        auto loss = diff * diff; 
        
        loss.backward();
        optimizer.step();
    }
    
    float val = x.item<float>();
    std::cout << "Lion Final x: " << val << " (Target: -3.0)" << std::endl;
    assert(std::abs(val - target_val) < 0.1f);
    std::cout << "Lion Passed!" << std::endl;
}

int main() {
    test_adam_convergence();
    test_lion_convergence();
    return 0;
}
