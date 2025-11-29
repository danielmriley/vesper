#include <vesper/optim/schedulers.h>
#include <vesper/optim/sgd.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

class MockOptimizer : public vesper::optim::Optimizer {
public:
    MockOptimizer() : Optimizer({}) {} // No params
    void step() override {}
    void set_lr(float lr) override { lr_ = lr; }
    float get_lr() const override { return lr_; }
    float lr_ = 0.1f;
};

void test_linear_lr() {
    std::cout << "Testing LinearLR..." << std::endl;
    MockOptimizer opt;
    opt.set_lr(0.1f); // Base LR
    
    // Start factor 0.5 -> 0.05
    // End factor 1.0 -> 0.1
    // Total iters 5
    vesper::optim::LinearLR scheduler(opt, 0.5f, 1.0f, 5);
    
    // Epoch 0: 0.05
    scheduler.step();
    std::cout << "Epoch 0 LR: " << opt.get_lr() << std::endl;
    assert(std::abs(opt.get_lr() - 0.05f) < 1e-5);
    
    // Epoch 5: 0.1
    for(int i=0; i<5; ++i) scheduler.step();
    std::cout << "Epoch 5 LR: " << opt.get_lr() << std::endl;
    assert(std::abs(opt.get_lr() - 0.1f) < 1e-5);
    
    std::cout << "LinearLR passed!" << std::endl;
}

void test_step_lr() {
    std::cout << "Testing StepLR..." << std::endl;
    MockOptimizer opt;
    opt.set_lr(0.1f);
    
    // Step size 2, Gamma 0.1
    vesper::optim::StepLR scheduler(opt, 2, 0.1f);
    
    // Epoch 0: 0.1 (step 0)
    scheduler.step();
    assert(std::abs(opt.get_lr() - 0.1f) < 1e-5);
    
    // Epoch 1: 0.1
    scheduler.step();
    assert(std::abs(opt.get_lr() - 0.1f) < 1e-5);
    
    // Epoch 2: 0.01 (step 1)
    scheduler.step();
    std::cout << "Epoch 2 LR: " << opt.get_lr() << std::endl;
    assert(std::abs(opt.get_lr() - 0.01f) < 1e-5);
    
    std::cout << "StepLR passed!" << std::endl;
}

void test_cosine_lr() {
    std::cout << "Testing CosineAnnealingLR..." << std::endl;
    MockOptimizer opt;
    opt.set_lr(0.1f); // Base LR
    
    // T_max 10, eta_min 0.0
    // epoch 0: 0.1
    // epoch 10: 0.0
    vesper::optim::CosineAnnealingLR scheduler(opt, 10, 0.0f);
    
    // Step 0 (epoch 0 -> 1)
    scheduler.step(); 
    // epoch 1. 
    // lr = 0.05 * (1 + cos(pi * 1 / 10)) = 0.05 * (1 + cos(0.314))
    
    for(int i=0; i<9; ++i) scheduler.step();
    // epoch 10. lr should be 0.0
    
    std::cout << "Epoch 10 LR: " << opt.get_lr() << std::endl;
    assert(std::abs(opt.get_lr() - 0.0f) < 1e-5);
    
    std::cout << "CosineAnnealingLR passed!" << std::endl;
}

void test_cosine_warmup_lr() {
    std::cout << "Testing CosineAnnealingWithWarmup..." << std::endl;
    MockOptimizer opt;
    opt.set_lr(0.001f); // Base LR (max)
    
    // T_max 100, warmup 10, eta_min 0.0001
    vesper::optim::CosineAnnealingWithWarmup scheduler(opt, 100, 10, 0.0001f);
    
    // During warmup: LR should increase linearly
    scheduler.step(); // epoch 0 -> 1
    float lr_at_1 = opt.get_lr();
    std::cout << "Epoch 1 LR (warmup): " << lr_at_1 << std::endl;
    assert(std::abs(lr_at_1 - 0.0001f) < 1e-6); // 1/10 of max
    
    // Step to epoch 5
    for (int i = 0; i < 4; ++i) scheduler.step();
    float lr_at_5 = opt.get_lr();
    std::cout << "Epoch 5 LR (warmup): " << lr_at_5 << std::endl;
    assert(std::abs(lr_at_5 - 0.0005f) < 1e-6); // 5/10 of max
    
    // Step to epoch 10 (end of warmup, start of cosine)
    for (int i = 0; i < 5; ++i) scheduler.step();
    float lr_at_10 = opt.get_lr();
    std::cout << "Epoch 10 LR (peak): " << lr_at_10 << std::endl;
    assert(std::abs(lr_at_10 - 0.001f) < 1e-6); // Should be at max
    
    // Step to epoch 100 (should be at eta_min)
    for (int i = 0; i < 90; ++i) scheduler.step();
    float lr_at_100 = opt.get_lr();
    std::cout << "Epoch 100 LR (min): " << lr_at_100 << std::endl;
    assert(std::abs(lr_at_100 - 0.0001f) < 1e-6); // Should be at min
    
    std::cout << "CosineAnnealingWithWarmup passed!" << std::endl;
}

int main() {
    test_linear_lr();
    test_step_lr();
    test_cosine_lr();
    test_cosine_warmup_lr();
    return 0;
}
