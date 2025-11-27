#include <vesper/optim/optimizer.h>
#include <vesper/nn/module.h> // To create a dummy module
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>

// 1. A dummy module with one parameter
class SimpleModel : public vesper::nn::Module {
public:
    SimpleModel() : weight(vesper::full({4}, vesper::DType::Float32, vesper::Device::CPU, 1.0f, true)) {
        register_parameter("weight", &weight);  // Pass pointer
    }
    vesper::Tensor weight;
};

// 2. A dummy optimizer to make the base class concrete
class DummyOptimizer : public vesper::optim::Optimizer {
public:
    using vesper::optim::Optimizer::Optimizer; // Inherit constructor (takes vector<Tensor>)
    
    // Provide a dummy implementation for the pure virtual function
    void step() override {
        // Do nothing
    }
    
    void set_lr(float lr) override { lr_ = lr; }
    float get_lr() const override { return lr_; }
    float lr_ = 0.1f;
};

void test_optimizer_base() {
    std::cout << "Testing Optimizer base..." << std::endl;

    // 3. Set up the model and optimizer
    auto model = SimpleModel();
    auto optimizer = DummyOptimizer(model.parameters());

    // Manually set a grad
    auto p0 = model.parameters()[0];
    if (p0.requires_grad()) {
        // Set grad to all ones
        p0.grad() = vesper::full(p0.shape(), p0.dtype(), p0.device(), 1.0f);
        
        // Verify grad is set
        std::vector<float> grad_vec(p0.numel());
        p0.grad().copy_to_host(grad_vec.data());
        assert(grad_vec[0] == 1.0f);
        
        // Zero grad
        optimizer.zero_grad();
        
        // Verify grad is zero
        p0.grad().copy_to_host(grad_vec.data());
        assert(grad_vec[0] == 0.0f);
    }
    
    std::cout << "Optimizer base passed!" << std::endl;
}


int main() {
    test_optimizer_base();
    return 0;
}
