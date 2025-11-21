#include <vesper/optim/optimizer.h>
#include <vesper/nn/module.h> // To create a dummy module
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>

// 1. A dummy module with one parameter
class SimpleModel : public vesper::nn::Module {
public:
    SimpleModel() : weight(vesper::full({4}, vesper::DType::Float32, vesper::Device::CPU, 1.0f, true)) {
        register_parameter("weight", weight);
    }
    vesper::Tensor weight;
};

// 2. A dummy optimizer to make the base class concrete
class DummyOptimizer : public vesper::optim::Optimizer {
public:
    using vesper::optim::Optimizer::Optimizer; // Inherit constructor
    
    // Provide a dummy implementation for the pure virtual function
    void step() override {
        // Does nothing for this test
    }
};

void test_optimizer_base() {
    std::cout << "Testing Optimizer base..." << std::endl;

    // 3. Set up the model and optimizer
    auto model = SimpleModel();
    auto optimizer = DummyOptimizer(model.parameters());

    // 4. Manually give the parameter a gradient
    auto param = model.parameters()[0];
    param->grad() = vesper::full(param->shape(), param->dtype(), param->device(), 1.0f);
    
    // Check that the gradient is non-zero
    std::vector<float> grad_vec(param->numel());
    param->grad().copy_to_host(grad_vec.data());
    assert(grad_vec[0] == 1.0f);

    // 5. Call the optimizer's zero_grad method
    optimizer.zero_grad();

    // 6. Verify that the parameter's gradient is now zero
    param->grad().copy_to_host(grad_vec.data());
    assert(grad_vec[0] == 0.0f);

    std::cout << "Optimizer base test passed!" << std::endl;
}


int main() {
    test_optimizer_base();
    return 0;
}
