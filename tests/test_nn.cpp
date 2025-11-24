#include <vesper/nn/module.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>

// A dummy sub-module
class SubModule : public vesper::nn::Module {
public:
    SubModule() 
        : bias(vesper::zeros({8}, vesper::DType::Float32, vesper::Device::CPU))
    {
        register_parameter("bias", bias);
    }
    vesper::Tensor bias;
};

// A dummy top-level module
class TestModule : public vesper::nn::Module {
public:
    TestModule() 
        : weights(vesper::full({16, 8}, vesper::DType::Float32, vesper::Device::CPU, 1.0f))
    {
        register_parameter("weights", weights);

        // Create and register a sub-module
        sub = std::make_shared<SubModule>();
        register_module("sub", sub);
    }
    vesper::Tensor weights;
    std::shared_ptr<SubModule> sub;
};

void test_module_structure() {
    std::cout << "Testing nn::Module structure..." << std::endl;

    auto model = std::make_shared<TestModule>();

    // 1. Verify parameter collection
    auto params = model->parameters();
    assert(params.size() == 2); // Should find model.weights and model.sub.bias

    // 2. Verify zero_grad
    // Get a parameter and manually set its gradient to something non-zero
    vesper::Tensor bias_param = model->sub->parameters()[0];
    assert(bias_param.requires_grad());
    
    // Manually set a non-zero gradient
    bias_param.grad() = vesper::full(bias_param.shape(), bias_param.dtype(), bias_param.device(), 1.0f);
    
    // Verify grad is not zero
    std::vector<float> grad_vec(bias_param.grad().numel());
    bias_param.grad().copy_to_host(grad_vec.data());
    assert(grad_vec[0] == 1.0f);

    // Now, zero all gradients
    model->zero_grad();

    // Verify grad is now zero
    bias_param.grad().copy_to_host(grad_vec.data());
    assert(grad_vec[0] == 0.0f);

    std::cout << "nn::Module structure test passed!" << std::endl;
}


int main() {
    test_module_structure();
    return 0;
}
