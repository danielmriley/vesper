#include <vesper/nn/module.h>
#include <vesper/nn/linear.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>

// A dummy sub-module
class SubModule : public vesper::nn::Module {
public:
    SubModule() 
        : bias(vesper::zeros({8}, vesper::DType::Float32, vesper::Device::CPU))
    {
        register_parameter("bias", &bias);  // Pass pointer
    }
    vesper::Tensor bias;
};

// A dummy top-level module
class TestModule : public vesper::nn::Module {
public:
    TestModule() 
        : weights(vesper::full({16, 8}, vesper::DType::Float32, vesper::Device::CPU, 1.0f))
    {
        register_parameter("weights", &weights);  // Pass pointer
        register_module("sub", &sub);  // Pass pointer to member
    }
    vesper::Tensor weights;
    SubModule sub;  // Direct member, not shared_ptr
};

void test_module_structure() {
    std::cout << "Testing nn::Module structure..." << std::endl;

    TestModule model;

    // 1. Verify parameter collection
    auto params = model.parameters();
    assert(params.size() == 2); // Should find model.weights and model.sub.bias

    // 2. Verify zero_grad
    // Get a parameter and manually set its gradient to something non-zero
    vesper::Tensor bias_param = model.sub.parameters()[0];
    assert(bias_param.requires_grad());
    
    // Manually set a non-zero gradient
    bias_param.grad() = vesper::full(bias_param.shape(), bias_param.dtype(), bias_param.device(), 1.0f);
    
    // Verify grad is not zero
    std::vector<float> grad_vec(bias_param.grad().numel());
    bias_param.grad().copy_to_host(grad_vec.data());
    assert(grad_vec[0] == 1.0f);

    // Now, zero all gradients
    model.zero_grad();

    // Verify grad is now zero
    bias_param.grad().copy_to_host(grad_vec.data());
    assert(grad_vec[0] == 0.0f);

    std::cout << "nn::Module structure test passed!" << std::endl;
}

// Test that to(device) properly updates member variables used in forward()
void test_to_device_updates_members() {
    std::cout << "Testing to(device) updates member variables..." << std::endl;
    
    // Use Linear which has weight and bias as members
    vesper::nn::Linear linear(4, 8);
    
    // Verify initially on CPU
    assert(linear.weight.device() == vesper::Device::CPU);
    assert(linear.bias.device() == vesper::Device::CPU);
    
    // Create input on CPU
    auto x = vesper::randn({2, 4}, vesper::DType::Float32, vesper::Device::CPU);
    
    // Forward should work on CPU
    auto y_cpu = linear.forward(x);
    assert(y_cpu.device() == vesper::Device::CPU);
    
    // Move to HIP
    linear.to(vesper::Device::HIP);
    
    // CRITICAL: Member variables should now be on HIP
    assert(linear.weight.device() == vesper::Device::HIP && 
           "weight member should be on HIP after to(HIP)");
    assert(linear.bias.device() == vesper::Device::HIP && 
           "bias member should be on HIP after to(HIP)");
    
    // Forward should now work on HIP
    auto x_hip = x.to(vesper::Device::HIP);
    auto y_hip = linear.forward(x_hip);
    assert(y_hip.device() == vesper::Device::HIP);
    
    std::cout << "to(device) updates member variables test passed!" << std::endl;
}

// Test nested modules with to(device)
void test_nested_module_to_device() {
    std::cout << "Testing nested module to(device)..." << std::endl;
    
    TestModule model;
    
    // Initially on CPU
    assert(model.weights.device() == vesper::Device::CPU);
    assert(model.sub.bias.device() == vesper::Device::CPU);
    
    // Move entire model to HIP
    model.to(vesper::Device::HIP);
    
    // Both top-level and nested parameters should now be on HIP
    assert(model.weights.device() == vesper::Device::HIP && 
           "weights should be on HIP after to(HIP)");
    assert(model.sub.bias.device() == vesper::Device::HIP && 
           "sub.bias should be on HIP after to(HIP)");
    
    std::cout << "Nested module to(device) test passed!" << std::endl;
}


int main() {
    test_module_structure();
    test_to_device_updates_members();
    test_nested_module_to_device();
    return 0;
}
