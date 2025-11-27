#include <vesper/optim/sgd.h>
#include <vesper/nn/module.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>
#include <cmath>

// A dummy module with one parameter
class SimpleModel : public vesper::nn::Module {
public:
    SimpleModel() : weight(vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 1.0f, true)) {
        register_parameter("weight", &weight);  // Pass pointer
    }
    vesper::Tensor weight;
};

void test_sgd_step() {
    std::cout << "Testing SGD step..." << std::endl;

    // 1. Setup
    auto model = SimpleModel();
    float lr = 0.1f;
    auto optimizer = vesper::optim::SGD(model.parameters(), lr);

    // 2. Manually set gradient
    auto param = model.parameters()[0];
    
    param.grad() = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 0.5f);

    // 3. Step
    optimizer.step();

    // 4. Verify
    std::vector<float> data(1);
    param.copy_to_host(data.data());
    
    std::cout << "Weight after step: " << data[0] << std::endl;
    
    assert(std::abs(data[0] - 0.95f) < 1e-6);

    std::cout << "SGD step test passed!" << std::endl;
}

void test_sgd_multi_step() {
    std::cout << "Testing SGD multi-step..." << std::endl;
    
    auto model = SimpleModel();
    float lr = 0.1f;
    auto optimizer = vesper::optim::SGD(model.parameters(), lr);
    auto param = model.parameters()[0];

    // Step 1
    param.grad() = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 1.0f);
    optimizer.step();
    // w = 1.0 - 0.1 * 1.0 = 0.9
    
    // Step 2
    optimizer.zero_grad(); 
    
    param.grad() = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 0.5f);
    optimizer.step();
    // w = 0.9 - 0.1 * 0.5 = 0.85
    
    std::vector<float> data(1);
    param.copy_to_host(data.data());
    std::cout << "Weight after 2 steps: " << data[0] << std::endl;
    assert(std::abs(data[0] - 0.85f) < 1e-6);
    
    std::cout << "SGD multi-step test passed!" << std::endl;
}

int main() {
    test_sgd_step();
    test_sgd_multi_step();
    return 0;
}
