#include <vesper/nn/linear.h>
#include <vesper/core/factories.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <cassert>
#include <vector>

void test_linear_gpu() {
#if USE_HIP_BACKEND
    std::cout << "Testing Linear layer on HIP..." << std::endl;
    
    int64_t in_features = 10;
    int64_t out_features = 5;
    
    // Initialize on HIP
    vesper::nn::Linear layer(in_features, out_features, true, vesper::Device::HIP);
    
    // Check device of parameters
    assert(layer.weight.device() == vesper::Device::HIP);
    assert(layer.bias.device() == vesper::Device::HIP);
    
    // Create input on HIP
    auto input = vesper::full({2, in_features}, vesper::DType::Float32, vesper::Device::HIP, 1.0f, true);
    
    // Forward
    auto output = layer.forward(input);
    assert(output.device() == vesper::Device::HIP);
    assert(output.shape()[0] == 2);
    assert(output.shape()[1] == out_features);
    
    // Backward
    auto loss = vesper::ops::sum(output);
    loss.backward();
    
    // Check gradients exist and are on HIP
    assert(layer.weight.grad().device() == vesper::Device::HIP);
    assert(layer.bias.grad().device() == vesper::Device::HIP);
    
    std::cout << "Linear layer on HIP passed!" << std::endl;
#else
    std::cout << "Skipping HIP test." << std::endl;
#endif
}

int main() {
    test_linear_gpu();
    return 0;
}
