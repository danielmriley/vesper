#include <vesper/nn/linear.h>
#include <vesper/core/factories.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <cassert>
#include <vector>

void test_linear_gpu() {
#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)
    std::cout << "Testing Linear layer on GPU..." << std::endl;
    
#if defined(USE_HIP_BACKEND)
    vesper::Device device = vesper::Device::HIP;
#else
    vesper::Device device = vesper::Device::CUDA;
#endif

    int64_t in_features = 10;
    int64_t out_features = 5;
    
    // Initialize on GPU
    vesper::nn::Linear layer(in_features, out_features, true, device);
    
    // Check device of parameters
    assert(layer.weight.device() == device);
    assert(layer.bias.device() == device);
    
    // Create input on GPU
    auto input = vesper::full({2, in_features}, vesper::DType::Float32, device, 1.0f, true);
    
    // Forward
    auto output = layer.forward(input);
    assert(output.device() == device);
    assert(output.shape()[0] == 2);
    assert(output.shape()[1] == out_features);
    
    // Backward
    auto loss = vesper::ops::sum(output);
    loss.backward();
    
    // Check gradients exist and are on GPU
    assert(layer.weight.grad().device() == device);
    assert(layer.bias.grad().device() == device);
    
    std::cout << "Linear layer on GPU passed!" << std::endl;
#else
    std::cout << "Skipping GPU test." << std::endl;
#endif
}

int main() {
    test_linear_gpu();
    return 0;
}
