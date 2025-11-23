#include <vesper/nn/conv2d.h>
#include <vesper/core/factories.h>
#include <vesper/ops/random.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

#if defined(USE_HIP_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CUDA;
#elif defined(USE_CPU_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CPU;
#else
    #error "No backend enabled for testing"
#endif

void test_conv2d_forward_shape() {
    std::cout << "Testing Conv2d forward shape..." << std::endl;
    
    // Input: [B, C, H, W] = [2, 3, 5, 5]
    auto input = vesper::empty({2, 3, 5, 5}, vesper::DType::Float32, TEST_DEVICE);
    
    // Conv: In=3, Out=4, K=3, S=1, P=0
    auto conv = vesper::nn::Conv2d(3, 4, 3, 1, 0, true, TEST_DEVICE);
    
    // Expected Out H = (5 + 0 - 3)/1 + 1 = 3
    // Out shape: [2, 4, 3, 3]
    
    auto output = conv.forward(input);
    
    assert(output.shape() == std::vector<int64_t>({2, 4, 3, 3}));
    
    std::cout << "Conv2d forward shape passed!" << std::endl;
}

void test_conv2d_forward_val() {
    std::cout << "Testing Conv2d forward values (simple)..." << std::endl;
    
    // Input: 1x1x3x3 all ones
    auto input = vesper::full({1, 1, 3, 3}, vesper::DType::Float32, TEST_DEVICE, 1.0f);
    
    // Conv: 1->1, K=3, S=1, P=0. Weight all ones. Bias zero.
    auto conv = vesper::nn::Conv2d(1, 1, 3, 1, 0, false, TEST_DEVICE);
    vesper::ops::uniform_(conv.weight, 1.0f, 1.0f); // Set weights to 1
    
    // Output should be 1x1x1x1. Value = sum(input) = 9.
    auto output = conv.forward(input);
    
    std::vector<float> res(1);
    output.copy_to_host(res.data());
    
    assert(std::fabs(res[0] - 9.0f) < 1e-5);
    
    std::cout << "Conv2d forward value passed!" << std::endl;
}

int main() {
    try {
        test_conv2d_forward_shape();
        test_conv2d_forward_val();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
