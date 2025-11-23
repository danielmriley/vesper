#include <vesper/nn/pooling.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
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

void test_max_pool_forward() {
    std::cout << "Testing MaxPool2d forward..." << std::endl;
    
    // 1x1x4x4
    auto input = vesper::empty({1, 1, 4, 4}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    input.copy_from_host(h_data.data());
    input.set_requires_grad(true);
    
    auto pool = vesper::nn::MaxPool2d(2, 2); // Kernel 2, Stride 2
    
    // Expected output 2x2:
    // max(1,2,5,6)=6, max(3,4,7,8)=8
    // max(9,10,13,14)=14, max(11,12,15,16)=16
    
    auto output = pool.forward(input);
    
    std::vector<float> res(4);
    output.copy_to_host(res.data());
    
    assert(std::fabs(res[0] - 6.0f) < 1e-5);
    assert(std::fabs(res[1] - 8.0f) < 1e-5);
    assert(std::fabs(res[2] - 14.0f) < 1e-5);
    assert(std::fabs(res[3] - 16.0f) < 1e-5);
    
    std::cout << "MaxPool2d forward passed!" << std::endl;
    
    // Backward check
    // Loss = sum(output)
    // dL/dOutput = 1
    // dL/dInput should be 1 at max positions, 0 elsewhere.
    
    // Dummy backward (manually calling backward since we don't have sum reduction fully hooked up for all cases? 
    // No, we verified sum in prev tests).
    
    // Let's define a loss
    // We need to sum the output.
    // We have ops::sum.
    // But ops::sum currently returns a scalar tensor.
    
    // Since `ops::sum` logic was fixed, let's use it.
    // Wait, does `ops::sum` support reducing 4D tensor to scalar? Yes, `reduce_kernel` handles `n` elements.
    
    // However, we don't have `ops::sum` exposed in `include/vesper/ops/reduction.h` for public usage?
    // Yes we do: `Tensor sum(const Tensor& input);`
    
    // But `ops::sum` implementation in `reduction.cpp`:
    // `sum_hip_dispatch` supports n-blocks.
    // `backward_fn` uses `full` to broadcast scalar grad.
    // Should work.
    
    // We need to include `vesper/ops/reduction.h`
}

void test_max_pool_backward() {
    std::cout << "Testing MaxPool2d backward..." << std::endl;
    // Re-setup same test
    auto input = vesper::empty({1, 1, 4, 4}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    input.copy_from_host(h_data.data());
    input.set_requires_grad(true);
    
    auto pool = vesper::nn::MaxPool2d(2, 2);
    auto output = pool.forward(input);
    
    // Create a dummy gradient for output: all 1.0
    // Manually triggering backward on output with grad 1.0
    output.backward(); 
    
    std::vector<float> grad_in(16);
    input.grad().copy_to_host(grad_in.data());
    
    // Expected: 1.0 at indices 5, 7, 13, 15. 0.0 elsewhere.
    // 5 -> 6
    // 7 -> 8
    // 13 -> 14
    // 15 -> 16
    
    for(int i=0; i<16; ++i) {
        if (i == 5 || i == 7 || i == 13 || i == 15) {
            assert(std::fabs(grad_in[i] - 1.0f) < 1e-5);
        } else {
            assert(std::fabs(grad_in[i] - 0.0f) < 1e-5);
        }
    }
    std::cout << "MaxPool2d backward passed!" << std::endl;
}

int main() {
    try {
        test_max_pool_forward();
        test_max_pool_backward();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
