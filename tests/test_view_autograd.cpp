#include <vesper/core/factories.h>
#include <vesper/core/tensor.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

using namespace vesper;

#if defined(USE_HIP_BACKEND)
    constexpr Device TEST_DEVICE = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    constexpr Device TEST_DEVICE = Device::CUDA;
#elif defined(USE_CPU_BACKEND)
    constexpr Device TEST_DEVICE = Device::CPU;
#else
    #error "No backend enabled for testing"
#endif

void verify_tensor(const Tensor& t, const std::vector<float>& expected) {
    std::vector<float> data(t.numel());
    t.copy_to_host(data.data());
    for (size_t i = 0; i < data.size(); ++i) {
        if (std::fabs(data[i] - expected[i]) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": " << data[i] << " vs " << expected[i] << std::endl;
            assert(false);
        }
    }
}

void test_view_backward() {
    std::cout << "Testing view backward..." << std::endl;
    // x = [1, 2, 3, 4]
    // y = x.view(2, 2)
    // z = sum(y)
    // grad_x should be [1, 1, 1, 1]
    
    Tensor x = full({4}, DType::Float32, TEST_DEVICE, 1.0f, true);
    Tensor y = x.view({2, 2});
    Tensor z = ops::sum(y);
    
    z.backward();
    
    verify_tensor(x.grad(), {1, 1, 1, 1});
    std::cout << "View backward passed!" << std::endl;
}

void test_transpose_backward() {
    std::cout << "Testing transpose backward..." << std::endl;
    // x = [[1, 2], [3, 4]]
    // y = x.transpose(0, 1) -> [[1, 3], [2, 4]]
    // z = y * 2
    // loss = sum(z)
    // grad_y = [[2, 2], [2, 2]]
    // grad_x = grad_y.transpose(0, 1) = [[2, 2], [2, 2]]
    
    Tensor x = full({2, 2}, DType::Float32, TEST_DEVICE, 1.0f, true);
    Tensor y = x.transpose(0, 1);
    Tensor z = ops::mul(y, 2.0f);
    Tensor loss = ops::sum(z);
    
    loss.backward();
    
    verify_tensor(x.grad(), {2, 2, 2, 2});
    std::cout << "Transpose backward passed!" << std::endl;
}

void test_permute_backward() {
    std::cout << "Testing permute backward..." << std::endl;
    // x: [2, 3]
    // y = x.permute(1, 0) -> [3, 2]
    // z = y * 3
    // loss = sum(z)
    // grad_x should be 3 everywhere
    
    Tensor x = full({2, 3}, DType::Float32, TEST_DEVICE, 1.0f, true);
    Tensor y = x.permute({1, 0});
    Tensor z = ops::mul(y, 3.0f);
    Tensor loss = ops::sum(z);
    
    loss.backward();
    
    verify_tensor(x.grad(), {3, 3, 3, 3, 3, 3});
    std::cout << "Permute backward passed!" << std::endl;
}

int main() {
    test_view_backward();
    test_transpose_backward();
    test_permute_backward();
    return 0;
}
