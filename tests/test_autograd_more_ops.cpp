#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <stdexcept>

#if defined(USE_HIP_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CUDA;
#elif defined(USE_CPU_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CPU;
#else
    #error "No backend enabled for testing"
#endif

void verify_tensor(const vesper::Tensor& t, const std::vector<float>& expected, float tol = 1e-5) {
    std::vector<float> data(t.numel());
    t.copy_to_host(data.data());
    for (size_t i = 0; i < data.size(); ++i) {
        if (std::fabs(data[i] - expected[i]) > tol) {
            std::cerr << "Mismatch at index " << i << ": " << data[i] << " vs " << expected[i] << std::endl;
            throw std::runtime_error("Tensor verification failed!");
        }
    }
}

void test_sub_backward() {
    std::cout << "Testing sub backward pass..." << std::endl;
    // y = a - b
    // Loss = sum(y)
    // dL/da = dL/dy * dy/da = 1 * 1 = 1
    // dL/db = dL/dy * dy/db = 1 * -1 = -1
    
    auto a = vesper::full({2, 2}, vesper::DType::Float32, TEST_DEVICE, 10.0f, true);
    auto b = vesper::full({2, 2}, vesper::DType::Float32, TEST_DEVICE, 2.0f, true);
    
    auto y = vesper::ops::sub(a, b);
    auto loss = vesper::ops::sum(y);
    
    loss.backward();
    
    // Verify gradients
    verify_tensor(a.grad(), {1.0f, 1.0f, 1.0f, 1.0f});
    verify_tensor(b.grad(), {-1.0f, -1.0f, -1.0f, -1.0f});
    
    std::cout << "Sub backward passed!" << std::endl;
}

void test_div_backward() {
    std::cout << "Testing div backward pass..." << std::endl;
    // y = a / b
    // Loss = sum(y)
    // dL/da = 1 * (1/b)
    // dL/db = 1 * (-a/b^2)
    
    auto a = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 10.0f, true);
    auto b = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 2.0f, true);
    
    auto y = vesper::ops::div(a, b); // y = [5, 5]
    auto loss = vesper::ops::sum(y);
    
    loss.backward();
    
    // Expected grads
    // da = 1/2 = 0.5
    // db = -10 / (2^2) = -10/4 = -2.5
    
    verify_tensor(a.grad(), {0.5f, 0.5f});
    verify_tensor(b.grad(), {-2.5f, -2.5f});
    
    std::cout << "Div backward passed!" << std::endl;
}

void test_div_scalar_backward() {
    std::cout << "Testing div scalar backward pass..." << std::endl;
    // y = a / 2.0
    // dL/da = 1 / 2.0 = 0.5
    
    auto a = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 10.0f, true);
    auto y = vesper::ops::div(a, 2.0f);
    auto loss = vesper::ops::sum(y);
    
    loss.backward();
    
    verify_tensor(a.grad(), {0.5f, 0.5f});
    
    std::cout << "Div scalar backward passed!" << std::endl;
}

void test_broadcast_sub_backward() {
    std::cout << "Testing broadcast sub backward..." << std::endl;
    // a: [2, 2], b: [2, 1]
    // y = a - b
    // loss = sum(y)
    
    // gradients should accumulate
    
    auto a = vesper::full({2, 2}, vesper::DType::Float32, TEST_DEVICE, 10.0f, true);
    auto b = vesper::full({2, 1}, vesper::DType::Float32, TEST_DEVICE, 2.0f, true);
    
    auto y = vesper::ops::sub(a, b);
    auto loss = vesper::ops::sum(y);
    
    loss.backward();
    
    // dL/da is all 1s
    verify_tensor(a.grad(), {1.0f, 1.0f, 1.0f, 1.0f});
    
    // dL/db: b is broadcasted across dim 1 (cols).
    // b[0] contributes to y[0,0] and y[0,1]. dL/db[0] = -1 + -1 = -2.
    // b[1] contributes to y[1,0] and y[1,1]. dL/db[1] = -1 + -1 = -2.
    verify_tensor(b.grad(), {-2.0f, -2.0f});
    
    std::cout << "Broadcast sub backward passed!" << std::endl;
}

void test_broadcast_mul_backward() {
    std::cout << "Testing broadcast mul backward..." << std::endl;
    // a: [2, 2], b: [2, 1]
    // y = a * b
    // loss = sum(y)
    // dL/da = b (broadcasted)
    // dL/db = sum_cols(a)
    
    auto a = vesper::full({2, 2}, vesper::DType::Float32, TEST_DEVICE, 3.0f, true);
    auto b = vesper::full({2, 1}, vesper::DType::Float32, TEST_DEVICE, 2.0f, true);
    
    auto y = vesper::ops::mul(a, b);
    auto loss = vesper::ops::sum(y);
    
    loss.backward();
    
    // dL/da = b (2.0)
    verify_tensor(a.grad(), {2.0f, 2.0f, 2.0f, 2.0f});
    
    // dL/db = sum(a, dim=1). a is all 3.0. sum is 3+3=6.
    verify_tensor(b.grad(), {6.0f, 6.0f});
    
    std::cout << "Broadcast mul backward passed!" << std::endl;
}

int main() {
    try {
        test_sub_backward();
        test_div_backward();
        test_div_scalar_backward();
        test_broadcast_sub_backward();
        test_broadcast_mul_backward();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}