#include <vesper/ops/comparison.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include <vesper/nn/functional.h>
#include <vesper/autograd/guard.h>
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

void test_greater_than() {
    std::cout << "Testing greater_than..." << std::endl;
    auto a = vesper::empty({4}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    a.copy_from_host(data.data());
    
    auto result = vesper::ops::greater_than(a, 2.5f);
    
    // Expected: 0, 0, 1, 1
    verify_tensor(result, {0.0f, 0.0f, 1.0f, 1.0f});
    
    std::cout << "greater_than passed!" << std::endl;
}

void test_sqrt() {
    std::cout << "Testing sqrt forward and backward..." << std::endl;
    auto a = vesper::empty({4}, vesper::DType::Float32, TEST_DEVICE, true);
    std::vector<float> data = {1.0f, 4.0f, 9.0f, 16.0f};
    a.copy_from_host(data.data());
    
    auto result = vesper::ops::sqrt(a);
    
    // Forward: 1, 2, 3, 4
    verify_tensor(result, {1.0f, 2.0f, 3.0f, 4.0f});
    
    // Backward
    auto loss = vesper::ops::sum(result);
    loss.backward();
    
    // d(sqrt(x))/dx = 1 / (2 * sqrt(x))
    // x=1 -> 0.5
    // x=4 -> 0.25
    // x=9 -> 0.166666
    // x=16 -> 0.125
    verify_tensor(a.grad(), {0.5f, 0.25f, 1.0f/6.0f, 0.125f});
    
    std::cout << "sqrt passed!" << std::endl;
}

void test_sign() {
    std::cout << "Testing sign forward and backward..." << std::endl;
    auto a = vesper::empty({5}, vesper::DType::Float32, TEST_DEVICE, true);
    std::vector<float> data = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    a.copy_from_host(data.data());
    
    auto result = vesper::ops::sign(a);
    
    // Forward: -1, -1, 0, 1, 1
    verify_tensor(result, {-1.0f, -1.0f, 0.0f, 1.0f, 1.0f});
    
    // Backward
    auto loss = vesper::ops::sum(result);
    loss.backward();
    
    // Gradient of sign is 0 everywhere (technically undefined at 0, but usually 0)
    verify_tensor(a.grad(), {0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    
    std::cout << "sign passed!" << std::endl;
}

void test_div_forward() {
    std::cout << "Testing div forward..." << std::endl;
    auto a = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 10.0f);
    auto b = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 2.0f);
    
    auto result = vesper::ops::div(a, b);
    verify_tensor(result, {5.0f, 5.0f});
    
    auto result_scalar = vesper::ops::div(a, 4.0f);
    verify_tensor(result_scalar, {2.5f, 2.5f});
    
    std::cout << "div forward passed!" << std::endl;
}

void test_div_backward() {
    std::cout << "Testing div backward..." << std::endl;
    
    // Case 1: Tensor / Tensor
    // z = a / b
    // dz/da = 1/b
    // dz/db = -a / b^2
    
    auto a = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 6.0f, true);
    auto b = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 3.0f, true);
    
    auto z = vesper::ops::div(a, b);
    auto loss = vesper::ops::sum(z);
    loss.backward();
    
    // Check grad_a: 1/3 = 0.3333
    verify_tensor(a.grad(), {0.333333f, 0.333333f});
    
    // Check grad_b: -6 / 9 = -0.6666
    verify_tensor(b.grad(), {-0.666666f, -0.666666f});
    
    // Case 2: Tensor / Scalar
    // z = a / s
    // dz/da = 1/s
    
    auto c = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 10.0f, true);
    float s = 2.0f;
    
    auto z2 = vesper::ops::div(c, s);
    auto loss2 = vesper::ops::sum(z2);
    loss2.backward();
    
    // Check grad_c: 1/2 = 0.5
    verify_tensor(c.grad(), {0.5f, 0.5f});
    
    std::cout << "div backward passed!" << std::endl;
}

void test_inplace_correctness() {
    std::cout << "Testing in-place correctness..." << std::endl;
    
    // We must use requires_grad=false for in-place ops to be safe/allowed by default
    
    // 1. add_
    {
        auto a = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 1.0f, false);
        auto b = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 2.0f, false);
        a.add_(b);
        verify_tensor(a, {3.0f, 3.0f});
        
        a.add_(5.0f);
        verify_tensor(a, {8.0f, 8.0f});
    }
    
    // 2. sub_
    {
        auto a = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 10.0f, false);
        auto b = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 3.0f, false);
        a.sub_(b);
        verify_tensor(a, {7.0f, 7.0f});
        
        a.sub_(2.0f);
        verify_tensor(a, {5.0f, 5.0f});
    }
    
    // 3. mul_
    {
        auto a = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 2.0f, false);
        a.mul_(3.0f);
        verify_tensor(a, {6.0f, 6.0f});
    }
    
    std::cout << "in-place correctness passed!" << std::endl;
}

void test_mse_loss_functional() {
    std::cout << "Testing functional::mse_loss..." << std::endl;
    
    auto pred = vesper::empty({3}, vesper::DType::Float32, TEST_DEVICE, true);
    auto target = vesper::empty({3}, vesper::DType::Float32, TEST_DEVICE, false);
    
    std::vector<float> p_data = {1.0f, 2.0f, 3.0f};
    std::vector<float> t_data = {1.0f, 2.0f, 3.0f};
    
    pred.copy_from_host(p_data.data());
    target.copy_from_host(t_data.data());
    
    // Case 1: Zero loss
    auto loss = vesper::nn::functional::mse_loss(pred, target);
    verify_tensor(loss, {0.0f});
    
    // Case 2: Non-zero
    // pred = [1, 2, 3]
    // target = [2, 2, 2]
    // diff = [-1, 0, 1]
    // sq = [1, 0, 1]
    // mean = 2/3 = 0.6666
    
    std::vector<float> t_data2 = {2.0f, 2.0f, 2.0f};
    target.copy_from_host(t_data2.data());
    
    // Clear grads
    pred.grad() = vesper::zeros(pred.shape(), pred.dtype(), pred.device());
    
    loss = vesper::nn::functional::mse_loss(pred, target);
    verify_tensor(loss, {2.0f/3.0f});
    
    // Backward
    loss.backward();
    
    // grad = 2/N * (pred - target)
    // N=3
    // i=0: 2/3 * (1-2) = -0.6666
    // i=1: 2/3 * (2-2) = 0
    // i=2: 2/3 * (3-2) = 0.6666
    
    verify_tensor(pred.grad(), {-2.0f/3.0f, 0.0f, 2.0f/3.0f});
    
    std::cout << "functional::mse_loss passed!" << std::endl;
}

int main() {
    try {
        test_greater_than();
        test_sqrt();
        test_sign();
        test_div_forward();
        test_div_backward();
        test_inplace_correctness();
        test_mse_loss_functional();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
