#include <vesper/core/factories.h>
#include <vesper/core/tensor.h>
#include <vesper/ops/reduction.h>
#include <iostream>
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

using namespace vesper;

void test_operators() {
    std::cout << "Testing operator overloading..." << std::endl;
    
    auto a = full({2}, DType::Float32, TEST_DEVICE, 2.0f);
    auto b = full({2}, DType::Float32, TEST_DEVICE, 3.0f);
    
    // +
    auto c = a + b;
    assert(c.item<float>() == 5.0f || c.slice(0).item<float>() == 5.0f); // Check value
    // Check vector values
    std::vector<float> c_data(2);
    c.copy_to_host(c_data.data());
    assert(c_data[0] == 5.0f);
    
    // -
    auto d = a - b; // -1
    std::vector<float> d_data(2);
    d.copy_to_host(d_data.data());
    assert(d_data[0] == -1.0f);
    
    // *
    auto e = a * b; // 6
    std::vector<float> e_data(2);
    e.copy_to_host(e_data.data());
    assert(e_data[0] == 6.0f);
    
    // /
    auto f = a / b; // 2/3
    std::vector<float> f_data(2);
    f.copy_to_host(f_data.data());
    assert(std::abs(f_data[0] - 0.6666f) < 1e-4);
    
    // Scalar ops
    auto g = a + 10.0f; // 12
    std::vector<float> g_data(2);
    g.copy_to_host(g_data.data());
    assert(g_data[0] == 12.0f);
    
    auto h = 10.0f - a; // 8
    std::vector<float> h_data(2);
    h.copy_to_host(h_data.data());
    assert(h_data[0] == 8.0f);
    
    std::cout << "Operator overloading tests passed!" << std::endl;
}

void test_item() {
    std::cout << "Testing item()..." << std::endl;
    
    auto a = full({1}, DType::Float32, TEST_DEVICE, 42.0f);
    float val = a.item<float>();
    
    assert(val == 42.0f);
    
    // Test error on non-scalar
    auto b = full({2}, DType::Float32, TEST_DEVICE, 1.0f);
    try {
        b.item<float>();
        assert(false);
    } catch (...) {}
    
    std::cout << "item() tests passed!" << std::endl;
}

void test_clone() {
    std::cout << "Testing clone()..." << std::endl;
    
    // 1. Independence Check (No Autograd)
    auto a_no_grad = full({2}, DType::Float32, TEST_DEVICE, 1.0f, false);
    auto b_no_grad = a_no_grad.clone();
    
    b_no_grad.add_(a_no_grad); // Allowed since no grad
    
    std::vector<float> a_data(2), b_data(2);
    a_no_grad.copy_to_host(a_data.data());
    b_no_grad.copy_to_host(b_data.data());
    
    assert(a_data[0] == 1.0f); // a unchanged
    assert(b_data[0] == 2.0f); // b changed
    
    // 2. Autograd Check
    auto a = full({2}, DType::Float32, TEST_DEVICE, 1.0f, true);
    auto b = a.clone(); // b requires grad, has grad_node
    
    // Operation on clone: y = b + a
    auto y = b + a; 
    
    auto loss = ops::sum(y); 
    loss.backward();
    
    // y = clone(a) + a
    // dy/da = d(clone)/da + d(a)/da = 1 + 1 = 2
    
    std::vector<float> a_grad(2);
    a.grad().copy_to_host(a_grad.data());
    assert(a_grad[0] == 2.0f);
    
    std::cout << "clone() tests passed!" << std::endl;
}

int main() {
    test_operators();
    test_item();
    test_clone();
    return 0;
}
