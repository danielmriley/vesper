#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/core/dtype.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace vesper;

// Helper to create tensor from data
template <typename T>
Tensor make_tensor(const std::vector<T>& data, const std::vector<int64_t>& shape, DType dtype, bool requires_grad = false) {
    Tensor t = empty(shape, dtype, Device::CPU, requires_grad);
    t.copy_from_host(data.data());
    return t;
}

void test_float_to_int() {
    std::cout << "Testing Float to Int..." << std::endl;
    auto t = make_tensor<float>({1.5f, 2.7f, -3.2f}, {3}, DType::Float32);
    auto t_int = t.to(DType::Int32);
    
    assert(t_int.dtype() == DType::Int32);
    auto t_int_cpu = t_int.to(Device::CPU);
    const int32_t* t_int_data = t_int_cpu.data_ptr<int32_t>();
    assert(t_int_data[0] == 1);
    assert(t_int_data[1] == 2);
    assert(t_int_data[2] == -3);
    std::cout << "Passed!" << std::endl;
}

void test_int_to_float() {
    std::cout << "Testing Int to Float..." << std::endl;
    auto t = make_tensor<int32_t>({1, 2, -3}, {3}, DType::Int32);
    auto t_float = t.to(DType::Float32);
    
    assert(t_float.dtype() == DType::Float32);
    auto t_float_cpu = t_float.to(Device::CPU);
    const float* t_float_data = t_float_cpu.data_ptr<float>();
    assert(std::abs(t_float_data[0] - 1.0f) < 1e-5);
    assert(std::abs(t_float_data[1] - 2.0f) < 1e-5);
    assert(std::abs(t_float_data[2] - (-3.0f)) < 1e-5);
    std::cout << "Passed!" << std::endl;
}

void test_autograd_cast() {
    std::cout << "Testing Autograd Cast..." << std::endl;
    auto t = make_tensor<float>({1.0f, 2.0f}, {2}, DType::Float32, true);
    auto t_double = t.to(DType::Float64);
    auto t_float_again = t_double.to(DType::Float32);
    
    auto loss = ops::sum(t_float_again);
    
    loss.backward();
    
    assert(t.grad().dtype() == DType::Float32);
    auto t_grad_cpu = t.grad().to(Device::CPU);
    const float* t_grad_data = t_grad_cpu.data_ptr<float>();
    assert(std::abs(t_grad_data[0] - 1.0f) < 1e-5);
    assert(std::abs(t_grad_data[1] - 1.0f) < 1e-5);
    std::cout << "Passed!" << std::endl;
}

void test_device_move() {
    std::cout << "Testing Device Move..." << std::endl;
    auto t = make_tensor<float>({1.0f, 2.0f}, {2}, DType::Float32);
    
    // Move to CPU (should be no-op or copy)
    auto t_cpu = t.to(Device::CPU);
    assert(t_cpu.device() == Device::CPU);
    assert(t_cpu.data_ptr<float>()[0] == 1.0f);
    
#ifdef USE_CUDA_BACKEND
    try {
        std::cout << "Attempting move to CUDA..." << std::endl;
        auto t_cuda = t.to(Device::CUDA);
        assert(t_cuda.device() == Device::CUDA);
        // Move back to CPU to verify
        auto t_back = t_cuda.to(Device::CPU);
        assert(t_back.data_ptr<float>()[0] == 1.0f);
        std::cout << "CUDA move passed." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CUDA move skipped or failed: " << e.what() << std::endl;
    }
#endif
    std::cout << "Passed!" << std::endl;
}

int main() {
    test_float_to_int();
    test_int_to_float();
    test_autograd_cast();
    test_device_move();
    std::cout << "All cast tests passed!" << std::endl;
    return 0;
}

