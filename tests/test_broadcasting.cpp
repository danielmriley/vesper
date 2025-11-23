#include <vesper/ops/elementwise.h>
#include <vesper/core/factories.h>
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

void test_broadcast_add_vector_to_matrix() {
    std::cout << "Testing broadcasting add: [2, 3] + [3]..." << std::endl;
    // A: 2x3
    // B: 3 (broadcasts to 2x3 by adding to each row)
    
    std::vector<int64_t> shape_a = {2, 3};
    std::vector<int64_t> shape_b = {3};
    
    auto a = vesper::empty(shape_a, vesper::DType::Float32, TEST_DEVICE);
    auto b = vesper::empty(shape_b, vesper::DType::Float32, TEST_DEVICE);
    
    // A = [[1, 2, 3], [4, 5, 6]]
    std::vector<float> data_a = {1, 2, 3, 4, 5, 6};
    // B = [10, 20, 30]
    std::vector<float> data_b = {10, 20, 30};
    
    a.copy_from_host(data_a.data());
    b.copy_from_host(data_b.data());
    
    auto c = vesper::ops::add(a, b);
    
    // Expected: [[11, 22, 33], [14, 25, 36]]
    std::vector<float> result(6);
    c.copy_to_host(result.data());
    
    assert(c.shape() == shape_a);
    assert(std::fabs(result[0] - 11.0f) < 1e-5);
    assert(std::fabs(result[1] - 22.0f) < 1e-5);
    assert(std::fabs(result[2] - 33.0f) < 1e-5);
    assert(std::fabs(result[3] - 14.0f) < 1e-5);
    assert(std::fabs(result[4] - 25.0f) < 1e-5);
    assert(std::fabs(result[5] - 36.0f) < 1e-5);
    
    std::cout << "Broadcast add [2,3]+[3] passed!" << std::endl;
}

void test_broadcast_scalar() {
    std::cout << "Testing broadcasting add: [2, 2] + [1]..." << std::endl;
    // A: 2x2
    // B: 1 (scalar broadcast)
    
    auto a = vesper::full({2, 2}, vesper::DType::Float32, TEST_DEVICE, 1.0f);
    auto b = vesper::full({1}, vesper::DType::Float32, TEST_DEVICE, 5.0f);
    
    auto c = vesper::ops::add(a, b);
    
    std::vector<float> result(4);
    c.copy_to_host(result.data());
    
    for(float v : result) {
        assert(std::fabs(v - 6.0f) < 1e-5);
    }
    std::cout << "Broadcast scalar passed!" << std::endl;
}

void test_broadcast_column() {
    std::cout << "Testing broadcasting add: [2, 3] + [2, 1]..." << std::endl;
    // A: 2x3
    // B: 2x1 (broadcasts to 2x3 by adding to each column)
    
    auto a = vesper::full({2, 3}, vesper::DType::Float32, TEST_DEVICE, 1.0f);
    // B = [[10], [20]]
    auto b = vesper::empty({2, 1}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> data_b = {10.0f, 20.0f};
    b.copy_from_host(data_b.data());
    
    auto c = vesper::ops::add(a, b);
    // Expected: [[11, 11, 11], [21, 21, 21]]
    
    std::vector<float> result(6);
    c.copy_to_host(result.data());
    
    assert(std::fabs(result[0] - 11.0f) < 1e-5); // row 0
    assert(std::fabs(result[1] - 11.0f) < 1e-5);
    assert(std::fabs(result[2] - 11.0f) < 1e-5);
    assert(std::fabs(result[3] - 21.0f) < 1e-5); // row 1
    assert(std::fabs(result[4] - 21.0f) < 1e-5);
    assert(std::fabs(result[5] - 21.0f) < 1e-5);
    
    std::cout << "Broadcast column passed!" << std::endl;
}

int main() {
    test_broadcast_add_vector_to_matrix();
    test_broadcast_scalar();
    test_broadcast_column();
    return 0;
}
