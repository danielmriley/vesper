#include <vesper/ops/elementwise.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <numeric>

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
            assert(false);
        }
    }
}

void test_broadcast_3d_2d() {
    std::cout << "Testing 3D + 2D broadcast: [2, 3, 4] + [3, 1]..." << std::endl;
    // A: [2, 3, 4]
    // B: [3, 1] -> broadcasts to [1, 3, 1] -> [2, 3, 4]
    // Value added depends on dim 1.
    
    auto a = vesper::full({2, 3, 4}, vesper::DType::Float32, TEST_DEVICE, 1.0f);
    auto b = vesper::empty({3, 1}, vesper::DType::Float32, TEST_DEVICE);
    
    std::vector<float> b_data = {10.0f, 20.0f, 30.0f};
    b.copy_from_host(b_data.data());
    
    auto c = vesper::ops::add(a, b);
    
    // Validation logic:
    // c[i, j, k] = a[i, j, k] + b[j, 0]
    //            = 1.0 + b_data[j]
    std::vector<float> expected(2 * 3 * 4);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                expected[i*12 + j*4 + k] = 1.0f + b_data[j];
            }
        }
    }
    
    verify_tensor(c, expected);
    std::cout << "3D + 2D broadcast passed!" << std::endl;
}

void test_broadcast_4d_scalar() {
    std::cout << "Testing 4D + scalar broadcast: [2, 2, 2, 2] + [1]..." << std::endl;
    
    auto a = vesper::full({2, 2, 2, 2}, vesper::DType::Float32, TEST_DEVICE, 2.0f);
    auto b = vesper::full({1}, vesper::DType::Float32, TEST_DEVICE, 3.0f);
    
    auto c = vesper::ops::mul(a, b);
    
    std::vector<float> expected(16, 6.0f);
    verify_tensor(c, expected);
    
    std::cout << "4D + scalar broadcast passed!" << std::endl;
}

void test_broadcast_mismatch_error() {
    std::cout << "Testing broadcast shape mismatch error..." << std::endl;
    
    auto a = vesper::empty({2, 3}, vesper::DType::Float32, TEST_DEVICE);
    auto b = vesper::empty({2, 4}, vesper::DType::Float32, TEST_DEVICE);
    
    try {
        auto c = vesper::ops::add(a, b);
        assert(false && "Should have thrown exception");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        assert(msg.find("Shapes are not broadcastable") != std::string::npos);
    }
    
    std::cout << "Broadcast mismatch error passed!" << std::endl;
}

void test_broadcast_complex_expansion() {
    std::cout << "Testing complex expansion: [1, 3, 1] + [2, 1, 4]..." << std::endl;
    // Should result in [2, 3, 4]
    
    auto a = vesper::empty({1, 3, 1}, vesper::DType::Float32, TEST_DEVICE);
    auto b = vesper::empty({2, 1, 4}, vesper::DType::Float32, TEST_DEVICE);
    
    // Fill A with 0, 1, 2
    std::vector<float> a_data = {0.0f, 1.0f, 2.0f};
    a.copy_from_host(a_data.data());
    
    // Fill B with 10
    std::vector<float> b_data(8, 10.0f);
    b.copy_from_host(b_data.data());
    
    auto c = vesper::ops::add(a, b);
    
    assert(c.shape() == std::vector<int64_t>({2, 3, 4}));
    
    std::vector<float> expected(24);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                // a[0, j, 0] + b[i, 0, k]
                expected[i*12 + j*4 + k] = a_data[j] + 10.0f;
            }
        }
    }
    
    verify_tensor(c, expected);
    std::cout << "Complex expansion passed!" << std::endl;
}

void test_broadcast_sub_div() {
    std::cout << "Testing sub/div broadcasting..." << std::endl;
    
    // [2, 2]
    // [[10, 20],
    //  [30, 40]]
    auto a = vesper::empty({2, 2}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> a_data = {10, 20, 30, 40};
    a.copy_from_host(a_data.data());
    
    // [2, 1]
    // [[2],
    //  [5]]
    auto b = vesper::empty({2, 1}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> b_data = {2, 5};
    b.copy_from_host(b_data.data());
    
    // sub: [[8, 18], [25, 35]]
    auto s = vesper::ops::sub(a, b);
    verify_tensor(s, {8, 18, 25, 35});
    
    // div: [[5, 10], [6, 8]]
    auto d = vesper::ops::div(a, b);
    verify_tensor(d, {5, 10, 6, 8});
    
    std::cout << "Sub/Div broadcasting passed!" << std::endl;
}

int main() {
    test_broadcast_3d_2d();
    test_broadcast_4d_scalar();
    test_broadcast_mismatch_error();
    test_broadcast_complex_expansion();
    test_broadcast_sub_div();
    return 0;
}
