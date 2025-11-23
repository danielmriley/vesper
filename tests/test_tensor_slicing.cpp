#include <vesper/core/factories.h>
#include <vesper/core/tensor.h>
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

void verify_tensor(const vesper::Tensor& t, const std::vector<float>& expected) {
    std::vector<float> data(t.numel());
    t.copy_to_host(data.data());
    for (size_t i = 0; i < data.size(); ++i) {
        if (std::fabs(data[i] - expected[i]) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": " << data[i] << " vs " << expected[i] << std::endl;
            assert(false);
        }
    }
}

void test_slice_basic() {
    std::cout << "Testing slice basic..." << std::endl;
    // [3, 2] tensor
    // [[0, 1],
    //  [2, 3],
    //  [4, 5]]
    auto t = vesper::empty({3, 2}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data = {0, 1, 2, 3, 4, 5};
    t.copy_from_host(h_data.data());
    
    // Slice index 1 -> [2, 3]
    auto s = t.slice(1);
    
    assert(s.shape() == std::vector<int64_t>({2}));
    assert(s.strides() == std::vector<int64_t>({1}));
    assert(s.offset() == 2); // 1 * stride[0] = 1 * 2 = 2
    
    verify_tensor(s, {2, 3});
    
    std::cout << "Slice basic passed!" << std::endl;
}

void test_slice_view_modification() {
    std::cout << "Testing slice view modification..." << std::endl;
    // Verify that modifying the slice affects the original tensor (shared storage)
    
    auto t = vesper::zeros({2, 2}, vesper::DType::Float32, TEST_DEVICE);
    auto s = t.slice(0); // First row
    
    // Modify slice on host and copy back
    // Since we don't have fill_ yet, copy from host
    std::vector<float> s_data = {10.0f, 20.0f};
    s.copy_from_host(s_data.data());
    
    // Verify original tensor
    // Should be [[10, 20], [0, 0]]
    verify_tensor(t, {10.0f, 20.0f, 0.0f, 0.0f});
    
    std::cout << "Slice view modification passed!" << std::endl;
}

void test_slice_3d() {
    std::cout << "Testing slice 3D..." << std::endl;
    // [2, 2, 2]
    auto t = vesper::zeros({2, 2, 2}, vesper::DType::Float32, TEST_DEVICE);
    // Fill with 0..7
    std::vector<float> h_data = {0, 1, 2, 3, 4, 5, 6, 7};
    t.copy_from_host(h_data.data());
    
    // slice(1) -> second 2x2 matrix -> 4, 5, 6, 7
    auto s1 = t.slice(1);
    assert(s1.shape() == std::vector<int64_t>({2, 2}));
    verify_tensor(s1, {4, 5, 6, 7});
    
    // slice again -> s1.slice(0) -> first row of second matrix -> 4, 5
    auto s2 = s1.slice(0);
    assert(s2.shape() == std::vector<int64_t>({2}));
    verify_tensor(s2, {4, 5});
    
    std::cout << "Slice 3D passed!" << std::endl;
}

void test_slice_out_of_bounds() {
    std::cout << "Testing slice out of bounds..." << std::endl;
    auto t = vesper::zeros({2, 2}, vesper::DType::Float32, TEST_DEVICE);
    
    try {
        auto s = t.slice(2); // Valid indices are 0, 1
        assert(false && "Should have thrown exception");
    } catch (const std::runtime_error& e) {
        // Expected
    }
    
    std::cout << "Slice out of bounds passed!" << std::endl;
}

int main() {
    test_slice_basic();
    test_slice_view_modification();
    test_slice_3d();
    test_slice_out_of_bounds();
    return 0;
}
