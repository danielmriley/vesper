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

void test_view_basic() {
    std::cout << "Testing view basic..." << std::endl;
    // [2, 3] -> [6] -> [3, 2]
    auto t = vesper::empty({2, 3}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data = {0, 1, 2, 3, 4, 5};
    t.copy_from_host(h_data.data());
    
    auto v1 = t.view({6});
    assert(v1.shape() == std::vector<int64_t>({6}));
    assert(v1.is_contiguous());
    verify_tensor(v1, {0, 1, 2, 3, 4, 5});
    
    auto v2 = v1.view({3, 2});
    assert(v2.shape() == std::vector<int64_t>({3, 2}));
    verify_tensor(v2, {0, 1, 2, 3, 4, 5});
    
    std::cout << "View basic passed!" << std::endl;
}

void test_permute_basic() {
    std::cout << "Testing permute basic..." << std::endl;
    // [2, 3]
    // [[0, 1, 2],
    //  [3, 4, 5]]
    auto t = vesper::empty({2, 3}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data = {0, 1, 2, 3, 4, 5};
    t.copy_from_host(h_data.data());
    
    // permute(1, 0) -> [3, 2] (Transpose)
    // [[0, 3],
    //  [1, 4],
    //  [2, 5]]
    auto p = t.permute({1, 0});
    
    assert(p.shape() == std::vector<int64_t>({3, 2}));
    assert(!p.is_contiguous()); // Should have strides [1, 3]
    assert(p.strides()[0] == 1);
    assert(p.strides()[1] == 3);
    
    // Verify data logic (copy_to_host handles strided copy)
    verify_tensor(p, {0, 3, 1, 4, 2, 5});
    
    std::cout << "Permute basic passed!" << std::endl;
}

void test_reshape_with_copy() {
    std::cout << "Testing reshape with copy fallback..." << std::endl;
    // [2, 3] -> permute -> [3, 2] (non-contiguous)
    auto t = vesper::empty({2, 3}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data = {0, 1, 2, 3, 4, 5};
    t.copy_from_host(h_data.data());
    
    auto p = t.permute({1, 0}); // Non-contiguous
    
    // reshape to [6]
    // view() would fail here. reshape() should call contiguous() then view().
    auto r = p.reshape({6});
    
    assert(r.is_contiguous());
    assert(r.shape() == std::vector<int64_t>({6}));
    
    // Data should be in permuted order: 0, 3, 1, 4, 2, 5
    verify_tensor(r, {0, 3, 1, 4, 2, 5});
    
    std::cout << "Reshape with copy passed!" << std::endl;
}

void test_view_inferred_dim() {
    std::cout << "Testing view inferred dim..." << std::endl;
    auto t = vesper::empty({2, 3, 4}, vesper::DType::Float32, TEST_DEVICE); // 24 elements
    
    auto v = t.view({-1, 2}); // Should be [12, 2]
    assert(v.shape() == std::vector<int64_t>({12, 2}));
    
    auto v2 = t.view({4, -1, 3}); // Should be [4, 2, 3]
    assert(v2.shape() == std::vector<int64_t>({4, 2, 3}));
    
    std::cout << "View inferred dim passed!" << std::endl;
}

int main() {
    test_view_basic();
    test_permute_basic();
    test_reshape_with_copy();
    test_view_inferred_dim();
    return 0;
}
