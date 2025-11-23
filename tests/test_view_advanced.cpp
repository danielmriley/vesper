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

void test_view_on_slice() {
    std::cout << "Testing view on non-contiguous slice..." << std::endl;
    // [4, 4]
    auto t = vesper::empty({4, 4}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data(16);
    for(int i=0; i<16; ++i) h_data[i] = (float)i;
    t.copy_from_host(h_data.data());
    
    // Slice row 1 (index 1)
    // [4] tensor, stride [1], offset 4. Contiguous chunk.
    auto s = t.slice(1);
    
    // View as [2, 2]. Stride [2, 1].
    // Should work because slice is contiguous in memory (stride 1).
    auto v = s.view({2, 2});
    
    assert(v.shape() == std::vector<int64_t>({2, 2}));
    verify_tensor(v, {4, 5, 6, 7});
    
    std::cout << "View on slice passed!" << std::endl;
}

void test_view_on_transpose_fail() {
    std::cout << "Testing view on transpose (should fail)..." << std::endl;
    // [4, 2] -> transpose -> [2, 4]. Strides [1, 2].
    // Not contiguous. Cannot be viewed as [8] or [4, 2] with default strides.
    
    auto t = vesper::empty({4, 2}, vesper::DType::Float32, TEST_DEVICE);
    auto p = t.permute({1, 0}); 
    
    bool failed = false;
    try {
        auto v = p.view({8});
    } catch (...) {
        failed = true;
    }
    assert(failed && "View on transpose should fail");
    
    std::cout << "View on transpose failure passed!" << std::endl;
}

void test_view_merge_dims() {
    std::cout << "Testing view merging dimensions..." << std::endl;
    // [2, 2, 2] -> [2, 4]
    auto t = vesper::empty({2, 2, 2}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data(8);
    for(int i=0; i<8; ++i) h_data[i] = (float)i;
    t.copy_from_host(h_data.data());
    
    auto v = t.view({2, 4});
    assert(v.shape() == std::vector<int64_t>({2, 4}));
    verify_tensor(v, h_data);
    
    std::cout << "View merge dims passed!" << std::endl;
}

void test_view_split_dims() {
    std::cout << "Testing view splitting dimensions..." << std::endl;
    // [2, 4] -> [2, 2, 2]
    auto t = vesper::empty({2, 4}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data(8);
    for(int i=0; i<8; ++i) h_data[i] = (float)i;
    t.copy_from_host(h_data.data());
    
    auto v = t.view({2, 2, 2});
    assert(v.shape() == std::vector<int64_t>({2, 2, 2}));
    verify_tensor(v, h_data);
    
    std::cout << "View split dims passed!" << std::endl;
}

int main() {
    test_view_on_slice();
    test_view_on_transpose_fail();
    test_view_merge_dims();
    test_view_split_dims();
    return 0;
}
