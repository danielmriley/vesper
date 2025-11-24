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

void test_slice_range() {
    std::cout << "Testing slice range [start:stop]..." << std::endl;
    // [5] -> 0, 1, 2, 3, 4
    auto t = vesper::empty({5}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data = {0, 1, 2, 3, 4};
    t.copy_from_host(h_data.data());
    
    // slice(1, 4) -> indices 1, 2, 3
    auto s = t.slice(1, 4);
    
    assert(s.shape() == std::vector<int64_t>({3}));
    verify_tensor(s, {1, 2, 3});
    
    std::cout << "Slice range passed!" << std::endl;
}

void test_slice_step() {
    std::cout << "Testing slice step [::2]..." << std::endl;
    // [6] -> 0, 1, 2, 3, 4, 5
    auto t = vesper::empty({6}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data = {0, 1, 2, 3, 4, 5};
    t.copy_from_host(h_data.data());
    
    // slice(0, 6, 2) -> 0, 2, 4
    auto s = t.slice(0, 6, 2);
    
    assert(s.shape() == std::vector<int64_t>({3}));
    verify_tensor(s, {0, 2, 4});
    
    std::cout << "Slice step passed!" << std::endl;
}

void test_slice_negative() {
    std::cout << "Testing negative slice indices..." << std::endl;
    // [5] -> 0, 1, 2, 3, 4
    auto t = vesper::empty({5}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data = {0, 1, 2, 3, 4};
    t.copy_from_host(h_data.data());
    
    // slice(1, -1) -> 1, 2, 3 (excludes 4)
    auto s = t.slice(1, -1);
    
    assert(s.shape() == std::vector<int64_t>({3}));
    verify_tensor(s, {1, 2, 3});
    
    std::cout << "Negative slice passed!" << std::endl;
}

void test_multi_dim_index() {
    std::cout << "Testing multi-dim index [:, 1:3]..." << std::endl;
    // [2, 4]
    // [[0, 1, 2, 3],
    //  [4, 5, 6, 7]]
    auto t = vesper::empty({2, 4}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data = {0, 1, 2, 3, 4, 5, 6, 7};
    t.copy_from_host(h_data.data());
    
    // Select all rows (slice()), cols 1:3 (slice(1, 3))
    auto s = t.index({ vesper::Slice(), vesper::Slice(1, 3) });
    
    // Expected: [[1, 2], [5, 6]]
    assert(s.shape() == std::vector<int64_t>({2, 2}));
    verify_tensor(s, {1, 2, 5, 6});
    
    std::cout << "Multi-dim index passed!" << std::endl;
}

void test_reverse_slice() {
    std::cout << "Testing reverse slice [::-1]..." << std::endl;
    // [3] -> 0, 1, 2
    auto t = vesper::empty({3}, vesper::DType::Float32, TEST_DEVICE);
    std::vector<float> h_data = {0, 1, 2};
    t.copy_from_host(h_data.data());
    
    // slice(2, -1, -1) -> indices 2, 1, 0
    // Note: Stop is exclusive. If stop is -1 (index 2), it excludes index 2?
    // Python: [::-1] implicitly start=2, stop=-1 (sentinel), step=-1.
    // My slice implementation takes int64_t.
    // Slice() defaults start/stop for negative step?
    
    // Using index with default Slice struct for full reversal
    auto s = t.index({ vesper::Slice(std::nullopt, std::nullopt, -1) });
    
    assert(s.shape() == std::vector<int64_t>({3}));
    assert(s.strides()[0] == -1); // Negative stride
    verify_tensor(s, {2, 1, 0});
    
    std::cout << "Reverse slice passed!" << std::endl;
}

int main() {
    test_slice_range();
    test_slice_step();
    test_slice_negative();
    test_multi_dim_index();
    test_reverse_slice();
    return 0;
}
