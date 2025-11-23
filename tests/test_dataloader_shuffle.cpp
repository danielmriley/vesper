#include <vesper/data/dataloader.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <set>

#if defined(USE_HIP_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CUDA;
#elif defined(USE_CPU_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CPU;
#else
    #error "No backend enabled for testing"
#endif

void test_dataloader_shuffle() {
    std::cout << "Testing DataLoader shuffle..." << std::endl;
    
    int N = 20;
    // Create dummy data: x = 0..19
    auto x = vesper::empty({N, 1}, vesper::DType::Float32, TEST_DEVICE);
    auto y = vesper::empty({N, 1}, vesper::DType::Float32, TEST_DEVICE);
    
    std::vector<float> h_data(N);
    for(int i=0; i<N; ++i) h_data[i] = static_cast<float>(i);
    x.copy_from_host(h_data.data());
    y.copy_from_host(h_data.data()); // y doesn't matter much
    
    auto dataset = std::make_shared<vesper::data::TensorDataset>(x, y);
    
    // Create loader with shuffle=true
    // Batch size 1 to easily track indices
    auto loader = vesper::data::DataLoader(dataset, 1, true);
    
    std::vector<float> seen_indices;
    
    for (const auto& batch : loader) {
        // batch.first is [1, 1] tensor containing the index
        float val;
        batch.first.copy_to_host(&val);
        seen_indices.push_back(val);
    }
    
    assert(seen_indices.size() == N);
    
    // Check coverage (all indices 0..19 present)
    std::set<int> unique_indices;
    for (float v : seen_indices) unique_indices.insert(static_cast<int>(v));
    
    for (int i = 0; i < N; ++i) {
        assert(unique_indices.count(i) && "Missing index in shuffled loader");
    }
    
    // Check that it's not just sequential 0..19 (statistically extremely unlikely to be sequential if N=20)
    bool is_sequential = true;
    for (int i = 0; i < N; ++i) {
        if (static_cast<int>(seen_indices[i]) != i) {
            is_sequential = false;
            break;
        }
    }
    assert(!is_sequential && "Shuffle produced sequential output (unlikely failure)");
    
    std::cout << "DataLoader shuffle passed!" << std::endl;
}

int main() {
    test_dataloader_shuffle();
    return 0;
}
