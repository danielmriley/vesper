#include <vesper/core/factories.h>
#include <vesper/ops/random.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_uniform_gpu() {
#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)
    std::cout << "Testing uniform random on GPU..." << std::endl;
    
#if defined(USE_HIP_BACKEND)
    vesper::Device device = vesper::Device::HIP;
#else
    vesper::Device device = vesper::Device::CUDA;
#endif

    int64_t N = 10000;
    float min = -1.0f;
    float max = 1.0f;
    
    auto t = vesper::empty({N}, vesper::DType::Float32, device);
    vesper::ops::uniform_(t, min, max);
    
    // Calculate mean and variance on GPU
    auto m = vesper::ops::mean(t);
    
    std::vector<float> m_data(1);
    m.copy_to_host(m_data.data());
    
    float mean_val = m_data[0];
    float expected_mean = (min + max) / 2.0f; // 0.0
    
    std::cout << "  Mean: " << mean_val << " (Expected: " << expected_mean << ")" << std::endl;
    
    // Check range
    std::vector<float> data(N);
    t.copy_to_host(data.data());
    
    bool in_range = true;
    for (float val : data) {
        if (val < min || val > max) {
            in_range = false;
            break;
        }
    }
    assert(in_range);
    
    // Check that it's not all the same value
    bool all_same = true;
    for (size_t i = 1; i < N; ++i) {
        if (data[i] != data[0]) {
            all_same = false;
            break;
        }
    }
    assert(!all_same);
    
    if (std::abs(mean_val - expected_mean) < 0.1f) {
        std::cout << "Uniform random on GPU passed!" << std::endl;
    } else {
        std::cerr << "Mean is too far off!" << std::endl;
        exit(1);
    }
#else
    std::cout << "Skipping GPU test." << std::endl;
#endif
}

int main() {
    test_uniform_gpu();
    return 0;
}
