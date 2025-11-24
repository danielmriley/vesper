#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_div_mean_gpu() {
#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)
    std::cout << "Testing div and mean on GPU..." << std::endl;
    
#if defined(USE_HIP_BACKEND)
    vesper::Device device = vesper::Device::HIP;
#else
    vesper::Device device = vesper::Device::CUDA;
#endif

    int64_t N = 10;
    auto a = vesper::full({N}, vesper::DType::Float32, device, 10.0f, true);
    auto b = vesper::full({N}, vesper::DType::Float32, device, 2.0f, true);
    
    // Test div
    auto c = vesper::ops::div(a, b); // 5.0
    
    std::vector<float> c_data(N);
    c.copy_to_host(c_data.data());
    assert(std::abs(c_data[0] - 5.0f) < 1e-5);
    
    // Test mean
    auto m = vesper::ops::mean(c); // 5.0
    
    std::vector<float> m_data(1);
    m.copy_to_host(m_data.data());
    assert(std::abs(m_data[0] - 5.0f) < 1e-5);
    
    // Test backward
    m.backward();
    
    // d(mean)/da = d(mean)/dc * dc/da
    // mean = sum(c)/N
    // d(mean)/dc = 1/N
    // c = a/b
    // dc/da = 1/b
    // d(mean)/da = (1/N) * (1/b) = 1/10 * 1/2 = 0.05
    
    std::vector<float> grad_a(N);
    a.grad().copy_to_host(grad_a.data());
    
    if (std::abs(grad_a[0] - 0.05f) > 1e-5) {
        std::cerr << "Gradient mismatch: expected 0.05, got " << grad_a[0] << std::endl;
        exit(1);
    }
    
    std::cout << "Div and Mean on GPU passed!" << std::endl;
#else
    std::cout << "Skipping GPU test." << std::endl;
#endif
}

int main() {
    test_div_mean_gpu();
    return 0;
}
