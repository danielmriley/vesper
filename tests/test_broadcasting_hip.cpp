#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

void test_broadcasting_backward_hip() {
#if USE_HIP_BACKEND
    std::cout << "Testing broadcasting backward on HIP..." << std::endl;
    
    int64_t M = 10;
    int64_t N = 5;
    
    // A: [M, N], B: [N]
    // C = A + B
    // Loss = sum(C)
    // dL/dB = sum_rows(dL/dC) = sum_rows(ones) = [M, M, ..., M]
    
    auto a = vesper::full({M, N}, vesper::DType::Float32, vesper::Device::HIP, 1.0f, true);
    auto b = vesper::full({N}, vesper::DType::Float32, vesper::Device::HIP, 2.0f, true);
    
    auto c = vesper::ops::add(a, b);
    auto loss = vesper::ops::sum(c);
    loss.backward();
    
    std::vector<float> grad_b(N);
    b.grad().copy_to_host(grad_b.data());
    
    for (int i = 0; i < N; ++i) {
        // Expected gradient is M (since we sum M rows of 1s)
        if (std::abs(grad_b[i] - (float)M) > 1e-5) {
            std::cerr << "Mismatch at " << i << ": expected " << M << ", got " << grad_b[i] << std::endl;
            exit(1);
        }
    }
    
    std::cout << "Broadcasting backward HIP passed!" << std::endl;

    std::cout << "Testing large broadcasting on HIP (1000x1000)..." << std::endl;
    int64_t LargeN = 1000;
    auto a_large = vesper::full({LargeN, LargeN}, vesper::DType::Float32, vesper::Device::HIP, 1.0f, true);
    auto b_large = vesper::full({LargeN}, vesper::DType::Float32, vesper::Device::HIP, 2.0f, true);
    
    auto c_large = vesper::ops::add(a_large, b_large);
    auto loss_large = vesper::ops::sum(c_large);
    loss_large.backward();
    
    std::vector<float> grad_b_large(LargeN);
    b_large.grad().copy_to_host(grad_b_large.data());
    
    for (int i = 0; i < LargeN; ++i) {
        if (std::abs(grad_b_large[i] - (float)LargeN) > 1e-5) {
            std::cerr << "Mismatch at " << i << ": expected " << LargeN << ", got " << grad_b_large[i] << std::endl;
            exit(1);
        }
    }
    std::cout << "Large broadcasting HIP passed!" << std::endl;

#else
    std::cout << "Skipping HIP test." << std::endl;
#endif
}

int main() {
    test_broadcasting_backward_hip();
    return 0;
}
