#include "vesper/core/factories.h"
#include "vesper/core/reference_ops.h"
#include "vesper/ops/reduction.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

using namespace vesper;

void check_float(float actual, float expected, float tol = 1e-4) {
    if (std::fabs(actual - expected) > tol) {
        std::cerr << "Check failed: actual=" << actual << ", expected=" << expected << std::endl;
        exit(1);
    }
}

void test_sum_cpu() {
    std::cout << "Running SumCPU..." << std::endl;
    auto t = vesper::full({100}, DType::Float32, Device::CPU, 1.0f);
    auto res = ops::sum(t);
    
    assert(res.numel() == 1);
    float val = *res.data_ptr<float>();
    check_float(val, 100.0f);
    std::cout << "PASSED" << std::endl;
}

void test_sum_hip() {
#ifdef USE_HIP_BACKEND
    std::cout << "Running SumHIP..." << std::endl;
    int n = 1024 * 1024;
    auto t = vesper::full({n}, DType::Float32, Device::HIP, 1.0f);
    auto res = ops::sum(t);
    
    float val = 0.0f;
    res.copy_to_host(&val);

    check_float(val, (float)n);
    std::cout << "PASSED" << std::endl;
#else
    std::cout << "Skipping SumHIP (HIP disabled)" << std::endl;
#endif
}

void test_sum_hip_large() {
#ifdef USE_HIP_BACKEND
    std::cout << "Running SumHIPLarge..." << std::endl;
    int n = 10000000; 
    auto t = vesper::full({n}, DType::Float32, Device::HIP, 1.0f);
    auto res = ops::sum(t);
    
    float val = 0.0f;
    res.copy_to_host(&val);
    
    check_float(val, (float)n);
    std::cout << "PASSED" << std::endl;
#else
    std::cout << "Skipping SumHIPLarge (HIP disabled)" << std::endl;
#endif
}

void test_sum_hip_odd() {
#ifdef USE_HIP_BACKEND
    std::cout << "Running SumHIPOdd..." << std::endl;
    int n = 12345;
    auto t = vesper::full({n}, DType::Float32, Device::HIP, 1.0f);
    auto res = ops::sum(t);
    
    float val = 0.0f;
    res.copy_to_host(&val);
    
    check_float(val, (float)n);
    std::cout << "PASSED" << std::endl;
#else
    std::cout << "Skipping SumHIPOdd (HIP disabled)" << std::endl;
#endif
}

void test_sum_random() {
#ifdef USE_HIP_BACKEND
    std::cout << "Running SumRandom..." << std::endl;
    int n = 10000;
    std::vector<float> h_data(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : h_data) x = dist(rng);

    // Reference
    auto ref_t = vesper::empty({n}, DType::Float32, Device::CPU);
    ref_t.copy_from_host(h_data.data());
    auto ref_res = vesper::empty({1}, DType::Float32, Device::CPU);
    vesper::reference::sum(ref_t, ref_res);
    float expected = *ref_res.data_ptr<float>();

    // HIP
    auto d_t = vesper::empty({n}, DType::Float32, Device::HIP);
    d_t.copy_from_host(h_data.data());
    auto d_res = ops::sum(d_t);
    float actual = 0.0f;
    d_res.copy_to_host(&actual);

    // Sum accumulation can have larger errors
    check_float(actual, expected, 1e-2);
    std::cout << "PASSED" << std::endl;
#else
    std::cout << "Skipping SumRandom (HIP disabled)" << std::endl;
#endif
}

int main() {
    test_sum_cpu();
    test_sum_hip();
    test_sum_hip_large();
    test_sum_hip_odd();
    test_sum_random();
    return 0;
}
