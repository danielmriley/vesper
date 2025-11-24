#include <vesper/nn/init.h>
#include <vesper/core/factories.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#if defined(USE_HIP_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CUDA;
#elif defined(USE_CPU_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CPU;
#else
    #error "No backend enabled for testing"
#endif

void verify_mean_std(const vesper::Tensor& t, float expected_mean, float expected_std, float tol_mean=0.1f, float tol_std=0.1f) {
    // Calculate mean and std manually on host
    std::vector<float> data(t.numel());
    t.copy_to_host(data.data());
    
    double sum = 0.0;
    for (float x : data) sum += x;
    double mean = sum / data.size();
    
    double sq_sum = 0.0;
    for (float x : data) sq_sum += (x - mean) * (x - mean);
    double std = std::sqrt(sq_sum / data.size());
    
    std::cout << "Mean: " << mean << " (exp: " << expected_mean << "), Std: " << std << " (exp: " << expected_std << ")" << std::endl;
    
    assert(std::fabs(mean - expected_mean) < tol_mean);
    assert(std::fabs(std - expected_std) < tol_std);
}

void test_uniform() {
    std::cout << "Testing uniform_..." << std::endl;
    auto t = vesper::empty({10000}, vesper::DType::Float32, TEST_DEVICE);
    vesper::nn::init::uniform_(t, -1.0f, 1.0f);
    
    // Mean 0, Std for U[-1, 1] = range/sqrt(12) = 2/sqrt(12) = 1/sqrt(3) ~= 0.577
    verify_mean_std(t, 0.0f, 0.577f);
    std::cout << "uniform_ passed!" << std::endl;
}

void test_normal() {
    std::cout << "Testing normal_..." << std::endl;
    auto t = vesper::empty({10000}, vesper::DType::Float32, TEST_DEVICE);
    vesper::nn::init::normal_(t, 0.0f, 1.0f);
    
    verify_mean_std(t, 0.0f, 1.0f);
    std::cout << "normal_ passed!" << std::endl;
}

void test_kaiming_uniform() {
    std::cout << "Testing kaiming_uniform_ (ReLU)..." << std::endl;
    // Linear layer equivalent: [100, 100]
    // fan_in = 100.
    // bound = sqrt(3) * gain / sqrt(fan_in) = sqrt(3) * sqrt(2) / 10 = sqrt(6)/10 ~= 0.245
    // range [-0.245, 0.245].
    // std = 0.245 / sqrt(3) = sqrt(2)/10 ~= 0.1414
    
    auto t = vesper::empty({100, 100}, vesper::DType::Float32, TEST_DEVICE);
    vesper::nn::init::kaiming_uniform_(t, 0.0f, "fan_in", "relu");
    
    verify_mean_std(t, 0.0f, std::sqrt(2.0f/100.0f));
    std::cout << "kaiming_uniform_ passed!" << std::endl;
}

void test_xavier_normal() {
    std::cout << "Testing xavier_normal_ (Sigmoid)..." << std::endl;
    // Linear [100, 100]. fan_in=100, fan_out=100.
    // std = gain * sqrt(2 / (fan_in + fan_out)) = 1 * sqrt(2/200) = 0.1
    
    auto t = vesper::empty({100, 100}, vesper::DType::Float32, TEST_DEVICE);
    vesper::nn::init::xavier_normal_(t);
    
    verify_mean_std(t, 0.0f, 0.1f);
    std::cout << "xavier_normal_ passed!" << std::endl;
}

int main() {
    test_uniform();
    test_normal();
    test_kaiming_uniform();
    test_xavier_normal();
    return 0;
}
