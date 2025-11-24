#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/random.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

void assert_tensors_close(const vesper::Tensor& t1, const vesper::Tensor& t2, float tol = 1e-4f) {
    assert(t1.numel() == t2.numel());
    std::vector<float> d1(t1.numel());
    std::vector<float> d2(t2.numel());
    t1.copy_to_host(d1.data());
    t2.copy_to_host(d2.data());
    for(size_t i=0; i<t1.numel(); ++i) {
        if (std::abs(d1[i] - d2[i]) > tol) {
            std::cerr << "Mismatch at " << i << ": " << d1[i] << " vs " << d2[i] << std::endl;
            assert(false);
        }
    }
}

void test_softmax(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing Softmax on " << dev_str << "..." << std::endl;
    
    // Shape: [2, 3]
    auto input = vesper::empty({2, 3}, vesper::DType::Float32, device);
    std::vector<float> data = {
        1.0f, 2.0f, 3.0f,
        10.0f, 10.0f, 10.0f
    };
    input.copy_from_host(data.data());
    
    // Dim 1
    auto output = vesper::nn::functional::softmax(input, 1);
    
    std::vector<float> out_data(6);
    output.copy_to_host(out_data.data());
    
    // Row 1: exp(1), exp(2), exp(3) -> sum approx 2.718 + 7.389 + 20.085 = 30.19
    // 2.718/30.19 = 0.09
    // 7.389/30.19 = 0.24
    // 20.085/30.19 = 0.66
    
    float sum1 = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
    assert(std::abs(out_data[0] - std::exp(1.0f)/sum1) < 1e-4);
    assert(std::abs(out_data[1] - std::exp(2.0f)/sum1) < 1e-4);
    assert(std::abs(out_data[2] - std::exp(3.0f)/sum1) < 1e-4);
    
    // Row 2: exp(10), exp(10), exp(10) -> equal prob -> 0.333
    assert(std::abs(out_data[3] - 1.0f/3.0f) < 1e-4);
    assert(std::abs(out_data[4] - 1.0f/3.0f) < 1e-4);
    assert(std::abs(out_data[5] - 1.0f/3.0f) < 1e-4);
    
    // Sum check
    assert(std::abs(out_data[0]+out_data[1]+out_data[2] - 1.0f) < 1e-5);
    assert(std::abs(out_data[3]+out_data[4]+out_data[5] - 1.0f) < 1e-5);
    
    std::cout << "Softmax passed!" << std::endl;
}

void test_softmax_consistency() {
    std::cout << "Testing Softmax Consistency..." << std::endl;
    auto input_cpu = vesper::empty({10, 20}, vesper::DType::Float32, vesper::Device::CPU);
    vesper::ops::uniform_(input_cpu, -1.0f, 1.0f);
    
    auto out_cpu = vesper::nn::functional::softmax(input_cpu, 1);

#ifdef USE_CUDA_BACKEND
    {
        auto input_cuda = input_cpu.to(vesper::Device::CUDA);
        auto out_cuda = vesper::nn::functional::softmax(input_cuda, 1);
        assert_tensors_close(out_cpu, out_cuda.to(vesper::Device::CPU));
        std::cout << "Softmax CPU vs CUDA passed!" << std::endl;
    }
#endif

#ifdef USE_HIP_BACKEND
    {
        auto input_hip = input_cpu.to(vesper::Device::HIP);
        auto out_hip = vesper::nn::functional::softmax(input_hip, 1);
        assert_tensors_close(out_cpu, out_hip.to(vesper::Device::CPU));
        std::cout << "Softmax CPU vs HIP passed!" << std::endl;
    }
#endif
}

int main() {
    test_softmax(vesper::Device::CPU);
    test_softmax_consistency();
#ifdef USE_CUDA_BACKEND
    try {
        test_softmax(vesper::Device::CUDA);
    } catch (const std::exception& e) {
        std::cerr << "CUDA test failed: " << e.what() << std::endl;
        return 1;
    }
#endif

#ifdef USE_HIP_BACKEND
    try {
        test_softmax(vesper::Device::HIP);
    } catch (const std::exception& e) {
        std::cerr << "HIP test failed: " << e.what() << std::endl;
        return 1;
    }
#endif
    return 0;
}
