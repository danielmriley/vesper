#include <vesper/nn/normalization.h>
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

void test_layer_norm(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing LayerNorm on " << dev_str << "..." << std::endl;
    
    // Shape: [2, 5]
    // Normalized shape: [5]
    auto ln = vesper::nn::LayerNorm({5});
    
    if (device != vesper::Device::CPU) {
        if (ln.weight.defined()) ln.weight = ln.weight.to(device);
        if (ln.bias.defined()) ln.bias = ln.bias.to(device);
    }
    
    std::vector<float> data = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,  // Mean=3, Var=2, Std=1.414
        10.0f, 20.0f, 30.0f, 40.0f, 50.0f // Mean=30, Var=200, Std=14.14
    };
    
    auto input = vesper::empty({2, 5}, vesper::DType::Float32, device);
    input.copy_from_host(data.data());
    
    auto output = ln.forward(input);
    
    std::vector<float> out_data(10);
    output.copy_to_host(out_data.data());
    
    // Check first row
    // (1-3)/1.414 = -1.414
    // (2-3)/1.414 = -0.707
    // (3-3)/1.414 = 0
    // (4-3)/1.414 = 0.707
    // (5-3)/1.414 = 1.414
    
    float eps = 1e-5;
    float std1 = std::sqrt(2.0f + eps);
    float std2 = std::sqrt(200.0f + eps);
    
    assert(std::abs(out_data[0] - (1.0f-3.0f)/std1) < 1e-4);
    assert(std::abs(out_data[2] - 0.0f) < 1e-4);
    assert(std::abs(out_data[4] - (5.0f-3.0f)/std1) < 1e-4);
    
    // Check second row
    assert(std::abs(out_data[5] - (10.0f-30.0f)/std2) < 1e-4);
    
    std::cout << "LayerNorm passed!" << std::endl;
}

void test_rms_norm(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing RMSNorm on " << dev_str << "..." << std::endl;
    
    auto rms = vesper::nn::RMSNorm({2});
    
    if (device != vesper::Device::CPU) {
        if (rms.weight.defined()) rms.weight = rms.weight.to(device);
    }
    
    std::vector<float> data = {3.0f, 4.0f}; // RMS = sqrt((9+16)/2) = sqrt(12.5) = 3.5355
    auto input = vesper::empty({1, 2}, vesper::DType::Float32, device);
    input.copy_from_host(data.data());
    
    auto output = rms.forward(input);
    
    std::vector<float> out_data(2);
    output.copy_to_host(out_data.data());
    
    float val = std::sqrt(12.5f + 1e-5f);
    assert(std::abs(out_data[0] - 3.0f/val) < 1e-4);
    assert(std::abs(out_data[1] - 4.0f/val) < 1e-4);
    
    std::cout << "RMSNorm passed!" << std::endl;
}

void test_layer_norm_consistency() {
    std::cout << "Testing LayerNorm Consistency..." << std::endl;
    auto input_cpu = vesper::empty({10, 20}, vesper::DType::Float32, vesper::Device::CPU);
    vesper::ops::uniform_(input_cpu, -1.0f, 1.0f);
    
    auto ln_cpu = vesper::nn::LayerNorm({20});
    auto out_cpu = ln_cpu.forward(input_cpu);

#ifdef USE_CUDA_BACKEND
    {
        auto input_cuda = input_cpu.to(vesper::Device::CUDA);
        auto ln_cuda = vesper::nn::LayerNorm({20});
        ln_cuda.weight = ln_cpu.weight.to(vesper::Device::CUDA);
        ln_cuda.bias = ln_cpu.bias.to(vesper::Device::CUDA);
        
        auto out_cuda = ln_cuda.forward(input_cuda);
        assert_tensors_close(out_cpu, out_cuda.to(vesper::Device::CPU));
        std::cout << "LayerNorm CPU vs CUDA passed!" << std::endl;
    }
#endif

#ifdef USE_HIP_BACKEND
    {
        auto input_hip = input_cpu.to(vesper::Device::HIP);
        auto ln_hip = vesper::nn::LayerNorm({20});
        ln_hip.weight = ln_cpu.weight.to(vesper::Device::HIP);
        ln_hip.bias = ln_cpu.bias.to(vesper::Device::HIP);
        
        auto out_hip = ln_hip.forward(input_hip);
        assert_tensors_close(out_cpu, out_hip.to(vesper::Device::CPU));
        std::cout << "LayerNorm CPU vs HIP passed!" << std::endl;
    }
#endif
}

void test_rms_norm_consistency() {
    std::cout << "Testing RMSNorm Consistency..." << std::endl;
    auto input_cpu = vesper::empty({10, 20}, vesper::DType::Float32, vesper::Device::CPU);
    vesper::ops::uniform_(input_cpu, -1.0f, 1.0f);
    
    auto rms_cpu = vesper::nn::RMSNorm({20});
    auto out_cpu = rms_cpu.forward(input_cpu);

#ifdef USE_CUDA_BACKEND
    {
        auto input_cuda = input_cpu.to(vesper::Device::CUDA);
        auto rms_cuda = vesper::nn::RMSNorm({20});
        rms_cuda.weight = rms_cpu.weight.to(vesper::Device::CUDA);
        
        auto out_cuda = rms_cuda.forward(input_cuda);
        assert_tensors_close(out_cpu, out_cuda.to(vesper::Device::CPU));
        std::cout << "RMSNorm CPU vs CUDA passed!" << std::endl;
    }
#endif

#ifdef USE_HIP_BACKEND
    {
        auto input_hip = input_cpu.to(vesper::Device::HIP);
        auto rms_hip = vesper::nn::RMSNorm({20});
        rms_hip.weight = rms_cpu.weight.to(vesper::Device::HIP);
        
        auto out_hip = rms_hip.forward(input_hip);
        assert_tensors_close(out_cpu, out_hip.to(vesper::Device::CPU));
        std::cout << "RMSNorm CPU vs HIP passed!" << std::endl;
    }
#endif
}

int main() {
    test_layer_norm(vesper::Device::CPU);
    test_rms_norm(vesper::Device::CPU);
    
    test_layer_norm_consistency();
    test_rms_norm_consistency();
    
#ifdef USE_CUDA_BACKEND
    try {
        test_layer_norm(vesper::Device::CUDA);
        test_rms_norm(vesper::Device::CUDA);
    } catch (const std::exception& e) {
        std::cerr << "CUDA test failed: " << e.what() << std::endl;
        // Don't fail the whole test suite if CUDA is not present at runtime but compiled in?
        // But user said "We are on a Nvidia GPU system". So we should fail.
        return 1;
    }
#endif

#ifdef USE_HIP_BACKEND
    try {
        test_layer_norm(vesper::Device::HIP);
        test_rms_norm(vesper::Device::HIP);
    } catch (const std::exception& e) {
        std::cerr << "HIP test failed: " << e.what() << std::endl;
        return 1;
    }
#endif
    return 0;
}
