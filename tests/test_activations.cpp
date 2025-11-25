#include <vesper/nn/functional.h>
#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include <vesper/ops/random.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

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

void test_sigmoid(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing sigmoid activation on " << dev_str << "..." << std::endl;

    auto input = vesper::empty({2}, vesper::DType::Float32, device, true);
    std::vector<float> in_data = {0.0f, 2.0f};
    input.copy_from_host(in_data.data());

    // 1. Forward Pass Verification
    auto output = vesper::nn::functional::sigmoid(input);
    std::vector<float> out_data(2);
    output.copy_to_host(out_data.data());

    // y = 1 / (1 + exp(-x))
    assert(fabs(out_data[0] - 0.5f) < 1e-6); // sigmoid(0) = 0.5
    assert(fabs(out_data[1] - (1.0f / (1.0f + exp(-2.0f)))) < 1e-6);

    // 2. Backward Pass Verification
    auto loss = vesper::ops::sum(output);
    loss.backward();

    std::vector<float> in_grad(2);
    input.grad().copy_to_host(in_grad.data());

    // dy/dx = y * (1-y)
    // For x=0, y=0.5, grad=0.5*(1-0.5)=0.25
    // For x=2, y=0.88079, grad=0.88079*(1-0.88079)=0.10499
    assert(fabs(in_grad[0] - 0.25f) < 1e-6);
    assert(fabs(in_grad[1] - 0.104993585f) < 1e-4);

    std::cout << "Sigmoid activation test passed!" << std::endl;
}

void test_relu_correct_backward(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing ReLU with correct backward pass on " << dev_str << "..." << std::endl;

    auto input = vesper::empty({4}, vesper::DType::Float32, device, true);
    std::vector<float> in_data = {-1.0f, 0.5f, 1.0f, -2.0f};
    input.copy_from_host(in_data.data());

    auto output = vesper::nn::functional::relu(input);
    auto loss = vesper::ops::sum(output);
    loss.backward();

    std::vector<float> in_grad(4);
    input.grad().copy_to_host(in_grad.data());

    // Gradient should be 1.0 where input > 0, and 0.0 otherwise.
    assert(fabs(in_grad[0] - 0.0f) < 1e-6);
    assert(fabs(in_grad[1] - 1.0f) < 1e-6);
    assert(fabs(in_grad[2] - 1.0f) < 1e-6);
    assert(fabs(in_grad[3] - 0.0f) < 1e-6);

    std::cout << "ReLU correct backward test passed!" << std::endl;
}

void test_gelu(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing GELU activation on " << dev_str << "..." << std::endl;
    
    auto input = vesper::empty({3}, vesper::DType::Float32, device, true);
    std::vector<float> in_data = {0.0f, 1.0f, -1.0f};
    input.copy_from_host(in_data.data());
    
    auto output = vesper::nn::functional::gelu(input);
    std::vector<float> out_data(3);
    output.copy_to_host(out_data.data());
    
    // GELU(0) = 0
    // GELU(1) approx 0.8413
    // GELU(-1) approx -0.1587
    
    assert(fabs(out_data[0] - 0.0f) < 1e-5);
    assert(fabs(out_data[1] - 0.84119f) < 1e-3); // Using approx formula
    assert(fabs(out_data[2] - (-0.1588f)) < 1e-3);
    
    std::cout << "GELU activation test passed!" << std::endl;
}

void test_gelu_backward(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing GELU backward on " << dev_str << "..." << std::endl;
    
    auto input = vesper::empty({3}, vesper::DType::Float32, device, true);
    std::vector<float> in_data = {0.0f, 1.0f, -1.0f};
    input.copy_from_host(in_data.data());
    
    auto output = vesper::nn::functional::gelu(input);
    auto loss = vesper::ops::sum(output);
    loss.backward();
    
    std::vector<float> in_grad(3);
    input.grad().copy_to_host(in_grad.data());
    
    // d/dx GELU(x)
    // x=0: 0.5
    // x=1: approx 1.08
    // x=-1: approx -0.08
    
    assert(fabs(in_grad[0] - 0.5f) < 1e-5);
    assert(fabs(in_grad[1] - 1.083f) < 1e-2);
    assert(fabs(in_grad[2] - (-0.083f)) < 1e-2);
    
    std::cout << "GELU backward test passed!" << std::endl;
}

void test_gelu_variants(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing GELU variants on " << dev_str << "..." << std::endl;
    
    auto input = vesper::empty({3}, vesper::DType::Float32, device, true);
    std::vector<float> in_data = {0.0f, 1.0f, -1.0f};
    input.copy_from_host(in_data.data());
    
    auto out_tanh = vesper::nn::functional::gelu_tanh(input);
    auto out_erf = vesper::nn::functional::gelu_erf(input);
    
    // Currently they map to the same implementation, so they should be identical
    std::vector<float> d_tanh(3);
    std::vector<float> d_erf(3);
    out_tanh.copy_to_host(d_tanh.data());
    out_erf.copy_to_host(d_erf.data());
    
    for(int i=0; i<3; ++i) {
        assert(fabs(d_tanh[i] - d_erf[i]) < 1e-6);
    }
    
    std::cout << "GELU variants test passed!" << std::endl;
}

void test_activations_consistency() {
    std::cout << "Testing Activations Consistency..." << std::endl;
    auto input_cpu = vesper::empty({10, 20}, vesper::DType::Float32, vesper::Device::CPU);
    vesper::ops::uniform_(input_cpu, -1.0f, 1.0f);
    
    // Sigmoid
    auto sig_cpu = vesper::nn::functional::sigmoid(input_cpu);
    // ReLU
    auto relu_cpu = vesper::nn::functional::relu(input_cpu);
    // GELU
    auto gelu_cpu = vesper::nn::functional::gelu(input_cpu);

#ifdef USE_CUDA_BACKEND
    {
        auto input_cuda = input_cpu.to(vesper::Device::CUDA);
        
        auto sig_cuda = vesper::nn::functional::sigmoid(input_cuda);
        assert_tensors_close(sig_cpu, sig_cuda.to(vesper::Device::CPU));
        std::cout << "Sigmoid CPU vs CUDA passed!" << std::endl;
        
        auto relu_cuda = vesper::nn::functional::relu(input_cuda);
        assert_tensors_close(relu_cpu, relu_cuda.to(vesper::Device::CPU));
        std::cout << "ReLU CPU vs CUDA passed!" << std::endl;
        
        auto gelu_cuda = vesper::nn::functional::gelu(input_cuda);
        assert_tensors_close(gelu_cpu, gelu_cuda.to(vesper::Device::CPU));
        std::cout << "GELU CPU vs CUDA passed!" << std::endl;
    }
#endif

#ifdef USE_HIP_BACKEND
    {
        auto input_hip = input_cpu.to(vesper::Device::HIP);
        
        auto sig_hip = vesper::nn::functional::sigmoid(input_hip);
        assert_tensors_close(sig_cpu, sig_hip.to(vesper::Device::CPU));
        std::cout << "Sigmoid CPU vs HIP passed!" << std::endl;
        
        auto relu_hip = vesper::nn::functional::relu(input_hip);
        assert_tensors_close(relu_cpu, relu_hip.to(vesper::Device::CPU));
        std::cout << "ReLU CPU vs HIP passed!" << std::endl;
        
        auto gelu_hip = vesper::nn::functional::gelu(input_hip);
        assert_tensors_close(gelu_cpu, gelu_hip.to(vesper::Device::CPU));
        std::cout << "GELU CPU vs HIP passed!" << std::endl;
    }
#endif
}

int main() {
    test_sigmoid(vesper::Device::CPU);
    test_relu_correct_backward(vesper::Device::CPU);
    test_gelu(vesper::Device::CPU);
    test_gelu_backward(vesper::Device::CPU);
    test_gelu_variants(vesper::Device::CPU);
    
    test_activations_consistency();
    
#ifdef USE_CUDA_BACKEND
    try {
        test_sigmoid(vesper::Device::CUDA);
        test_relu_correct_backward(vesper::Device::CUDA);
        test_gelu(vesper::Device::CUDA);
        test_gelu_variants(vesper::Device::CUDA);
    } catch (const std::exception& e) {
        std::cerr << "CUDA test failed: " << e.what() << std::endl;
        return 1;
    }
#endif

#ifdef USE_HIP_BACKEND
    try {
        test_sigmoid(vesper::Device::HIP);
        test_relu_correct_backward(vesper::Device::HIP);
        test_gelu(vesper::Device::HIP);
        test_gelu_variants(vesper::Device::HIP);
    } catch (const std::exception& e) {
        std::cerr << "HIP test failed: " << e.what() << std::endl;
        return 1;
    }
#endif
    return 0;
}
