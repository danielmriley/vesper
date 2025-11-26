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

void test_layer_norm_no_affine(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing LayerNorm (no affine) on " << dev_str << "..." << std::endl;
    
    auto ln = vesper::nn::LayerNorm({5}, 1e-5, false);
    
    assert(!ln.weight.defined());
    assert(!ln.bias.defined());
    
    auto input = vesper::empty({2, 5}, vesper::DType::Float32, device);
    vesper::ops::uniform_(input, 0.0f, 1.0f);
    
    auto output = ln.forward(input);
    
    // Just check it runs and output shape is correct
    assert(output.shape()[0] == 2);
    assert(output.shape()[1] == 5);
    
    std::cout << "LayerNorm (no affine) passed!" << std::endl;
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

void test_layer_norm_backward(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing LayerNorm Backward on " << dev_str << "..." << std::endl;
    
    auto ln = vesper::nn::LayerNorm({2});
    if (device != vesper::Device::CPU) {
        ln.weight = ln.weight.to(device);
        ln.bias = ln.bias.to(device);
    }
    
    auto input = vesper::empty({2, 2}, vesper::DType::Float32, device, true);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    input.copy_from_host(data.data());
    
    auto output = ln.forward(input);
    auto loss = vesper::ops::sum(output);
    loss.backward();
    
    std::vector<float> grad_data(4);
    input.grad().copy_to_host(grad_data.data());
    
    for (float g : grad_data) {
        assert(std::isfinite(g));
    }
    
    std::cout << "LayerNorm Backward passed!" << std::endl;
}

void test_rms_norm_backward(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing RMSNorm Backward on " << dev_str << "..." << std::endl;
    
    auto rms = vesper::nn::RMSNorm({2});
    if (device != vesper::Device::CPU) {
        rms.weight = rms.weight.to(device);
    }
    
    auto input = vesper::empty({2, 2}, vesper::DType::Float32, device, true);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    input.copy_from_host(data.data());
    
    auto output = rms.forward(input);
    auto loss = vesper::ops::sum(output);
    loss.backward();
    
    std::vector<float> grad_data(4);
    input.grad().copy_to_host(grad_data.data());
    
    for (float g : grad_data) {
        assert(std::isfinite(g));
    }
    
    std::cout << "RMSNorm Backward passed!" << std::endl;
}

void test_layer_norm_output_stats(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing LayerNorm Output Stats on " << dev_str << "..." << std::endl;
    
    int hidden = 100;
    auto ln = vesper::nn::LayerNorm({hidden});
    if (device != vesper::Device::CPU) {
        ln.weight = ln.weight.to(device);
        ln.bias = ln.bias.to(device);
    }
    
    auto input = vesper::empty({10, hidden}, vesper::DType::Float32, device);
    vesper::ops::uniform_(input, -10.0f, 10.0f);
    
    auto output = ln.forward(input);
    
    // Check mean and var for each row
    std::vector<float> out_data(10 * hidden);
    output.copy_to_host(out_data.data());
    
    for (int i = 0; i < 10; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < hidden; ++j) {
            sum += out_data[i * hidden + j];
        }
        float mean = sum / hidden;
        
        float sum_sq = 0.0f;
        for (int j = 0; j < hidden; ++j) {
            float diff = out_data[i * hidden + j] - mean;
            sum_sq += diff * diff;
        }
        float var = sum_sq / hidden;
        
        assert(std::abs(mean) < 1e-4);
        assert(std::abs(var - 1.0f) < 1e-3);
    }
    
    std::cout << "LayerNorm Output Stats passed!" << std::endl;
}

void test_layer_norm_affine(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing LayerNorm Affine (gamma=2, beta=1) on " << dev_str << "..." << std::endl;
    
    auto ln = vesper::nn::LayerNorm({4});
    
    // Set gamma=2, beta=1
    std::vector<float> gamma = {2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> beta = {1.0f, 1.0f, 1.0f, 1.0f};
    ln.weight.copy_from_host(gamma.data());
    ln.bias.copy_from_host(beta.data());
    
    if (device != vesper::Device::CPU) {
        ln.weight = ln.weight.to(device);
        ln.bias = ln.bias.to(device);
    }
    
    // Input: [1, 2, 3, 4] -> mean=2.5, var=1.25, std=1.118
    auto input = vesper::empty({1, 4}, vesper::DType::Float32, device);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    input.copy_from_host(data.data());
    
    auto output = ln.forward(input);
    
    std::vector<float> out_data(4);
    output.copy_to_host(out_data.data());
    
    // Expected: normalized * 2 + 1
    float eps = 1e-5f;
    float mean = 2.5f;
    float var = 1.25f;
    float std_val = std::sqrt(var + eps);
    
    for (int i = 0; i < 4; ++i) {
        float normalized = (data[i] - mean) / std_val;
        float expected = normalized * 2.0f + 1.0f;
        assert(std::abs(out_data[i] - expected) < 1e-4f);
    }
    
    std::cout << "LayerNorm Affine passed!" << std::endl;
}

void test_layer_norm_3d_input(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing LayerNorm 3D Input [B, S, H] on " << dev_str << "..." << std::endl;
    
    // Common transformer shape: [Batch, SeqLen, Hidden]
    int B = 2, S = 4, H = 8;
    auto ln = vesper::nn::LayerNorm({H});
    
    if (device != vesper::Device::CPU) {
        ln.weight = ln.weight.to(device);
        ln.bias = ln.bias.to(device);
    }
    
    auto input = vesper::empty({B, S, H}, vesper::DType::Float32, device);
    vesper::ops::uniform_(input, -1.0f, 1.0f);
    
    auto output = ln.forward(input);
    
    // Check output shape
    assert(output.shape()[0] == B);
    assert(output.shape()[1] == S);
    assert(output.shape()[2] == H);
    
    // Check mean/var for each position
    std::vector<float> out_data(B * S * H);
    output.copy_to_host(out_data.data());
    
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            float sum = 0.0f;
            for (int h = 0; h < H; ++h) {
                sum += out_data[b * S * H + s * H + h];
            }
            float mean = sum / H;
            
            float sum_sq = 0.0f;
            for (int h = 0; h < H; ++h) {
                float diff = out_data[b * S * H + s * H + h] - mean;
                sum_sq += diff * diff;
            }
            float var = sum_sq / H;
            
            assert(std::abs(mean) < 1e-4f);
            assert(std::abs(var - 1.0f) < 1e-2f);
        }
    }
    
    std::cout << "LayerNorm 3D Input passed!" << std::endl;
}

void test_rms_norm_vs_layer_norm(vesper::Device device) {
    std::string dev_str = (device == vesper::Device::CPU) ? "CPU" : 
                          (device == vesper::Device::CUDA) ? "CUDA" : "HIP";
    std::cout << "Testing RMSNorm vs LayerNorm on " << dev_str << "..." << std::endl;
    
    auto ln = vesper::nn::LayerNorm({4}, 1e-5f, false);  // No affine
    auto rms = vesper::nn::RMSNorm({4}, 1e-5f, false);
    
    // Zero-mean input: RMSNorm and LayerNorm should differ
    // because RMSNorm doesn't subtract mean
    auto input = vesper::empty({1, 4}, vesper::DType::Float32, device);
    std::vector<float> data = {-2.0f, -1.0f, 1.0f, 2.0f};  // mean=0
    input.copy_from_host(data.data());
    
    auto out_ln = ln.forward(input);
    auto out_rms = rms.forward(input);
    
    std::vector<float> ln_data(4), rms_data(4);
    out_ln.copy_to_host(ln_data.data());
    out_rms.copy_to_host(rms_data.data());
    
    // For zero-mean input:
    // LayerNorm: x / std(x) = x / sqrt(var(x))
    // RMSNorm: x / RMS(x) = x / sqrt(mean(x^2))
    // When mean=0: var(x) = mean(x^2), so they should be EQUAL
    
    // Non-zero mean input should differ
    std::vector<float> data2 = {1.0f, 2.0f, 3.0f, 4.0f};  // mean=2.5
    input.copy_from_host(data2.data());
    
    out_ln = ln.forward(input);
    out_rms = rms.forward(input);
    
    out_ln.copy_to_host(ln_data.data());
    out_rms.copy_to_host(rms_data.data());
    
    // They should differ because LayerNorm subtracts mean
    bool differs = false;
    for (int i = 0; i < 4; ++i) {
        if (std::abs(ln_data[i] - rms_data[i]) > 1e-4f) {
            differs = true;
            break;
        }
    }
    assert(differs && "RMSNorm and LayerNorm should differ for non-zero-mean input");
    
    std::cout << "RMSNorm vs LayerNorm passed!" << std::endl;
}

int main() {
    test_layer_norm(vesper::Device::CPU);
    test_rms_norm(vesper::Device::CPU);
    test_layer_norm_no_affine(vesper::Device::CPU);
    test_layer_norm_backward(vesper::Device::CPU);
    test_rms_norm_backward(vesper::Device::CPU);
    test_layer_norm_output_stats(vesper::Device::CPU);
    test_layer_norm_affine(vesper::Device::CPU);
    test_layer_norm_3d_input(vesper::Device::CPU);
    test_rms_norm_vs_layer_norm(vesper::Device::CPU);
    
    test_layer_norm_consistency();
    test_rms_norm_consistency();
    
#ifdef USE_CUDA_BACKEND
    try {
        test_layer_norm(vesper::Device::CUDA);
        test_rms_norm(vesper::Device::CUDA);
        test_layer_norm_no_affine(vesper::Device::CUDA);
        test_layer_norm_backward(vesper::Device::CUDA);
        test_rms_norm_backward(vesper::Device::CUDA);
        test_layer_norm_output_stats(vesper::Device::CUDA);
        test_layer_norm_affine(vesper::Device::CUDA);
        test_layer_norm_3d_input(vesper::Device::CUDA);
        test_rms_norm_vs_layer_norm(vesper::Device::CUDA);
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
        test_layer_norm_no_affine(vesper::Device::HIP);
        test_layer_norm_backward(vesper::Device::HIP);
        test_rms_norm_backward(vesper::Device::HIP);
        test_layer_norm_output_stats(vesper::Device::HIP);
        test_layer_norm_affine(vesper::Device::HIP);
        test_layer_norm_3d_input(vesper::Device::HIP);
        test_rms_norm_vs_layer_norm(vesper::Device::HIP);
    } catch (const std::exception& e) {
        std::cerr << "HIP test failed: " << e.what() << std::endl;
        return 1;
    }
#endif
    return 0;
}
