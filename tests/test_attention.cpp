#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/nn/functional.h>
#include <vesper/ops/random.h>
#include <vesper/ops/elementwise.h>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace vesper;

void test_attention_shapes() {
    std::cout << "Testing Attention Shapes..." << std::endl;
    
    int B = 2;
    int H = 4;
    int S = 32;
    int D = 64;
    
    Device device = Device::CPU;
#if USE_CUDA_BACKEND
    device = Device::CUDA;
#endif

    Tensor Q = vesper::empty({B, H, S, D}, DType::Float32, device);
    Tensor K = vesper::empty({B, H, S, D}, DType::Float32, device);
    Tensor V = vesper::empty({B, H, S, D}, DType::Float32, device);
    
    ops::normal_(Q, 0.0f, 1.0f);
    ops::normal_(K, 0.0f, 1.0f);
    ops::normal_(V, 0.0f, 1.0f);
    
    Tensor out = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    
    assert(out.ndim() == 4);
    assert(out.shape()[0] == B);
    assert(out.shape()[1] == H);
    assert(out.shape()[2] == S);
    assert(out.shape()[3] == D);
    
    std::cout << "Attention Shapes Passed!" << std::endl;
}

void test_causality() {
    std::cout << "Testing Attention Causality..." << std::endl;
    
    int B = 1;
    int H = 1;
    int S = 4;
    int D = 4;
    
    Device device = Device::CPU;
#if USE_CUDA_BACKEND
    device = Device::CUDA;
#endif

    Tensor Q = vesper::empty({B, H, S, D}, DType::Float32, device);
    Tensor K = vesper::empty({B, H, S, D}, DType::Float32, device);
    Tensor V = vesper::empty({B, H, S, D}, DType::Float32, device);
    
    ops::normal_(Q, 0.0f, 1.0f);
    ops::normal_(K, 0.0f, 1.0f);
    ops::normal_(V, 0.0f, 1.0f);
    
    // Run 1
    Tensor out1 = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    
    // Modify K and V at the last position (S-1)
    // This should NOT affect output at position 0 (if S > 1)
    
    // We need to modify the tensor. Since we don't have easy indexing setter yet,
    // we can copy to CPU, modify, copy back.
    Tensor K_cpu = K.to(Device::CPU);
    float* k_ptr = K_cpu.data_ptr<float>();
    // Modify last token's key
    // Index: (0, 0, S-1, 0)
    k_ptr[(S-1) * D] += 100.0f; 
    K.copy_from(K_cpu);
    
    // Run 2
    Tensor out2 = nn::functional::scaled_dot_product_attention(Q, K, V, true);
    
    // Check that out1[0, 0, 0, :] == out2[0, 0, 0, :]
    // Because token 0 can only attend to token 0. Changes in token S-1 should be masked.
    
    Tensor out1_cpu = out1.to(Device::CPU);
    Tensor out2_cpu = out2.to(Device::CPU);
    
    const float* p1 = out1_cpu.data_ptr<float>();
    const float* p2 = out2_cpu.data_ptr<float>();
    
    // Check first token output (size D)
    for (int i = 0; i < D; ++i) {
        float diff = std::abs(p1[i] - p2[i]);
        if (diff > 1e-5) {
            std::cerr << "Causality check failed at index " << i << ". Diff: " << diff << std::endl;
            std::cerr << "Value 1: " << p1[i] << ", Value 2: " << p2[i] << std::endl;
            assert(false);
        }
    }
    
    std::cout << "Attention Causality Passed!" << std::endl;
}

void test_attention_cpu_vs_gpu() {
    std::cout << "Testing Attention CPU vs GPU..." << std::endl;

#if USE_CUDA_BACKEND
    int B = 2;
    int H = 4;
    int S = 32;
    int D = 64;

    // Create inputs on CPU
    Tensor Q_cpu = vesper::empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor K_cpu = vesper::empty({B, H, S, D}, DType::Float32, Device::CPU);
    Tensor V_cpu = vesper::empty({B, H, S, D}, DType::Float32, Device::CPU);

    ops::normal_(Q_cpu, 0.0f, 1.0f);
    ops::normal_(K_cpu, 0.0f, 1.0f);
    ops::normal_(V_cpu, 0.0f, 1.0f);

    // Run on CPU
    Tensor out_cpu = nn::functional::scaled_dot_product_attention(Q_cpu, K_cpu, V_cpu, true);

    // Run on GPU
    Tensor Q_gpu = Q_cpu.to(Device::CUDA);
    Tensor K_gpu = K_cpu.to(Device::CUDA);
    Tensor V_gpu = V_cpu.to(Device::CUDA);

    Tensor out_gpu = nn::functional::scaled_dot_product_attention(Q_gpu, K_gpu, V_gpu, true);
    
    // Compare
    Tensor out_gpu_cpu = out_gpu.to(Device::CPU);
    
    const float* p_cpu = out_cpu.data_ptr<float>();
    const float* p_gpu = out_gpu_cpu.data_ptr<float>();
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < out_cpu.numel(); ++i) {
        float diff = std::abs(p_cpu[i] - p_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    std::cout << "Max difference CPU vs GPU: " << max_diff << std::endl;
    
    // Softmax and exp can accumulate errors, so tolerance might need to be slightly loose
    if (max_diff > 1e-4) {
        std::cerr << "CPU vs GPU mismatch!" << std::endl;
        assert(false);
    }

    std::cout << "Attention CPU vs GPU Passed!" << std::endl;
#else
    std::cout << "CUDA not enabled, skipping CPU vs GPU test." << std::endl;
#endif
}

int main() {
    try {
        test_attention_shapes();
        test_causality();
        test_attention_cpu_vs_gpu();
        std::cout << "All Attention tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
