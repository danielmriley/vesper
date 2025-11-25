#include <vesper/core/tensor.h>
#include <vesper/nn/functional.h>
#include <vesper/ops/elementwise.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vesper/core/factories.h>

using namespace vesper;

void test_rope_frequencies() {
    std::cout << "Testing RoPE Frequencies..." << std::endl;
    int seq_len = 4;
    int head_dim = 4;
    int start_pos = 0;
    
    Tensor freqs = nn::functional::compute_rope_frequencies(seq_len, head_dim, start_pos);
    
    // Expected:
    // dim = 4, half_dim = 2
    // theta_0 = 10000^(-0/4) = 1
    // theta_1 = 10000^(-2/4) = 10000^(-0.5) = 1/100 = 0.01
    
    // pos 0: [0, 0]
    // pos 1: [1, 0.01]
    // pos 2: [2, 0.02]
    // pos 3: [3, 0.03]
    
    Tensor freqs_cpu = freqs.to(Device::CPU);
    const float* ptr = freqs_cpu.data_ptr<float>();
    
    assert(std::abs(ptr[0] - 0.0f) < 1e-5);
    assert(std::abs(ptr[1] - 0.0f) < 1e-5);
    
    assert(std::abs(ptr[2] - 1.0f) < 1e-5);
    assert(std::abs(ptr[3] - 0.01f) < 1e-5);
    
    std::cout << "RoPE Frequencies Passed!" << std::endl;
}

void test_apply_rotary_emb() {
    std::cout << "Testing Apply RoPE..." << std::endl;
    int B=1, H=1, S=1, D=4;
    Tensor x = ones({B, H, S, D}, DType::Float32, Device::CPU);
    // x = [1, 1, 1, 1]
    // pairs: (1, 1), (1, 1)
    
    // Rotate by 0 (pos 0)
    Tensor freqs = nn::functional::compute_rope_frequencies(S, D, 0);
    Tensor out = nn::functional::apply_rotary_emb(x, freqs);
    
    // Should be same
    const float* ptr = out.data_ptr<float>();
    for(int i=0; i<4; ++i) assert(std::abs(ptr[i] - 1.0f) < 1e-5);
    
    // Rotate by 90 degrees (pi/2)
    // We need to manually construct freqs to be pi/2
    Tensor freqs_90 = full({S, D/2}, DType::Float32, Device::CPU, M_PI/2.0f);
    Tensor out_90 = nn::functional::apply_rotary_emb(x, freqs_90);
    
    // (1, 1) rotated by 90 deg:
    // x' = x cos - y sin = 1*0 - 1*1 = -1
    // y' = x sin + y cos = 1*1 + 1*0 = 1
    // Expected: [-1, 1, -1, 1]
    
    const float* ptr_90 = out_90.data_ptr<float>();
    assert(std::abs(ptr_90[0] - (-1.0f)) < 1e-5);
    assert(std::abs(ptr_90[1] - 1.0f) < 1e-5);
    assert(std::abs(ptr_90[2] - (-1.0f)) < 1e-5);
    assert(std::abs(ptr_90[3] - 1.0f) < 1e-5);
    
    std::cout << "Apply RoPE Passed!" << std::endl;
}

int main() {
    test_rope_frequencies();
    test_apply_rotary_emb();
    return 0;
}
