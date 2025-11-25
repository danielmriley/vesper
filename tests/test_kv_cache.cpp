#include <vesper/nn/transformer.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>

using namespace vesper;

void test_kv_cache_update() {
    std::cout << "Testing KV Cache Update..." << std::endl;
    int B=1, H=1, S=10, D=4;
    nn::KVCache cache(B, H, S, D, Device::CPU);
    
    Tensor k = vesper::ones({B, H, 2, D}, DType::Float32, Device::CPU);
    Tensor v = vesper::full({B, H, 2, D}, DType::Float32, Device::CPU, 2.0f);
    
    auto [k_out, v_out] = cache.update(k, v, 0);
    
    assert(cache.current_seq_len() == 2);
    assert(k_out.shape()[2] == 2);
    assert(v_out.shape()[2] == 2);
    
    // Check values
    const float* k_ptr = k_out.data_ptr<float>();
    assert(k_ptr[0] == 1.0f);
    
    // Update again
    Tensor k2 = vesper::full({B, H, 1, D}, DType::Float32, Device::CPU, 3.0f);
    Tensor v2 = vesper::full({B, H, 1, D}, DType::Float32, Device::CPU, 4.0f);
    
    auto [k_out2, v_out2] = cache.update(k2, v2, 2);
    
    assert(cache.current_seq_len() == 3);
    assert(k_out2.shape()[2] == 3);
    
    // Check values at pos 2
    // k_out2 is [B, H, 3, D]
    // Index (0,0,2,0)
    // stride for dim 2 is D=4
    const float* k_ptr2 = k_out2.data_ptr<float>();
    assert(k_ptr2[2*4] == 3.0f);
    
    std::cout << "KV Cache Update Passed!" << std::endl;
}

void test_kv_cache_reset() {
    std::cout << "Testing KV Cache Reset..." << std::endl;
    nn::KVCache cache(1, 1, 10, 4, Device::CPU);
    Tensor k = vesper::ones({1, 1, 2, 4}, DType::Float32, Device::CPU);
    Tensor v = vesper::ones({1, 1, 2, 4}, DType::Float32, Device::CPU);
    
    cache.update(k, v, 0);
    assert(cache.current_seq_len() == 2);
    
    cache.reset();
    assert(cache.current_seq_len() == 0);
    
    std::cout << "KV Cache Reset Passed!" << std::endl;
}

int main() {
    test_kv_cache_update();
    test_kv_cache_reset();
    return 0;
}
