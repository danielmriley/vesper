#include <vesper/nn/transformer.h>
#include <vesper/core/factories.h>
#include <vesper/ops/random.h>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace vesper;

void test_transformer_block_shape() {
    std::cout << "Testing Transformer Block Shape..." << std::endl;
    int B=2, S=10, E=32, H=4;
    nn::TransformerBlock block(E, H);
    
    Tensor x = vesper::randn({B, S, E}, DType::Float32, Device::CPU);
    Tensor out = block.forward(x);
    
    assert(out.shape()[0] == B);
    assert(out.shape()[1] == S);
    assert(out.shape()[2] == E);
    
    std::cout << "Transformer Block Shape Passed!" << std::endl;
}

void test_transformer_block_causal() {
    std::cout << "Testing Transformer Block Causal..." << std::endl;
    int B=1, S=4, E=4, H=1;
    nn::TransformerBlock block(E, H);
    
    Tensor x = vesper::randn({B, S, E}, DType::Float32, Device::CPU);
    
    // Run causal
    Tensor out1 = block.forward(x, true);
    
    // Modify last token of input
    Tensor x2 = x.clone();
    float* ptr = x2.data_ptr<float>();
    ptr[(S-1)*E] += 10.0f;
    
    Tensor out2 = block.forward(x2, true);
    
    // First token output should be same
    const float* p1 = out1.data_ptr<float>();
    const float* p2 = out2.data_ptr<float>();
    
    for(int i=0; i<E; ++i) {
        assert(std::abs(p1[i] - p2[i]) < 1e-5);
    }
    
    std::cout << "Transformer Block Causal Passed!" << std::endl;
}

int main() {
    test_transformer_block_shape();
    test_transformer_block_causal();
    return 0;
}
