#include <vesper/nn/linear.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>

void test_linear_layer_forward() {
    std::cout << "Testing nn::Linear layer forward pass..." << std::endl;

    const int64_t in_f = 16, out_f = 8, batch_size = 4;
    
    // 1. Create the layer
    // Note: We disable bias for now because ops::add does not support broadcasting yet.
    auto linear_layer = std::make_shared<vesper::nn::Linear>(in_f, out_f, false);

    // 2. Check that parameters were registered
    auto params = linear_layer->parameters();
    assert(params.size() == 1); // weight only (bias disabled)
    assert(params[0]->shape() == std::vector<int64_t>({out_f, in_f})); // Correct weight shape

    // 3. Create a dummy input tensor
    auto input = vesper::zeros({batch_size, in_f}, vesper::DType::Float32, vesper::Device::CPU);

    // 4. Perform the forward pass
    auto output = linear_layer->forward(input);

    // 5. Check the output shape
    assert(output.shape() == std::vector<int64_t>({batch_size, out_f}));

    std::cout << "nn::Linear layer forward test passed!" << std::endl;
}

int main() {
    test_linear_layer_forward();
    return 0;
}
