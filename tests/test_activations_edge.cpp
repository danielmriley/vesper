#include <vesper/nn/functional.h>
#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_relu_edge_cases() {
    std::cout << "Testing ReLU edge cases..." << std::endl;
    auto input = vesper::empty({3}, vesper::DType::Float32, vesper::Device::CPU, true);
    // -0.0f, 0.0f, epsilon
    std::vector<float> data = {-0.0f, 0.0f, 1e-6f};
    input.copy_from_host(data.data());

    auto output = vesper::nn::functional::relu(input);
    auto loss = vesper::ops::sum(output);
    loss.backward();

    std::vector<float> grad(3);
    input.grad().copy_to_host(grad.data());

    // Convention: > 0 is 1, <= 0 is 0.
    // -0.0f -> 0
    // 0.0f -> 0
    // 1e-6f -> 1
    assert(grad[0] == 0.0f);
    assert(grad[1] == 0.0f);
    assert(grad[2] == 1.0f);
    std::cout << "ReLU edge cases passed!" << std::endl;
}

void test_sigmoid_saturation() {
    std::cout << "Testing Sigmoid saturation..." << std::endl;
    auto input = vesper::empty({2}, vesper::DType::Float32, vesper::Device::CPU, true);
    std::vector<float> data = {100.0f, -100.0f};
    input.copy_from_host(data.data());

    auto output = vesper::nn::functional::sigmoid(input);
    auto loss = vesper::ops::sum(output);
    loss.backward();

    std::vector<float> grad(2);
    input.grad().copy_to_host(grad.data());
    std::vector<float> out_val(2);
    output.copy_to_host(out_val.data());

    // Output should be 1.0 and 0.0
    assert(std::abs(out_val[0] - 1.0f) < 1e-6);
    assert(std::abs(out_val[1] - 0.0f) < 1e-6);

    // Grad = y * (1-y). Should be ~0 for both.
    assert(std::abs(grad[0]) < 1e-6);
    assert(std::abs(grad[1]) < 1e-6);
    
    std::cout << "Sigmoid saturation passed!" << std::endl;
}

int main() {
    test_relu_edge_cases();
    test_sigmoid_saturation();
    return 0;
}
