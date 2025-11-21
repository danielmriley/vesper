#include <vesper/nn/functional.h>
#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

void test_sigmoid() {
    std::cout << "Testing sigmoid activation..." << std::endl;
    auto device = vesper::Device::CPU;

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

void test_relu_correct_backward() {
    std::cout << "Testing ReLU with correct backward pass..." << std::endl;
    auto device = vesper::Device::CPU;

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

int main() {
    test_sigmoid();
    test_relu_correct_backward();
    return 0;
}
