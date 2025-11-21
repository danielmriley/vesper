#include <vesper/nn/linear.h>
#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include "grad_check_utils.h"
#include <iostream>
#include <cassert>

void test_linear_grad_finite_diff() {
    std::cout << "Testing Linear layer gradient with finite differences..." << std::endl;
    
    // Setup
    int in_features = 3;
    int out_features = 2;
    auto linear = std::make_shared<vesper::nn::Linear>(in_features, out_features);
    
    // Fix weights for deterministic test
    auto w_data = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}; // 2x3
    linear->weight.copy_from_host(w_data.data());
    auto b_data = std::vector<float>{0.1f, -0.1f}; // 2
    linear->bias.copy_from_host(b_data.data());

    auto input = vesper::full({1, in_features}, vesper::DType::Float32, vesper::Device::CPU, 1.0f);
    input.set_requires_grad(true);

    // Define loss function: sum(linear(x))
    auto loss_fn = [&](const vesper::Tensor& x) -> float {
        auto y = linear->forward(x);
        auto loss = vesper::ops::sum(y);
        float val;
        loss.copy_to_host(&val);
        return val;
    };

    // 1. Analytical Gradient (Backward)
    linear->zero_grad();
    auto output = linear->forward(input);
    auto loss = vesper::ops::sum(output);
    loss.backward();

    // 2. Numerical Gradient for Input
    // We need a wrapper that takes input tensor and returns loss, which we have.
    // But we need to pass a copy of input to avoid modifying the graph? 
    // compute_numerical_gradient modifies the tensor in place and restores it.
    auto num_grad_input = vesper::test::compute_numerical_gradient(
        [&](const vesper::Tensor& t) { 
            // We need to run forward on the perturbed input
            // Note: linear layer parameters are fixed here
            auto y = linear->forward(t);
            auto l = vesper::ops::sum(y);
            float v;
            l.copy_to_host(&v);
            return v;
        }, 
        input
    );

    if (vesper::test::check_gradients(input.grad(), num_grad_input)) {
        std::cout << "Input gradient check passed." << std::endl;
    } else {
        std::cerr << "Input gradient check FAILED." << std::endl;
        exit(1);
    }

    // 3. Numerical Gradient for Weights
    // We need to perturb linear->weight
    auto num_grad_weight = vesper::test::compute_numerical_gradient(
        [&](const vesper::Tensor& w) {
            // We can't easily swap the weight tensor in the module, 
            // but since w IS linear->weight (passed by ref), modifying it works.
            // However, compute_numerical_gradient takes `Tensor& input`.
            // We need to ensure the forward pass uses this modified tensor.
            // Since linear->weight is what we pass, it works.
            auto y = linear->forward(input);
            auto l = vesper::ops::sum(y);
            float v;
            l.copy_to_host(&v);
            return v;
        },
        linear->weight
    );

    if (vesper::test::check_gradients(linear->weight.grad(), num_grad_weight)) {
        std::cout << "Weight gradient check passed." << std::endl;
    } else {
        std::cerr << "Weight gradient check FAILED." << std::endl;
        exit(1);
    }

    std::cout << "Linear layer finite difference test passed!" << std::endl;
}

int main() {
    test_linear_grad_finite_diff();
    return 0;
}
