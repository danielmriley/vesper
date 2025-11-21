#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/loss.h>
#include <vesper/optim/sgd.h>
#include <vesper/autograd/engine.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

void test_linear_regression() {
    std::cout << "Testing Linear Regression Training Loop..." << std::endl;

    // Determine device
    vesper::Device device = vesper::Device::CPU;
#if defined(USE_HIP_BACKEND)
    device = vesper::Device::HIP;
    std::cout << "Device: HIP (GPU)" << std::endl;
#else
    std::cout << "Device: CPU" << std::endl;
#endif

    // 1. Data Generation
    // y = 2x + 1
    int64_t N = 100;
    auto x = vesper::full({N, 1}, vesper::DType::Float32, device, 0.0f);
    auto y = vesper::full({N, 1}, vesper::DType::Float32, device, 0.0f);
    
    // Fill with some data
    std::vector<float> x_data(N);
    std::vector<float> y_data(N);
    for(int i=0; i<N; ++i) {
        x_data[i] = (float)i / N;
        y_data[i] = 2.0f * x_data[i] + 1.0f;
    }
    x.copy_from_host(x_data.data());
    y.copy_from_host(y_data.data());

    // 2. Model
    auto model = vesper::nn::Linear(1, 1, true, device);
    
    // Initialize weights to something known to ensure convergence
    // w = 0.5, b = 0.5
    // Note: Linear weights are usually [out_features, in_features] -> [1, 1]
    // Bias is [out_features] -> [1]
    std::vector<float> w_init = {0.5f};
    std::vector<float> b_init = {0.5f};
    model.weight.copy_from_host(w_init.data());
    model.bias.copy_from_host(b_init.data());

    // 3. Loss and Optimizer
    auto criterion = vesper::nn::MSELoss();
    auto optimizer = vesper::optim::SGD(model.parameters(), 0.1f);

    // 4. Training Loop
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < 200; ++epoch) {
        // Forward
        auto output = model.forward(x);
        auto loss = criterion.forward(output, y);

        if (epoch == 0) {
            std::vector<float> loss_val(1);
            loss.copy_to_host(loss_val.data());
            initial_loss = loss_val[0];
            std::cout << "Initial Loss: " << initial_loss << std::endl;
        }

        // Backward
        optimizer.zero_grad();
        loss.backward();

        // Update
        optimizer.step();
        
        if (epoch == 199) {
            std::vector<float> loss_val(1);
            loss.copy_to_host(loss_val.data());
            final_loss = loss_val[0];
            std::cout << "Final Loss: " << final_loss << std::endl;
        }
    }

    // 5. Verification
    assert(final_loss < initial_loss);
    assert(final_loss < 0.05f); // Should be close to 0

    // Check learned parameters
    std::vector<float> w_val(1);
    std::vector<float> b_val(1);
    model.weight.copy_to_host(w_val.data());
    model.bias.copy_to_host(b_val.data());
    
    std::cout << "Learned w: " << w_val[0] << " (expected 2.0)" << std::endl;
    std::cout << "Learned b: " << b_val[0] << " (expected 1.0)" << std::endl;

    assert(std::abs(w_val[0] - 2.0f) < 0.2f);
    assert(std::abs(b_val[0] - 1.0f) < 0.2f);

    std::cout << "Linear Regression Test Passed!" << std::endl;
}

int main() {
    test_linear_regression();
    return 0;
}
