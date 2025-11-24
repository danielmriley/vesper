#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/loss.h>
#include <vesper/optim/sgd.h>
#include <vesper/ops/random.h>
#include <vesper/ops/elementwise.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <chrono>

void test_large_linear_regression() {
    std::cout << "Testing Large Linear Regression (N=1,000,000)..." << std::endl;

    // Determine device
    vesper::Device device = vesper::Device::CPU;
#if defined(USE_HIP_BACKEND)
    device = vesper::Device::HIP;
    std::cout << "Device: HIP (GPU)" << std::endl;
#elif defined(USE_CUDA_BACKEND)
    device = vesper::Device::CUDA;
    std::cout << "Device: CUDA (GPU)" << std::endl;
#else
    std::cout << "Device: CPU" << std::endl;
#endif

    int64_t N = 1000000;
    
    // 1. Data Generation
    // X ~ Uniform(-1, 1) to balance gradients for w and b
    auto x = vesper::empty({N, 1}, vesper::DType::Float32, device);
    vesper::ops::uniform_(x, -1.0f, 1.0f);
    
    // y = 2x + 1 + noise
    // noise ~ Uniform(-0.1, 0.1)
    auto noise = vesper::empty({N, 1}, vesper::DType::Float32, device);
    vesper::ops::uniform_(noise, -0.1f, 0.1f);
    
    auto y = vesper::ops::mul(x, 2.0f);
    y = vesper::ops::add(y, 1.0f);
    y = vesper::ops::add(y, noise);
    
    // 2. Model
    auto model = vesper::nn::Linear(1, 1, true, device);
    
    // Initialize weights far from solution
    // w = 0.0, b = 0.0
    std::vector<float> w_init = {0.0f};
    std::vector<float> b_init = {0.0f};
    model.weight.copy_from_host(w_init.data());
    model.bias.copy_from_host(b_init.data());

    // 3. Loss and Optimizer
    auto criterion = vesper::nn::MSELoss();
    // We can use a larger LR now that inputs are smaller
    auto optimizer = vesper::optim::SGD(model.parameters(), 0.1f);

    // 4. Training Loop
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int epochs = 100;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward
        auto output = model.forward(x);
        auto loss = criterion.forward(output, y);

        // Backward
        optimizer.zero_grad();
        loss.backward();

        // Update
        optimizer.step();
        
        if (epoch % 10 == 0) {
             float loss_val = loss.item<float>();
             std::cout << "Epoch " << epoch << ", Loss: " << loss_val << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    
    std::cout << "Training time: " << diff.count() << " s" << std::endl;
    std::cout << "Avg time per epoch: " << diff.count() / epochs << " s" << std::endl;

    // 5. Verification
    float w_val = model.weight.item<float>();
    float b_val = model.bias.item<float>();
    
    std::cout << "Learned w: " << w_val << " (expected 2.0)" << std::endl;
    std::cout << "Learned b: " << b_val << " (expected 1.0)" << std::endl;

    assert(std::abs(w_val - 2.0f) < 0.1f);
    assert(std::abs(b_val - 1.0f) < 0.1f);

    std::cout << "Large Linear Regression Test Passed!" << std::endl;
}

int main() {
    test_large_linear_regression();
    return 0;
}
