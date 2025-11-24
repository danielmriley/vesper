#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/loss.h>
#include <vesper/optim/sgd.h>
#include <vesper/ops/random.h>
#include <vesper/core/allocator.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>

void test_large_linear_regression() {
    std::cout << "Testing Large Scale Linear Regression (Stress Test)..." << std::endl;

    // Device selection
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

    // Parameters
    // Reduced N to avoid OOM on smaller systems during stress test
    const int64_t N = 5000; 
    const int64_t D_in = 512;
    const int64_t D_out = 512;
    const int epochs = 100; 
    const float learning_rate = 2.0f; 

    std::cout << "Problem Size: N=" << N << ", D_in=" << D_in << ", D_out=" << D_out << std::endl;

    // 1. Data Generation
    auto x = vesper::empty({N, D_in}, vesper::DType::Float32, device);
    vesper::ops::normal_(x, 0.0f, 1.0f);

    // Initialization scale (Xavier-like)
    float init_scale = 1.0f / std::sqrt((float)D_in);

    // True Model (Target)
    auto true_model = vesper::nn::Linear(D_in, D_out, true, device);
    vesper::ops::normal_(true_model.weight, 0.0f, init_scale); 
    vesper::ops::normal_(true_model.bias, 0.0f, init_scale);

    // Generate Y target
    auto y = true_model.forward(x);
    
    // 2. Trainable Model
    auto model = vesper::nn::Linear(D_in, D_out, true, device);
    vesper::ops::normal_(model.weight, 0.0f, init_scale);
    vesper::ops::normal_(model.bias, 0.0f, init_scale);

    // 3. Optimizer
    auto criterion = vesper::nn::MSELoss();
    auto optimizer = vesper::optim::SGD(model.parameters(), learning_rate);

    // 4. Training Loop
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward
        auto output = model.forward(x);
        auto loss = criterion.forward(output, y);

        float current_loss_val = 0.0f;
        
        // Log occasionally
        if (epoch == 0 || epoch == epochs - 1 || epoch % 10 == 0) {
             std::vector<float> loss_vec(1);
             loss.copy_to_host(loss_vec.data());
             current_loss_val = loss_vec[0];
             
             if (epoch == 0) initial_loss = current_loss_val;
             if (epoch == epochs - 1) final_loss = current_loss_val;

             std::cout << "Epoch " << epoch << ", Loss: " << current_loss_val << std::endl;
        }

        // Backward
        optimizer.zero_grad();
        loss.backward();

        // Update
        optimizer.step();
        
        // Clean cache to avoid OOM
        auto* alloc = dynamic_cast<vesper::CachingAllocator*>(vesper::get_allocator(device));
        if (alloc) {
            alloc->empty_cache();
        }
    }

    std::cout << "Initial Loss: " << initial_loss << std::endl;
    std::cout << "Final Loss: " << final_loss << std::endl;

    // 5. Verification
    // Should decrease by at least 40%
    if (final_loss >= initial_loss * 0.6f) {
         std::cerr << "Test Failed: Loss did not decrease significantly! " 
                   << initial_loss << " -> " << final_loss << std::endl;
         exit(1);
    }
    
    std::cout << "Large Scale Linear Regression Test Passed!" << std::endl;
}

int main() {
    test_large_linear_regression();
    return 0;
}
