#include <vesper/nn/loss.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

void test_mse_loss_gpu() {
#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)
    std::cout << "Testing MSELoss on GPU..." << std::endl;

#if defined(USE_HIP_BACKEND)
    vesper::Device device = vesper::Device::HIP;
#else
    vesper::Device device = vesper::Device::CUDA;
#endif

    int64_t N = 4;
    // y_pred = [0, 1, 2, 3]
    // y_true = [0, 1, 2, 3] -> Loss should be 0
    
    auto y_pred = vesper::empty({N}, vesper::DType::Float32, device, true);
    auto y_true = vesper::empty({N}, vesper::DType::Float32, device, false);
    
    std::vector<float> pred_data = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> true_data = {0.0f, 1.0f, 2.0f, 3.0f};
    
    y_pred.copy_from_host(pred_data.data());
    y_true.copy_from_host(true_data.data());

    auto loss_fn = vesper::nn::MSELoss();
    auto loss = loss_fn.forward(y_pred, y_true);
    
    std::vector<float> loss_val(1);
    loss.copy_to_host(loss_val.data());
    
    assert(std::abs(loss_val[0]) < 1e-5);
    
    // Case 2: Non-zero loss
    // y_pred = [1, 1, 1, 1]
    // y_true = [0, 0, 0, 0]
    // diff = [1, 1, 1, 1]
    // sq_diff = [1, 1, 1, 1]
    // mean = 1.0
    
    std::vector<float> pred_ones = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> true_zeros = {0.0f, 0.0f, 0.0f, 0.0f};
    
    y_pred.copy_from_host(pred_ones.data());
    y_true.copy_from_host(true_zeros.data());
    
    // Reset gradients
    y_pred.grad() = vesper::zeros(y_pred.shape(), y_pred.dtype(), y_pred.device());
    
    loss = loss_fn.forward(y_pred, y_true);
    loss.copy_to_host(loss_val.data());
    
    assert(std::abs(loss_val[0] - 1.0f) < 1e-5);
    
    // Backward
    loss.backward();
    
    // d(MSE)/dy_pred = 2/N * (y_pred - y_true)
    // = 2/4 * (1 - 0) = 0.5
    
    std::vector<float> grad_data(N);
    y_pred.grad().copy_to_host(grad_data.data());
    
    for (int i = 0; i < N; ++i) {
        if (std::abs(grad_data[i] - 0.5f) > 1e-5) {
            std::cerr << "Gradient mismatch at " << i << ": expected 0.5, got " << grad_data[i] << std::endl;
            exit(1);
        }
    }
    
    std::cout << "MSELoss on GPU passed!" << std::endl;
#else
    std::cout << "Skipping GPU test." << std::endl;
#endif
}

int main() {
    test_mse_loss_gpu();
    return 0;
}
