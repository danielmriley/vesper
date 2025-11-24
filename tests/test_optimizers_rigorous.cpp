#include <vesper/optim/adam.h>
#include <vesper/optim/lion.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Access protected members for testing
class TestAdam : public vesper::optim::Adam {
public:
    using Adam::Adam;
    using Adam::state_;
    using Adam::t_;
    
    vesper::Tensor get_exp_avg(int param_idx) {
        return state_[param_idx]["exp_avg"];
    }
    vesper::Tensor get_exp_avg_sq(int param_idx) {
        return state_[param_idx]["exp_avg_sq"];
    }
};

class TestLion : public vesper::optim::Lion {
public:
    using Lion::Lion;
    using Lion::state_;
    
    vesper::Tensor get_exp_avg(int param_idx) {
        return state_[param_idx]["exp_avg"];
    }
};

// Helper to check tensor approx equality
void assert_all_close(const vesper::Tensor& t, const std::vector<float>& expected, float tol = 1e-5) {
    std::vector<float> data(t.numel());
    t.copy_to_host(data.data());
    for(size_t i=0; i<data.size(); ++i) {
        if(std::abs(data[i] - expected[i]) > tol) {
            std::cerr << "Mismatch at " << i << ": got " << data[i] << ", expected " << expected[i] << std::endl;
            assert(false);
        }
    }
}

void test_adam_step_math() {
    std::cout << "Testing Adam Step Math..." << std::endl;
    // Setup single parameter
    auto p = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 1.0f, true);
    
    float lr = 0.1f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float wd = 0.0f;
    
    TestAdam adam({p}, lr, beta1, beta2, eps, wd);
    
    // Set gradient manually
    // g = 0.5
    p.grad() = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 0.5f);
    
    // Step 1
    adam.step();
    
    // Calculations:
    // t = 1
    // m = 0.9 * 0 + 0.1 * 0.5 = 0.05
    // v = 0.999 * 0 + 0.001 * 0.5^2 = 0.001 * 0.25 = 0.00025
    // m_hat = 0.05 / (1 - 0.9) = 0.5
    // v_hat = 0.00025 / (1 - 0.999) = 0.25
    // step = 0.1 * 0.5 / (sqrt(0.25) + 1e-8) = 0.1 * 0.5 / 0.5 = 0.1
    // p_new = 1.0 - 0.1 = 0.9
    
    assert_all_close(p, {0.9f});
    assert_all_close(adam.get_exp_avg(0), {0.05f});
    assert_all_close(adam.get_exp_avg_sq(0), {0.00025f});
    
    // Step 2
    // Keep gradient 0.5
    adam.step();
    
    // Calculations:
    // t = 2
    // m = 0.9 * 0.05 + 0.1 * 0.5 = 0.045 + 0.05 = 0.095
    // v = 0.999 * 0.00025 + 0.001 * 0.25 = 0.00024975 + 0.00025 = 0.00049975
    // m_hat = 0.095 / (1 - 0.81) = 0.095 / 0.19 = 0.5
    // v_hat = 0.00049975 / (1 - 0.999^2) = 0.00049975 / (1 - 0.998001) = 0.00049975 / 0.001999 = 0.25
    // step = 0.1 * 0.5 / 0.5 = 0.1
    // p_new = 0.9 - 0.1 = 0.8
    
    assert_all_close(p, {0.8f});
    
    std::cout << "Adam math test passed!" << std::endl;
}

void test_adam_weight_decay() {
    std::cout << "Testing Adam Weight Decay..." << std::endl;
    auto p = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 1.0f, true);
    
    // High weight decay to make it obvious
    float wd = 0.1f;
    TestAdam adam({p}, 0.1f, 0.9f, 0.999f, 1e-8f, wd);
    
    // Zero gradient
    p.grad() = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 0.0f);
    
    // In standard Adam (not AdamW), L2 penalty is added to gradient.
    // g_eff = g + wd * p = 0 + 0.1 * 1.0 = 0.1
    
    adam.step();
    
    // m = 0.1 * 0.1 = 0.01
    // v = 0.001 * 0.1^2 = 0.00001
    // m_hat = 0.01 / 0.1 = 0.1
    // v_hat = 0.00001 / 0.001 = 0.01
    // step = lr * 0.1 / 0.1 = lr = 0.1
    // p_new = 1.0 - 0.1 = 0.9
    
    assert_all_close(p, {0.9f});
    std::cout << "Adam weight decay test passed!" << std::endl;
}

void test_lion_step_math() {
    std::cout << "Testing Lion Step Math..." << std::endl;
    auto p = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 1.0f, true);
    
    float lr = 0.1f;
    float beta1 = 0.9f;
    float beta2 = 0.99f;
    float wd = 0.0f;
    
    TestLion lion({p}, lr, beta1, beta2, wd);
    
    // grad = 0.5
    p.grad() = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 0.5f);
    
    // Step 1
    lion.step();
    
    // m_0 = 0
    // c = 0.9 * 0 + 0.1 * 0.5 = 0.05
    // update = sign(0.05) = 1.0
    // p_new = 1.0 - 0.1 * 1.0 = 0.9
    // m_new = 0.99 * 0 + 0.01 * 0.5 = 0.005
    
    assert_all_close(p, {0.9f});
    assert_all_close(lion.get_exp_avg(0), {0.005f});
    
    // Step 2
    // grad = -0.2
    p.grad() = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, -0.2f);
    
    lion.step();
    
    // m_1 = 0.005
    // c = 0.9 * 0.005 + 0.1 * (-0.2) = 0.0045 - 0.02 = -0.0155
    // update = sign(-0.0155) = -1.0
    // p_new = 0.9 - 0.1 * (-1.0) = 1.0
    // m_new = 0.99 * 0.005 + 0.01 * (-0.2) = 0.00495 - 0.002 = 0.00295
    
    assert_all_close(p, {1.0f});
    assert_all_close(lion.get_exp_avg(0), {0.00295f}, 1e-5f);
    
    std::cout << "Lion math test passed!" << std::endl;
}

int main() {
    test_adam_step_math();
    test_adam_weight_decay();
    test_lion_step_math();
    return 0;
}
