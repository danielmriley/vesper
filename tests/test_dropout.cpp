#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace vesper;

void test_dropout_forward() {
    std::cout << "Testing dropout forward..." << std::endl;
    
    int N = 10000;
    Tensor x = full({N}, DType::Float32, Device::CPU, 1.0f);
    float p = 0.5f;
    
    Tensor y = nn::functional::dropout(x, p, true);
    
    // Check mean
    // Expected mean is 1.0 (since we scale by 1/(1-p))
    // Elements are either 0 or 1/(1-p) = 2.
    // Probability of 2 is (1-p) = 0.5.
    // Mean = 0 * 0.5 + 2 * 0.5 = 1.0.
    
    Tensor mean_t = ops::mean(y);
    float mean_val = *mean_t.data_ptr<float>();
    
    std::cout << "Mean: " << mean_val << " (expected ~1.0)" << std::endl;
    assert(std::abs(mean_val - 1.0f) < 0.1f);
    
    // Check zeros count
    int zeros = 0;
    const float* y_ptr = y.data_ptr<float>();
    for (int i = 0; i < N; ++i) {
        if (y_ptr[i] == 0.0f) zeros++;
    }
    
    float zero_ratio = (float)zeros / N;
    std::cout << "Zero ratio: " << zero_ratio << " (expected ~" << p << ")" << std::endl;
    assert(std::abs(zero_ratio - p) < 0.05f);
}

void test_dropout_eval() {
    std::cout << "Testing dropout eval..." << std::endl;
    Tensor x = full({10}, DType::Float32, Device::CPU, 1.0f);
    Tensor y = nn::functional::dropout(x, 0.5, false);
    
    // Should be identical
    const float* y_ptr = y.data_ptr<float>();
    for (int i = 0; i < 10; ++i) {
        assert(y_ptr[i] == 1.0f);
    }
}

void test_dropout_backward() {
    std::cout << "Testing dropout backward..." << std::endl;
    
    Tensor x = full({10}, DType::Float32, Device::CPU, 1.0f, true);
    Tensor y = nn::functional::dropout(x, 0.5, true);
    
    Tensor loss = ops::sum(y);
    loss.backward();
    
    // Gradients should be mask * scale
    // mask is 1 where y != 0, 0 where y == 0.
    // scale is 1/(1-0.5) = 2.
    
    const float* y_ptr = y.data_ptr<float>();
    const float* g_ptr = x.grad().data_ptr<float>();
    
    for (int i = 0; i < 10; ++i) {
        if (y_ptr[i] == 0.0f) {
            assert(g_ptr[i] == 0.0f);
        } else {
            assert(std::abs(g_ptr[i] - 2.0f) < 1e-5f);
        }
    }
}

int main() {
    test_dropout_forward();
    test_dropout_eval();
    test_dropout_backward();
    std::cout << "All dropout tests passed!" << std::endl;
    return 0;
}
