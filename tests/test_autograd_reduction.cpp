#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace vesper;

void test_sum_backward() {
    std::cout << "Testing Sum Backward..." << std::endl;

    // x = [1, 2, 3]
    // y = sum(x) = 6
    // dy/dx = [1, 1, 1]
    
    auto x = full({3}, DType::Float32, Device::CPU, 0.0f, true);
    // Fill manually
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f};
    x.copy_from_host(x_data.data());

    auto y = ops::sum(x);

    y.backward();

    assert(x.grad().defined());
    
    std::vector<float> grad_x_data(3);
    x.grad().copy_to_host(grad_x_data.data());

    for (float val : grad_x_data) {
        assert(val == 1.0f);
    }

    std::cout << "Sum Backward Passed!" << std::endl;
}

void test_sum_square_backward() {
    std::cout << "Testing Sum(x*x) Backward..." << std::endl;

    // x = [1, 2, 3]
    // y = sum(x * x) = 1 + 4 + 9 = 14
    // dy/dx = 2x = [2, 4, 6]

    auto x = full({3}, DType::Float32, Device::CPU, 0.0f, true);
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f};
    x.copy_from_host(x_data.data());

    auto x2 = ops::mul(x, x);
    auto y = ops::sum(x2);

    y.backward();

    assert(x.grad().defined());

    std::vector<float> grad_x_data(3);
    x.grad().copy_to_host(grad_x_data.data());

    assert(std::abs(grad_x_data[0] - 2.0f) < 1e-5);
    assert(std::abs(grad_x_data[1] - 4.0f) < 1e-5);
    assert(std::abs(grad_x_data[2] - 6.0f) < 1e-5);

    std::cout << "Sum(x*x) Backward Passed!" << std::endl;
}

int main() {
    try {
        test_sum_backward();
        test_sum_square_backward();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
