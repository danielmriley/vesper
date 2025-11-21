#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/elementwise.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace vesper;

void test_matmul_backward_simple() {
    std::cout << "Testing Matmul Backward Simple..." << std::endl;

    // A = [[1, 2], [3, 4]]
    auto a = full({2, 2}, DType::Float32, Device::CPU, 0.0f, true);
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    a.copy_from_host(a_data.data());

    // B = [[1, 0], [0, 1]]
    auto b = full({2, 2}, DType::Float32, Device::CPU, 0.0f, true);
    std::vector<float> b_data = {1.0f, 0.0f, 0.0f, 1.0f};
    b.copy_from_host(b_data.data());

    auto c = ops::matmul(a, b);
    auto l = ops::sum(c); // L = sum(A*B)

    l.backward();

    // Check grad_A
    // dL/dC = ones(2, 2)
    // dL/dA = dL/dC * B^T = ones * I = ones
    assert(a.grad().defined());
    std::vector<float> grad_a_data(4);
    a.grad().copy_to_host(grad_a_data.data());
    for (float val : grad_a_data) {
        assert(std::abs(val - 1.0f) < 1e-5);
    }

    // Check grad_B
    // dL/dB = A^T * dL/dC = [[1, 3], [2, 4]] * [[1, 1], [1, 1]]
    // = [[1+3, 1+3], [2+4, 2+4]] = [[4, 4], [6, 6]]
    assert(b.grad().defined());
    std::vector<float> grad_b_data(4);
    b.grad().copy_to_host(grad_b_data.data());
    
    assert(std::abs(grad_b_data[0] - 4.0f) < 1e-5);
    assert(std::abs(grad_b_data[1] - 4.0f) < 1e-5);
    assert(std::abs(grad_b_data[2] - 6.0f) < 1e-5);
    assert(std::abs(grad_b_data[3] - 6.0f) < 1e-5);

    std::cout << "Matmul Backward Simple Passed!" << std::endl;
}

void test_matmul_backward_rectangular() {
    std::cout << "Testing Matmul Backward Rectangular..." << std::endl;

    // A: 2x3, B: 3x1
    // A = [[1, 1, 1], [1, 1, 1]]
    auto a = full({2, 3}, DType::Float32, Device::CPU, 1.0f, true);
    
    // B = [[1], [2], [3]]
    auto b = full({3, 1}, DType::Float32, Device::CPU, 0.0f, true);
    std::vector<float> b_data = {1.0f, 2.0f, 3.0f};
    b.copy_from_host(b_data.data());

    auto c = ops::matmul(a, b); // 2x1
    // C = [[1*1 + 1*2 + 1*3], [1*1 + 1*2 + 1*3]] = [[6], [6]]
    
    auto l = ops::sum(c); // L = 12

    l.backward();

    // dL/dC = [[1], [1]] (2x1)
    
    // dL/dA = dL/dC * B^T (2x1 * 1x3) = [[1], [1]] * [1, 2, 3]
    // = [[1, 2, 3], [1, 2, 3]]
    assert(a.grad().defined());
    std::vector<float> grad_a_data(6);
    a.grad().copy_to_host(grad_a_data.data());
    
    // Row 0
    assert(std::abs(grad_a_data[0] - 1.0f) < 1e-5);
    assert(std::abs(grad_a_data[1] - 2.0f) < 1e-5);
    assert(std::abs(grad_a_data[2] - 3.0f) < 1e-5);
    // Row 1
    assert(std::abs(grad_a_data[3] - 1.0f) < 1e-5);
    assert(std::abs(grad_a_data[4] - 2.0f) < 1e-5);
    assert(std::abs(grad_a_data[5] - 3.0f) < 1e-5);

    // dL/dB = A^T * dL/dC (3x2 * 2x1)
    // A^T = [[1, 1], [1, 1], [1, 1]]
    // dL/dB = [[1, 1], [1, 1], [1, 1]] * [[1], [1]] = [[2], [2], [2]]
    assert(b.grad().defined());
    std::vector<float> grad_b_data(3);
    b.grad().copy_to_host(grad_b_data.data());

    assert(std::abs(grad_b_data[0] - 2.0f) < 1e-5);
    assert(std::abs(grad_b_data[1] - 2.0f) < 1e-5);
    assert(std::abs(grad_b_data[2] - 2.0f) < 1e-5);

    std::cout << "Matmul Backward Rectangular Passed!" << std::endl;
}

int main() {
    try {
        test_matmul_backward_simple();
        test_matmul_backward_rectangular();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
