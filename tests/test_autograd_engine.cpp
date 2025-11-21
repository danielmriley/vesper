#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <iostream>
#include <cassert>

using namespace vesper;

void test_add_backward() {
    std::cout << "Testing Add Backward..." << std::endl;

    auto a = full({1}, DType::Float32, Device::CPU, 2.0f, true);
    auto b = full({1}, DType::Float32, Device::CPU, 3.0f, true);
    auto c = ops::add(a, b);

    c.backward();

    assert(a.grad().defined());
    assert(b.grad().defined());

    float grad_a = *a.grad().data_ptr<float>();
    float grad_b = *b.grad().data_ptr<float>();

    assert(grad_a == 1.0f);
    assert(grad_b == 1.0f);

    std::cout << "Add Backward Passed!" << std::endl;
}

void test_mul_backward() {
    std::cout << "Testing Mul Backward..." << std::endl;

    auto a = full({1}, DType::Float32, Device::CPU, 2.0f, true);
    auto b = full({1}, DType::Float32, Device::CPU, 3.0f, true);
    auto c = ops::mul(a, b);

    c.backward();

    assert(a.grad().defined());
    assert(b.grad().defined());

    float grad_a = *a.grad().data_ptr<float>();
    float grad_b = *b.grad().data_ptr<float>();

    // d(a*b)/da = b = 3.0
    // d(a*b)/db = a = 2.0
    assert(grad_a == 3.0f);
    assert(grad_b == 2.0f);

    std::cout << "Mul Backward Passed!" << std::endl;
}

void test_complex_graph() {
    std::cout << "Testing Complex Graph (d = a*b + a)..." << std::endl;

    auto a = full({1}, DType::Float32, Device::CPU, 2.0f, true);
    auto b = full({1}, DType::Float32, Device::CPU, 3.0f, true);
    
    auto c = ops::mul(a, b);
    auto d = ops::add(c, a);

    d.backward();

    // d = a*b + a
    // dd/da = b + 1 = 3 + 1 = 4
    // dd/db = a = 2

    float grad_a = *a.grad().data_ptr<float>();
    float grad_b = *b.grad().data_ptr<float>();

    assert(grad_a == 4.0f);
    assert(grad_b == 2.0f);

    std::cout << "Complex Graph Passed!" << std::endl;
}

int main() {
    try {
        test_add_backward();
        test_mul_backward();
        test_complex_graph();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
