#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <iostream>
#include <cassert>

using namespace vesper;

void test_graph_construction() {
    std::cout << "Testing Graph Construction..." << std::endl;

    auto a = full({1}, DType::Float32, Device::CPU, 2.0f, true);
    auto b = full({1}, DType::Float32, Device::CPU, 3.0f, true);

    auto c = ops::add(a, b);

    // Check c properties
    assert(c.requires_grad());
    assert(c.grad_node != nullptr);
    assert(c.grad_node->next_edges.empty()); // Inputs are leaves

    auto d = ops::mul(c, a);

    // Check d properties
    assert(d.requires_grad());
    assert(d.grad_node != nullptr);
    assert(d.grad_node->next_edges.size() >= 1);
    
    // Check edge points to c's grad_node
    bool found_c = false;
    for (const auto& edge : d.grad_node->next_edges) {
        if (edge.node == c.grad_node) {
            found_c = true;
            break;
        }
    }
    assert(found_c);

    std::cout << "Graph Construction Passed!" << std::endl;
}

void test_backward_fn_execution() {
    std::cout << "Testing Backward Function Execution (Manual)..." << std::endl;

    auto a = full({1}, DType::Float32, Device::CPU, 2.0f, true);
    auto b = full({1}, DType::Float32, Device::CPU, 3.0f, true);
    auto c = ops::add(a, b);

    // Simulate backward pass
    // 1. Seed gradient for c
    c.grad() = full({1}, DType::Float32, Device::CPU, 1.0f);

    // 2. Execute backward_fn
    c.grad_node->backward_fn();

    // 3. Check gradients of a and b
    // grad_a = grad_c * 1 = 1
    // grad_b = grad_c * 1 = 1
    
    // We need to access data to verify.
    // Since we are on CPU, we can use data_ptr.
    
    assert(a.grad().defined()); // Should be defined now
    assert(b.grad().defined());

    float grad_a = *a.grad().data_ptr<float>();
    float grad_b = *b.grad().data_ptr<float>();

    assert(grad_a == 1.0f);
    assert(grad_b == 1.0f);

    std::cout << "Backward Function Execution Passed!" << std::endl;
}

int main() {
    try {
        test_graph_construction();
        test_backward_fn_execution();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
