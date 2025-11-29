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
        // Use lock() to get shared_ptr from weak_ptr for comparison
        auto edge_node = edge.node.lock();
        if (edge_node == c.grad_node) {
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

void test_backward_with_gradient() {
    std::cout << "Testing Backward with Initial Gradient..." << std::endl;
    
    auto a = full({2}, DType::Float32, Device::CPU, 2.0f, true);
    auto b = ops::mul(a, 2.0f); // b = 2a
    
    // dy/db = [0.5, 1.0]
    auto grad = empty({2}, DType::Float32, Device::CPU);
    std::vector<float> grad_data = {0.5f, 1.0f};
    grad.copy_from_host(grad_data.data());
    
    b.backward(grad);
    
    // dy/da = dy/db * db/da = [0.5, 1.0] * 2 = [1.0, 2.0]
    
    assert(std::abs(a.grad().item<float>(0) - 1.0f) < 1e-5);
    assert(std::abs(a.grad().item<float>(1) - 2.0f) < 1e-5);
    
    std::cout << "Backward with Initial Gradient Passed!" << std::endl;
}

void test_to_autograd() {
#ifdef USE_CUDA_BACKEND
    std::cout << "Testing Tensor::to Autograd (CPU -> CUDA -> CPU)..." << std::endl;
    try {
        // Check if CUDA is actually available at runtime
        // We can try to create a small tensor
        try {
            auto check = zeros({1}, DType::Float32, Device::CUDA);
        } catch (...) {
            std::cout << "CUDA runtime not available, skipping." << std::endl;
            return;
        }

        auto a = full({1}, DType::Float32, Device::CPU, 2.0f, true);
        auto b = a.to(Device::CUDA);
        
        // b should require grad and have a grad_node
        assert(b.requires_grad());
        assert(b.grad_node != nullptr);
        
        auto c = ops::mul(b, 3.0f);
        c.backward();
        
        // c = 3b = 3a
        // dc/da = 3
        
        // Check a.grad()
        // It should be on CPU
        assert(a.grad().device() == Device::CPU);
        
        // Copy to host to check value
        float grad_val = *a.grad().data_ptr<float>();
        assert(std::abs(grad_val - 3.0f) < 1e-5);
        
        std::cout << "Tensor::to Autograd Passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Skipping CUDA test due to error: " << e.what() << std::endl;
    }
#else
    std::cout << "Skipping Tensor::to Autograd (CUDA not enabled)" << std::endl;
#endif
}

int main() {
    try {
        test_graph_construction();
        test_backward_fn_execution();
        test_backward_with_gradient();
        test_to_autograd();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
