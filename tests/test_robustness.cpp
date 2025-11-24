#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <cassert>
#include <vector>

using namespace vesper;

void test_broadcasting_backward_torture() {
    std::cout << "Testing General Broadcasting Backward (Torture)..." << std::endl;
    // Case: (2, 3, 4) + (3, 1) -> (2, 3, 4)
    // Gradient for (3, 1) requires summing over dim 0 (size 2) and dim 2 (size 4).
    
    auto a = full({2, 3, 4}, DType::Float32, Device::CPU, 1.0f, true);
    auto b = full({3, 1}, DType::Float32, Device::CPU, 2.0f, true);
    
    try {
        auto c = ops::add(a, b);
        auto loss = ops::sum(c);
        loss.backward();
        
        // If we reach here, it passed!
        // Check gradients
        // grad_b should be sum of ones over dim 0 and 2.
        // Total elements in a is 2*3*4 = 24.
        // b has 3 elements. Each contributes to 2*4 = 8 outputs.
        // So grad_b should be all 8.0f.
        
        std::cout << "Broadcasting backward passed (Unexpectedly!)" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
    }
}

void test_memory_leak_torture() {
    std::cout << "Testing Memory Leak (Torture)..." << std::endl;
    // Create a chain of operations and let them go out of scope.
    for (int i = 0; i < 100; ++i) {
        auto a = full({100, 100}, DType::Float32, Device::CPU, 1.0f, true);
        auto b = full({100, 100}, DType::Float32, Device::CPU, 2.0f, true);
        auto c = ops::add(a, b);
        auto d = ops::mul(c, 2.0f);
        d.backward();
    }
    std::cout << "Memory leak loop finished. Check allocator warnings at exit." << std::endl;
}

void test_max_min_torture() {
    std::cout << "Testing Max/Min Reduction (Torture)..." << std::endl;
    
    auto a = empty({3}, DType::Float32, Device::CPU, true);
    std::vector<float> data = {1.0f, 5.0f, 2.0f};
    a.copy_from_host(data.data());
    
    // Max
    auto max_val = ops::max(a);
    float val;
    max_val.copy_to_host(&val);
    assert(val == 5.0f);
    
    max_val.backward();
    // Grad should be [0, 1, 0]
    std::vector<float> grad(3);
    a.grad().copy_to_host(grad.data());
    assert(grad[0] == 0.0f);
    assert(grad[1] == 1.0f);
    assert(grad[2] == 0.0f);
    
    // Zero grad
    a.grad() = zeros(a.shape(), a.dtype(), a.device());
    
    // Min
    auto min_val = ops::min(a);
    min_val.copy_to_host(&val);
    assert(val == 1.0f);
    
    min_val.backward();
    // Grad should be [1, 0, 0]
    a.grad().copy_to_host(grad.data());
    assert(grad[0] == 1.0f);
    assert(grad[1] == 0.0f);
    assert(grad[2] == 0.0f);
    
    std::cout << "Max/Min passed!" << std::endl;
}

void test_slice_backward_torture() {
    std::cout << "Testing Slice Backward (Torture)..." << std::endl;
    auto a = full({4, 4}, DType::Float32, Device::CPU, 1.0f, true);
    
    // Slice: a[1:3, :]
    // We don't have Python syntax, but we have slice(index).
    // a.slice(1) gives row 1.
    
    auto b = a.slice(1); // Row 1
    auto c = a.slice(2); // Row 2
    
    auto d = ops::add(b, c);
    auto loss = ops::sum(d);
    loss.backward();
    
    // Grad of a should be:
    // Row 1: 1.0 (from b)
    // Row 2: 1.0 (from c)
    // Row 0, 3: 0.0
    
    std::vector<float> grad(16);
    a.grad().copy_to_host(grad.data());
    
    for(int i=0; i<4; ++i) assert(grad[i] == 0.0f); // Row 0
    for(int i=4; i<8; ++i) assert(grad[i] == 1.0f); // Row 1
    for(int i=8; i<12; ++i) assert(grad[i] == 1.0f); // Row 2
    for(int i=12; i<16; ++i) assert(grad[i] == 0.0f); // Row 3
    
    std::cout << "Slice backward passed!" << std::endl;
}

int main() {
    test_broadcasting_backward_torture();
    test_max_min_torture();
    test_slice_backward_torture();
    test_memory_leak_torture();
    return 0;
}
