#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <iostream>
#include <cassert>

using namespace vesper;

void test_memory_leak() {
    std::cout << "Testing Memory Leak..." << std::endl;
    
    long initial_use_count = 0;
    
    {
        Tensor a = zeros({10}, DType::Float32, Device::CPU, true);
        initial_use_count = a.storage_use_count();
        // initial_use_count should be 1
        
        Tensor b = ops::add(a, a);
        // b = a + a
        // b holds a grad_node.
        // grad_node holds a backward_fn.
        // If backward_fn captures b by value, we have a cycle: b -> grad_node -> backward_fn -> b
        
        // a is also captured by backward_fn.
    }
    
    // If there is a cycle involving b, b is not destroyed.
    // If b is not destroyed, and b's backward_fn captures a, then a is also not destroyed.
    // So we can check if a's storage is still alive? 
    // Wait, 'a' went out of scope. If 'a' is captured by value in b's backward_fn, 
    // and b is leaked, then a is leaked.
    
    // But we can't check 'a' after it goes out of scope.
    
    // Let's try this:
    std::shared_ptr<Storage> storage_copy;
    {
        Tensor a = zeros({10}, DType::Float32, Device::CPU, true);
        // Keep a weak reference or just a copy of the storage to check use_count
        // But Tensor doesn't expose the storage pointer directly to copy it easily 
        // without making a new Tensor.
        
        Tensor c = a; // c shares storage with a.
        storage_copy = std::shared_ptr<Storage>(c.storage_use_count() > 0 ? nullptr : nullptr); 
        // Wait, I can't get the shared_ptr itself from the public API.
        
        // I'll rely on the fact that I added storage_use_count().
        // But I need to check it *after* the scope.
    }
}

// Better test:
// Create a scope. Inside, create tensors and do operations.
// Ensure that after the scope, the memory is freed.
// Since we don't have a global memory counter, we can use a custom allocator or 
// just check the behavior with a weak_ptr if we could capture it.

// Let's use the fact that if I make a copy of 'a' outside, its use count should drop 
// when the inner computation is destroyed.

void test_cycle() {
    Tensor a = zeros({10}, DType::Float32, Device::CPU, true);
    long start_count = a.storage_use_count(); // Should be 1
    
    {
        Tensor b = ops::add(a, a);
        // b captures a in backward_fn.
        // If b leaks (cycle), it keeps holding a.
    }
    
    long end_count = a.storage_use_count();
    
    if (end_count > start_count) {
        std::cout << "FAIL: Memory leak detected! Storage use count increased from " 
                  << start_count << " to " << end_count << std::endl;
        exit(1);
    } else {
        std::cout << "PASS: No memory leak detected." << std::endl;
    }
}

int main() {
    try {
        test_cycle();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
