#include <vesper/core/allocator.h>
#include <vesper/core/storage.h>
#include <iostream>
#include <cassert>
#include <vector>

// Helper to verify pointer reuse
void test_caching_behavior() {
    std::cout << "Testing CachingAllocator behavior..." << std::endl;
    
    vesper::Device device = vesper::Device::CPU;
#if defined(USE_HIP_BACKEND)
    device = vesper::Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    device = vesper::Device::CUDA;
#endif

    void* ptr1 = nullptr;
    
    // Scope 1: Allocate and free
    {
        vesper::Storage s1(device, 1024);
        ptr1 = s1.data();
        assert(ptr1 != nullptr);
        // s1 destructor calls allocator->free(ptr1)
    }

    // Scope 2: Allocate same size again
    {
        vesper::Storage s2(device, 1024);
        void* ptr2 = s2.data();
        
        // Should reuse the same pointer!
        if (ptr1 == ptr2) {
            std::cout << "  Success: Pointer reused from cache." << std::endl;
        } else {
            std::cerr << "  Failure: Pointer NOT reused (got " << ptr2 << ", expected " << ptr1 << ")" << std::endl;
            assert(false);
        }
    }

    // Test split/different size
    {
        vesper::Storage s3(device, 2048); // Different bucket
        void* ptr3 = s3.data();
        assert(ptr3 != ptr1);
    }

    std::cout << "CachingAllocator test passed!" << std::endl;
}

int main() {
    test_caching_behavior();
    return 0;
}
