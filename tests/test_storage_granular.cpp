#include <vesper/core/storage.h>
#include <iostream>
#include <cassert>

void test_zero_allocation() {
    std::cout << "Testing zero-byte allocation..." << std::endl;
    // Should not throw, data might be nullptr or a sentinel, size should be 0
#if USE_HIP_BACKEND
    {
        vesper::Storage s(vesper::Device::HIP, 0);
        assert(s.size() == 0);
        assert(s.data() == nullptr); 
    }
#endif
#if USE_CPU_BACKEND
    {
        vesper::Storage s(vesper::Device::CPU, 0);
        assert(s.size() == 0);
        assert(s.data() == nullptr);
    }
#endif
}

void test_move_assignment() {
    std::cout << "Testing move assignment..." << std::endl;
#if USE_HIP_BACKEND
    {
        vesper::Storage s1(vesper::Device::HIP, 1024);
        void* ptr1 = s1.data();
        
        vesper::Storage s2(vesper::Device::HIP, 512);
        
        s2 = std::move(s1);
        
        // s1 should be empty
        assert(s1.data() == nullptr);
        assert(s1.size() == 0);
        
        // s2 should have s1's data
        assert(s2.data() == ptr1);
        assert(s2.size() == 1024);
        assert(s2.device() == vesper::Device::HIP);
    }
#endif
}

int main() {
    test_zero_allocation();
    test_move_assignment();
    std::cout << "Storage granular tests passed!" << std::endl;
    return 0;
}
