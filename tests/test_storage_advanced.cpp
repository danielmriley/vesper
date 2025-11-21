#include <vesper/core/storage.h>
#include <iostream>
#include <cassert>
#include <vector>

void test_multiple_allocations() {
    std::cout << "Testing multiple storage allocations..." << std::endl;
#if USE_HIP_BACKEND
    std::vector<vesper::Storage> storages;
    for (int i = 0; i < 10; ++i) {
        storages.emplace_back(vesper::Device::HIP, 1024);
        assert(storages.back().data() != nullptr);
    }
    // Check they are distinct
    for (size_t i = 0; i < storages.size(); ++i) {
        for (size_t j = i + 1; j < storages.size(); ++j) {
            assert(storages[i].data() != storages[j].data());
        }
    }
#endif
}

void test_large_allocation() {
    std::cout << "Testing large storage allocation..." << std::endl;
#if USE_HIP_BACKEND
    // 100 MB
    size_t size = 100 * 1024 * 1024;
    vesper::Storage s(vesper::Device::HIP, size);
    assert(s.data() != nullptr);
    assert(s.size() == size);
#endif
}

int main() {
    test_multiple_allocations();
    test_large_allocation();
    std::cout << "Advanced storage tests passed!" << std::endl;
    return 0;
}
