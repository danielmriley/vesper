#include <vesper/core/storage.h>
#include <iostream>
#include <cassert>

void test_hip_allocation() {
#if USE_HIP_BACKEND
    std::cout << "Testing HIP Storage allocation..." << std::endl;
    const size_t bytes = 1024;
    vesper::Storage storage(vesper::Device::HIP, bytes);

    assert(storage.data() != nullptr);
    assert(storage.device() == vesper::Device::HIP);
    assert(storage.size() == bytes);

    // Test move semantics
    vesper::Storage moved_storage(std::move(storage));
    assert(storage.data() == nullptr); // Original is now empty
    assert(storage.size() == 0);
    assert(moved_storage.data() != nullptr);
    assert(moved_storage.device() == vesper::Device::HIP);
    assert(moved_storage.size() == bytes);

    std::cout << "HIP Storage Test Passed!" << std::endl;
#else
    std::cout << "Skipping HIP Storage test (backend disabled)." << std::endl;
#endif
}

void test_cpu_allocation() {
#if USE_CPU_BACKEND
    std::cout << "Testing CPU Storage allocation..." << std::endl;
    const size_t bytes = 1024;
    vesper::Storage storage(vesper::Device::CPU, bytes);

    assert(storage.data() != nullptr);
    assert(storage.device() == vesper::Device::CPU);
    assert(storage.size() == bytes);

    // Test move semantics
    vesper::Storage moved_storage(std::move(storage));
    assert(storage.data() == nullptr); // Original is now empty
    assert(storage.size() == 0);
    assert(moved_storage.data() != nullptr);
    assert(moved_storage.device() == vesper::Device::CPU);
    assert(moved_storage.size() == bytes);

    std::cout << "CPU Storage Test Passed!" << std::endl;
#else
    std::cout << "Skipping CPU Storage test (backend disabled)." << std::endl;
#endif
}

int main() {
    test_hip_allocation();
    test_cpu_allocation();
    return 0;
}
