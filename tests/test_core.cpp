#include <vesper/core/device.h>
#include <vesper/core/dtype.h>
#include <vesper/core/macros.h>
#include <iostream>
#include <cassert>

int main() {
    vesper::Device my_device = vesper::Device::HIP;
    vesper::DType my_dtype = vesper::DType::Float32;

    std::cout << "Testing Core Utilities..." << std::endl;
    std::cout << "Default Device: " << my_device << std::endl;
    std::cout << "Default DType: " << my_dtype << std::endl;
    std::cout << "Size of Float32: " << vesper::GetDTypeSize(my_dtype) << " bytes" << std::endl;
    std::cout << "Size of Float16: " << vesper::GetDTypeSize(vesper::DType::Float16) << " bytes" << std::endl;

    assert(vesper::GetDTypeSize(vesper::DType::Float32) == 4);
    assert(vesper::GetDTypeSize(vesper::DType::Int64) == 8);
    assert(vesper::GetDTypeSize(vesper::DType::Float16) == 2);
    assert(vesper::GetDTypeSize(vesper::DType::BFloat16) == 2);

    // Test VESPER_CHECK
    try {
        VESPER_CHECK(true, "This should not fail");
        // VESPER_CHECK(false, "This should fail"); // Uncomment to test failure
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Core Utilities Test Passed!" << std::endl;

    return 0;
}
