#include <vesper/core/device.h>
#include <iostream>
#include <sstream>
#include <cassert>

void test_device_printing() {
    std::cout << "Testing Device printing..." << std::endl;
    {
        std::stringstream ss;
        ss << vesper::Device::CPU;
        assert(ss.str() == "CPU");
    }
    {
        std::stringstream ss;
        ss << vesper::Device::CUDA;
        assert(ss.str() == "CUDA");
    }
    {
        std::stringstream ss;
        ss << vesper::Device::HIP;
        assert(ss.str() == "HIP");
    }
}

int main() {
    test_device_printing();
    std::cout << "Device granular tests passed!" << std::endl;
    return 0;
}
