#pragma once

#include <iostream>

namespace vesper {

enum class Device {
    CPU,
    CUDA,
    HIP
};

// Utility for printing the device type
inline std::ostream& operator<<(std::ostream& os, const Device& device) {
    switch (device) {
        case Device::CPU:
            os << "CPU";
            break;
        case Device::CUDA:
            os << "CUDA";
            break;
        case Device::HIP:
            os << "HIP";
            break;
    }
    return os;
}

} // namespace vesper
