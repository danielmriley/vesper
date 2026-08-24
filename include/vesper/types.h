#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace vesper {

enum class Device { CPU, HIP };

enum class DType { F32 };

inline const char* device_name(Device device) {
    switch (device) {
        case Device::CPU:
            return "cpu";
        case Device::HIP:
            return "hip";
    }
    throw std::logic_error("unhandled Device");
}

inline std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return sizeof(float);
    }
    throw std::logic_error("unhandled DType");
}

[[noreturn]] inline void fail(const std::string& message) {
    throw std::runtime_error(message);
}

inline void check(bool ok, const std::string& message) {
    if (!ok) {
        fail(message);
    }
}

}  // namespace vesper
