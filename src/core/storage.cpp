#include <vesper/core/storage.h>
#include <stdexcept>
#include <utility>

#if USE_HIP_BACKEND
#include <hip/hip_runtime.h>
#endif

namespace vesper {

Storage::Storage(Device device, size_t size_bytes)
    : device_(device), size_bytes_(size_bytes), data_ptr(nullptr) {
    if (size_bytes_ == 0) {
        return;
    }

    switch (device_) {
        case Device::HIP: {
#if USE_HIP_BACKEND
            hipError_t status = hipMalloc(&data_ptr, size_bytes_);
            if (status != hipSuccess) {
                throw std::runtime_error("Failed to allocate memory on HIP device.");
            }
#else
            throw std::runtime_error("HIP backend not enabled during build.");
#endif
            break;
        }
        case Device::CPU: {
            // Use aligned allocation for CPU for potential SIMD optimizations later
            data_ptr = new char[size_bytes_];
            if (!data_ptr) {
                throw std::runtime_error("Failed to allocate memory on CPU.");
            }
            break;
        }
        case Device::CUDA: {
            throw std::runtime_error("CUDA backend not yet implemented.");
        }
    }
}

Storage::~Storage() {
    if (data_ptr == nullptr) {
        return;
    }

    switch (device_) {
        case Device::HIP: {
#if USE_HIP_BACKEND
            hipError_t err = hipFree(data_ptr);
            if (err != hipSuccess) {
                // In a destructor, we shouldn't throw. Log to stderr.
                // We can't use std::cerr easily without including iostream, 
                // but for now let's just silence the warning by using the variable.
                (void)err; 
            }
#endif
            break;
        }
        case Device::CPU: {
            delete[] static_cast<char*>(data_ptr);
            break;
        }
        case Device::CUDA: {
            // No-op for now
            break;
        }
    }
}

Storage::Storage(Storage&& other) noexcept
    : data_ptr(other.data_ptr),
      size_bytes_(other.size_bytes_),
      device_(other.device_) {
    // Leave the moved-from object in a valid, destructible state
    other.data_ptr = nullptr;
    other.size_bytes_ = 0;
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        // Free our own resource first
        this->~Storage();

        // Pilfer the other's resources
        data_ptr = other.data_ptr;
        size_bytes_ = other.size_bytes_;
        device_ = other.device_;

        // Reset the other object
        other.data_ptr = nullptr;
        other.size_bytes_ = 0;
    }
    return *this;
}

} // namespace vesper
