#include <vesper/core/storage.h>
#include <vesper/core/allocator.h>
#include <stdexcept>
#include <utility>

namespace vesper {

Storage::Storage(Device device, size_t size_bytes)
    : device_(device), size_bytes_(size_bytes), data_ptr(nullptr) {
    if (size_bytes_ == 0) {
        return;
    }

    data_ptr = get_allocator(device)->allocate(size_bytes);
    if (!data_ptr) {
        throw std::runtime_error("Failed to allocate memory via CachingAllocator.");
    }
}

Storage::~Storage() {
    if (data_ptr == nullptr) {
        return;
    }
    // We must use the allocator to free, ensuring we use the correct backend
    // Note: If get_allocator throws (e.g. static destruction order issue?), we might leak.
    // But typically allocators outlive storage.
    try {
        get_allocator(device_)->free(data_ptr);
    } catch (...) {
        // Swallow exceptions in destructor
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
        // Free our own resource first using a safe reset pattern
        // (avoids calling destructor directly which is non-idiomatic)
        if (data_ptr != nullptr) {
            try {
                get_allocator(device_)->free(data_ptr);
            } catch (...) {
                // Swallow exceptions in move assignment
            }
        }

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
