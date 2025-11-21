#pragma once

#include <vesper/core/device.h>
#include <memory>

namespace vesper {

class Storage {
public:
    // Constructor: Allocates a buffer of a given size on a specific device.
    Storage(Device device, size_t size_bytes);

    // Destructor: Frees the allocated memory.
    ~Storage();

    // --- Rule of 5 ---
    // Disable copying
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    // Enable moving
    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept;

    // --- Accessors ---
    void* data() { return data_ptr; }
    const void* data() const { return data_ptr; }
    Device device() const { return device_; }
    size_t size() const { return size_bytes_; }

private:
    void* data_ptr = nullptr;
    size_t size_bytes_ = 0;
    Device device_;
};

} // namespace vesper
