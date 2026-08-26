#pragma once

#include "vesper/types.h"

#include <cstddef>
#include <vector>

namespace vesper {

class Buffer {
public:
    Buffer() = default;
    Buffer(std::size_t n_elems, Device device);
    Buffer(const Buffer& other);
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(const Buffer& other);
    Buffer& operator=(Buffer&& other) noexcept;
    ~Buffer();

    float* data();
    const float* data() const;
    std::size_t size() const { return n_; }
    Device device() const { return device_; }

    Buffer to(Device device) const;

    void fill(float value);
    void copy_from(const float* host, std::size_t n);
    void copy_to(float* host, std::size_t n) const;

private:
    void release();
    void steal(Buffer& other) noexcept;

    std::vector<float> host_;
    float* hip_ = nullptr;
    std::size_t n_ = 0;
    Device device_ = Device::CPU;
};

}  // namespace vesper
