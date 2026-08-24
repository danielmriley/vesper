#pragma once

#include "vesper/types.h"

#include <cstddef>
#include <vector>

namespace vesper {

class Buffer {
public:
    Buffer() = default;
    Buffer(std::size_t n_elems, Device device);

    float* data();
    const float* data() const;
    std::size_t size() const { return host_.size(); }
    Device device() const { return device_; }

    void fill(float value);
    void copy_from(const float* host, std::size_t n);
    void copy_to(float* host, std::size_t n) const;

private:
    std::vector<float> host_;
    Device device_ = Device::CPU;
};

}  // namespace vesper
