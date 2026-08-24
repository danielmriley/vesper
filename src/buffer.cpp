#include "vesper/buffer.h"

#include <algorithm>

namespace vesper {

Buffer::Buffer(std::size_t n_elems, Device device) : device_(device) {
    switch (device) {
        case Device::CPU:
            host_.assign(n_elems, 0.0f);
            return;
        case Device::HIP:
            fail("HIP allocator is not implemented yet");
    }
    throw std::logic_error("unhandled Device");
}

float* Buffer::data() {
    check(device_ == Device::CPU, "host pointer requested for non-CPU buffer");
    return host_.data();
}

const float* Buffer::data() const {
    check(device_ == Device::CPU, "host pointer requested for non-CPU buffer");
    return host_.data();
}

void Buffer::fill(float value) {
    std::fill(host_.begin(), host_.end(), value);
}

void Buffer::copy_from(const float* host, std::size_t n) {
    check(n <= host_.size(), "copy_from overflow");
    std::copy(host, host + n, host_.begin());
}

void Buffer::copy_to(float* host, std::size_t n) const {
    check(n <= host_.size(), "copy_to overflow");
    std::copy(host_.begin(), host_.begin() + static_cast<std::ptrdiff_t>(n), host);
}

}  // namespace vesper
