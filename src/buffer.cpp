#include "vesper/buffer.h"

#include "vesper/hip.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace vesper {

void Buffer::release() {
    if (hip_ != nullptr) {
        hip_free(hip_);
        hip_ = nullptr;
    }
    host_.clear();
    n_ = 0;
    device_ = Device::CPU;
}

void Buffer::steal(Buffer& other) noexcept {
    host_ = std::move(other.host_);
    hip_ = other.hip_;
    n_ = other.n_;
    device_ = other.device_;
    other.hip_ = nullptr;
    other.n_ = 0;
    other.device_ = Device::CPU;
}

Buffer::Buffer(std::size_t n_elems, Device device) : n_(n_elems), device_(device) {
    switch (device) {
        case Device::CPU:
            host_.assign(n_elems, 0.0f);
            return;
        case Device::HIP:
            if (n_elems == 0) {
                return;
            }
            hip_ = static_cast<float*>(hip_alloc(n_elems * sizeof(float)));
            return;
    }
    throw std::logic_error("unhandled Device");
}

Buffer::Buffer(const Buffer& other) : n_(other.n_), device_(other.device_) {
    switch (other.device_) {
        case Device::CPU:
            host_ = other.host_;
            return;
        case Device::HIP:
            if (other.hip_ == nullptr || other.n_ == 0) {
                return;
            }
            hip_ = static_cast<float*>(hip_alloc(other.n_ * sizeof(float)));
            hip_copy_d2d(hip_, other.hip_, other.n_ * sizeof(float));
            return;
    }
    throw std::logic_error("unhandled Device");
}

Buffer::Buffer(Buffer&& other) noexcept {
    steal(other);
}

Buffer& Buffer::operator=(const Buffer& other) {
    if (this == &other) {
        return *this;
    }
    Buffer tmp(other);
    *this = std::move(tmp);
    return *this;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    release();
    steal(other);
    return *this;
}

Buffer::~Buffer() {
    release();
}

float* Buffer::data() {
    switch (device_) {
        case Device::CPU:
            return host_.data();
        case Device::HIP:
            return hip_;
    }
    throw std::logic_error("unhandled Device");
}

const float* Buffer::data() const {
    switch (device_) {
        case Device::CPU:
            return host_.data();
        case Device::HIP:
            return hip_;
    }
    throw std::logic_error("unhandled Device");
}

Buffer Buffer::to(Device device) const {
    if (device == device_) {
        return *this;
    }
    Buffer out(n_, device);
    if (n_ == 0) {
        return out;
    }
    if (device_ == Device::CPU && device == Device::HIP) {
        hip_copy_h2d(out.hip_, host_.data(), n_ * sizeof(float));
        return out;
    }
    if (device_ == Device::HIP && device == Device::CPU) {
        hip_copy_d2h(out.host_.data(), hip_, n_ * sizeof(float));
        return out;
    }
    throw std::logic_error("unhandled Device copy");
}

void Buffer::fill(float value) {
    switch (device_) {
        case Device::CPU:
            std::fill(host_.begin(), host_.end(), value);
            return;
        case Device::HIP:
            hip_fill(hip_, value, n_);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void Buffer::copy_from(const float* host, std::size_t n) {
    check(n <= n_, "copy_from overflow");
    if (n == 0) {
        return;
    }
    switch (device_) {
        case Device::CPU:
            std::copy(host, host + n, host_.begin());
            return;
        case Device::HIP:
            hip_copy_h2d(hip_, host, n * sizeof(float));
            return;
    }
    throw std::logic_error("unhandled Device");
}

void Buffer::copy_to(float* host, std::size_t n) const {
    check(n <= n_, "copy_to overflow");
    if (n == 0) {
        return;
    }
    switch (device_) {
        case Device::CPU:
            std::copy(host_.begin(), host_.begin() + static_cast<std::ptrdiff_t>(n), host);
            return;
        case Device::HIP:
            hip_copy_d2h(host, hip_, n * sizeof(float));
            return;
    }
    throw std::logic_error("unhandled Device");
}

}  // namespace vesper
