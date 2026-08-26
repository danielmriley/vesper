#include "vesper/weight.h"

#include "vesper/hip.h"
#include "vesper/q8.h"
#include "vesper/types.h"

#include <utility>

namespace vesper {

void WeightMatrix::release() {
    if (packed_hip_ != nullptr) {
        hip_free(packed_hip_);
        packed_hip_ = nullptr;
    }
    packed_host_.clear();
    f32_ = Buffer();
    packed_bytes_ = 0;
    rows_ = 0;
    cols_ = 0;
    device_ = Device::CPU;
    kind_ = WeightKind::F32;
}

void WeightMatrix::steal(WeightMatrix& other) noexcept {
    kind_ = other.kind_;
    rows_ = other.rows_;
    cols_ = other.cols_;
    device_ = other.device_;
    f32_ = std::move(other.f32_);
    packed_host_ = std::move(other.packed_host_);
    packed_hip_ = other.packed_hip_;
    packed_bytes_ = other.packed_bytes_;
    other.packed_hip_ = nullptr;
    other.packed_bytes_ = 0;
    other.rows_ = 0;
    other.cols_ = 0;
    other.device_ = Device::CPU;
    other.kind_ = WeightKind::F32;
}

WeightMatrix::WeightMatrix(const WeightMatrix& other)
    : kind_(other.kind_),
      rows_(other.rows_),
      cols_(other.cols_),
      device_(other.device_),
      packed_bytes_(other.packed_bytes_) {
    switch (other.kind_) {
        case WeightKind::F32:
            f32_ = other.f32_;
            return;
        case WeightKind::Q8_0:
            switch (other.device_) {
                case Device::CPU:
                    packed_host_ = other.packed_host_;
                    return;
                case Device::HIP:
                    if (other.packed_hip_ == nullptr || other.packed_bytes_ == 0) {
                        return;
                    }
                    packed_hip_ = hip_alloc(other.packed_bytes_);
                    hip_copy_d2d(packed_hip_, other.packed_hip_, other.packed_bytes_);
                    return;
            }
            throw std::logic_error("unhandled Device");
    }
    throw std::logic_error("unhandled WeightKind");
}

WeightMatrix::WeightMatrix(WeightMatrix&& other) noexcept {
    steal(other);
}

WeightMatrix& WeightMatrix::operator=(const WeightMatrix& other) {
    if (this == &other) {
        return *this;
    }
    WeightMatrix tmp(other);
    *this = std::move(tmp);
    return *this;
}

WeightMatrix& WeightMatrix::operator=(WeightMatrix&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    release();
    steal(other);
    return *this;
}

WeightMatrix::~WeightMatrix() {
    release();
}

GgmlType WeightMatrix::type() const {
    switch (kind_) {
        case WeightKind::F32:
            return GgmlType::F32;
        case WeightKind::Q8_0:
            return GgmlType::Q8_0;
    }
    throw std::logic_error("unhandled WeightKind");
}

WeightMatrix WeightMatrix::from_f32(Buffer data, int rows, int cols) {
    check(rows > 0 && cols > 0, "from_f32 empty shape");
    check(data.size() == static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols),
          "from_f32 size mismatch");
    WeightMatrix w;
    w.kind_ = WeightKind::F32;
    w.rows_ = rows;
    w.cols_ = cols;
    w.device_ = data.device();
    w.f32_ = std::move(data);
    return w;
}

WeightMatrix WeightMatrix::from_f32(const float* data, int rows, int cols) {
    check(data != nullptr, "from_f32 null");
    Buffer buf(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), Device::CPU);
    buf.copy_from(data, buf.size());
    return from_f32(std::move(buf), rows, cols);
}

WeightMatrix WeightMatrix::q8_from_f32(const float* data, int rows, int cols) {
    check(data != nullptr, "q8_from_f32 null");
    WeightMatrix w;
    w.kind_ = WeightKind::Q8_0;
    w.rows_ = rows;
    w.cols_ = cols;
    w.device_ = Device::CPU;
    w.packed_bytes_ = q8_packed_bytes(rows, cols);
    w.packed_host_.resize(w.packed_bytes_);
    quantize_q8(data, w.packed_host_.data(), rows, cols);
    return w;
}

WeightMatrix WeightMatrix::q8_from_bytes(const std::byte* data, int rows, int cols) {
    check(data != nullptr, "q8_from_bytes null");
    WeightMatrix w;
    w.kind_ = WeightKind::Q8_0;
    w.rows_ = rows;
    w.cols_ = cols;
    w.device_ = Device::CPU;
    w.packed_bytes_ = q8_packed_bytes(rows, cols);
    w.packed_host_.assign(data, data + w.packed_bytes_);
    return w;
}

WeightMatrix WeightMatrix::to(Device device) const {
    if (device == device_) {
        return *this;
    }
    WeightMatrix out;
    out.kind_ = kind_;
    out.rows_ = rows_;
    out.cols_ = cols_;
    out.device_ = device;
    out.packed_bytes_ = packed_bytes_;
    switch (kind_) {
        case WeightKind::F32:
            out.f32_ = f32_.to(device);
            return out;
        case WeightKind::Q8_0:
            switch (device) {
                case Device::HIP:
                    check(device_ == Device::CPU, "Q8 HIP upload expects CPU source");
                    if (packed_bytes_ == 0) {
                        return out;
                    }
                    out.packed_hip_ = hip_alloc(packed_bytes_);
                    hip_copy_h2d(out.packed_hip_, packed_host_.data(), packed_bytes_);
                    return out;
                case Device::CPU:
                    check(device_ == Device::HIP, "Q8 download expects HIP source");
                    out.packed_host_.resize(packed_bytes_);
                    if (packed_bytes_ > 0) {
                        hip_copy_d2h(out.packed_host_.data(), packed_hip_, packed_bytes_);
                    }
                    return out;
            }
            throw std::logic_error("unhandled Device");
    }
    throw std::logic_error("unhandled WeightKind");
}

WeightMatrix WeightMatrix::dequant_f32() const {
    check(device_ == Device::CPU, "dequant_f32 is CPU-only");
    switch (kind_) {
        case WeightKind::F32:
            return *this;
        case WeightKind::Q8_0: {
            std::vector<float> tmp(static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_));
            dequant_q8(tmp.data(), packed_host_.data(), rows_, cols_);
            return from_f32(tmp.data(), rows_, cols_);
        }
    }
    throw std::logic_error("unhandled WeightKind");
}

std::size_t WeightMatrix::bytes() const {
    switch (kind_) {
        case WeightKind::F32:
            return static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_) * sizeof(float);
        case WeightKind::Q8_0:
            return packed_bytes_;
    }
    throw std::logic_error("unhandled WeightKind");
}

const float* WeightMatrix::f32_data() const {
    check(kind_ == WeightKind::F32, "f32_data on non-F32 weight");
    return f32_.data();
}

const std::byte* WeightMatrix::packed() const {
    check(kind_ == WeightKind::Q8_0, "packed() on non-Q8 weight");
    switch (device_) {
        case Device::CPU:
            return packed_host_.data();
        case Device::HIP:
            return static_cast<const std::byte*>(packed_hip_);
    }
    throw std::logic_error("unhandled Device");
}

}  // namespace vesper
