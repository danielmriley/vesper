#pragma once

#include "vesper/buffer.h"
#include "vesper/gguf.h"
#include "vesper/types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vesper {

enum class WeightKind : std::uint8_t {
    F32,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
};

class WeightMatrix {
public:
    WeightMatrix() = default;
    WeightMatrix(const WeightMatrix& other);
    WeightMatrix(WeightMatrix&& other) noexcept;
    WeightMatrix& operator=(const WeightMatrix& other);
    WeightMatrix& operator=(WeightMatrix&& other) noexcept;
    ~WeightMatrix();

    static WeightMatrix from_f32(Buffer data, int rows, int cols);
    static WeightMatrix from_f32(const float* data, int rows, int cols);
    static WeightMatrix q8_from_f32(const float* data, int rows, int cols);
    static WeightMatrix q8_from_bytes(const std::byte* data, int rows, int cols);
    static WeightMatrix q4_from_f32(const float* data, int rows, int cols);
    static WeightMatrix q4_from_bytes(const std::byte* data, int rows, int cols);
    static WeightMatrix q5_from_f32(const float* data, int rows, int cols);
    static WeightMatrix q5_from_bytes(const std::byte* data, int rows, int cols);
    static WeightMatrix q6_from_f32(const float* data, int rows, int cols);
    static WeightMatrix q6_from_bytes(const std::byte* data, int rows, int cols);

    WeightMatrix to(Device device) const;
    WeightMatrix dequant_f32() const;

    WeightKind kind() const { return kind_; }
    GgmlType type() const;
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    Device device() const { return device_; }
    std::size_t bytes() const;

    const float* f32_data() const;
    const std::byte* packed() const;

private:
    void release();
    void steal(WeightMatrix& other) noexcept;

    WeightKind kind_ = WeightKind::F32;
    int rows_ = 0;
    int cols_ = 0;
    Device device_ = Device::CPU;
    Buffer f32_;
    std::vector<std::byte> packed_host_;
    void* packed_hip_ = nullptr;
    std::size_t packed_bytes_ = 0;
};

}  // namespace vesper
