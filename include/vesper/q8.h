#pragma once

#include <cstddef>
#include <cstdint>

namespace vesper {

inline constexpr int kQ8BlockElems = 32;
inline constexpr int kQ8BlockBytes = 34;
// llama.cpp MMVQ: QI8_0=8 ints/block, VDR_Q8_0_Q8_1_MMVQ=2. One thread
// covers the block so official K=5120 (160 blocks) is 1 K-trip.
inline constexpr int kQ8Qi = 8;
inline constexpr int kQ8VdrMmvq = 2;

struct BlockQ80 {
    std::uint16_t d;
    std::int8_t qs[kQ8BlockElems];
} __attribute__((packed));

static_assert(sizeof(BlockQ80) == kQ8BlockBytes, "Q8_0 block is 34 bytes");

std::uint16_t f32_to_f16(float value);
float f16_to_f32(std::uint16_t h);

void quantize_q8(const float* src, std::byte* packed, int rows, int cols);
void dequant_q8(float* dst, const std::byte* packed, int rows, int cols);
void dequant_q8_row(float* dst, const std::byte* packed, int row, int cols);
void gemv_q8(float* y, const std::byte* packed, const float* x, int rows, int cols);
void gemv_q8_q8x(float* y, const std::byte* packed, const std::int8_t* xq, const float* xd, int rows,
                 int cols);

std::size_t q8_packed_bytes(int rows, int cols);

}  // namespace vesper
