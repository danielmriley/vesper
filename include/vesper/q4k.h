#pragma once

#include <cstddef>
#include <cstdint>

namespace vesper {

inline constexpr int kQ4KBlockElems = 256;
inline constexpr int kQ4KBlockBytes = 144;
inline constexpr int kQ4KSubBlocks = 8;
inline constexpr int kQ4KSubElems = 32;

struct BlockQ4K {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t qs[128];
} __attribute__((packed));

static_assert(sizeof(BlockQ4K) == kQ4KBlockBytes, "Q4_K super-block is 144 bytes");

void quantize_q4k(const float* src, std::byte* packed, int rows, int cols);
void dequant_q4k(float* dst, const std::byte* packed, int rows, int cols);
void dequant_q4k_row(float* dst, const std::byte* packed, int row, int cols);
void gemv_q4k(float* y, const std::byte* packed, const float* x, int rows, int cols);
void gemv_q4k_q8x(float* y, const std::byte* packed, const std::int8_t* xq, const float* xd,
                  const float* xsum, int rows, int cols);

std::size_t q4k_packed_bytes(int rows, int cols);

}  // namespace vesper
