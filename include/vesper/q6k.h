#pragma once

#include <cstddef>
#include <cstdint>

namespace vesper {

inline constexpr int kQ6KBlockElems = 256;
inline constexpr int kQ6KBlockBytes = 210;

struct BlockQ6K {
    std::uint8_t ql[128];
    std::uint8_t qh[64];
    std::int8_t scales[16];
    std::uint16_t d;
} __attribute__((packed));

static_assert(sizeof(BlockQ6K) == kQ6KBlockBytes, "Q6_K super-block is 210 bytes");

void quantize_q6k(const float* src, std::byte* packed, int rows, int cols);
void dequant_q6k(float* dst, const std::byte* packed, int rows, int cols);
void dequant_q6k_row(float* dst, const std::byte* packed, int row, int cols);
void gemv_q6k(float* y, const std::byte* packed, const float* x, int rows, int cols);
void gemv_q6k_q8x(float* y, const std::byte* packed, const std::int8_t* xq, const float* xd,
                  int rows, int cols);

std::size_t q6k_packed_bytes(int rows, int cols);

}  // namespace vesper
