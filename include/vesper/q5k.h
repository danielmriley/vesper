#pragma once

#include <cstddef>
#include <cstdint>

namespace vesper {

inline constexpr int kQ5KBlockElems = 256;
inline constexpr int kQ5KBlockBytes = 176;

struct BlockQ5K {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t qh[32];
    std::uint8_t qs[128];
} __attribute__((packed));

static_assert(sizeof(BlockQ5K) == kQ5KBlockBytes, "Q5_K super-block is 176 bytes");

void quantize_q5k(const float* src, std::byte* packed, int rows, int cols);
void dequant_q5k(float* dst, const std::byte* packed, int rows, int cols);
void dequant_q5k_row(float* dst, const std::byte* packed, int row, int cols);
void gemv_q5k(float* y, const std::byte* packed, const float* x, int rows, int cols);
void gemv_q5k_q8x(float* y, const std::byte* packed, const std::int8_t* xq, const float* xd,
                  const float* xsum, int rows, int cols);

std::size_t q5k_packed_bytes(int rows, int cols);

}  // namespace vesper
