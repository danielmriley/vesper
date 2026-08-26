#pragma once

#include <cstddef>
#include <cstdint>

namespace vesper {

inline constexpr int kQ4KBlockElems = 256;
inline constexpr int kQ4KBlockBytes = 144;
inline constexpr int kQ4KHeaderBytes = 16;
inline constexpr int kQ4KQsBytes = 128;
inline constexpr int kQ4KSubBlocks = 8;
inline constexpr int kQ4KSubElems = 32;

// HIP decode layout is matrix SoA, super-major: all headers, then all
// qs. Same 144 B per super as GGUF. Official 20/68-super rows have no pad.
inline constexpr std::size_t q4k_soa_row_bytes(int supers) {
    return static_cast<std::size_t>(supers) * static_cast<std::size_t>(kQ4KBlockBytes);
}

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

// Same byte count as GGUF. HIP upload uses this; CPU mmap stays interleaved.
void q4k_repack_soa(std::byte* dst, const std::byte* src, int rows, int cols);
void q4k_unpack_soa(std::byte* dst, const std::byte* src, int rows, int cols);

std::size_t q4k_packed_bytes(int rows, int cols);

}  // namespace vesper
