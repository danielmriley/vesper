#pragma once

#include <cstddef>
#include <cstdint>

namespace vesper {

inline constexpr int kQ8BlockElems = 32;
inline constexpr int kQ8BlockBytes = 34;

// HIP SoA is matrix-wide: all f16 scales (16-byte padded per row), then
// all 32-byte qs. Official K 5120/6144 has no pad, so the matrix is
// still 34 B * nblocks * rows.
inline constexpr int q8_soa_scale_bytes(int nblocks) {
    return (nblocks * 2 + 15) & ~15;
}

inline constexpr std::size_t q8_soa_row_bytes(int nblocks) {
    return static_cast<std::size_t>(q8_soa_scale_bytes(nblocks)) +
           static_cast<std::size_t>(nblocks) * static_cast<std::size_t>(kQ8BlockElems);
}

inline constexpr std::size_t q8_soa_bytes(int rows, int cols) {
    return static_cast<std::size_t>(rows) * q8_soa_row_bytes(cols / kQ8BlockElems);
}

static_assert(q8_soa_scale_bytes(160) == 320, "official Q8 K 5120 scale table is 16-aligned");
static_assert(q8_soa_row_bytes(160) == 160u * kQ8BlockBytes, "official Q8 K 5120 SoA is same size");
static_assert(q8_soa_row_bytes(192) == 192u * kQ8BlockBytes, "official Q8 K 6144 SoA is same size");
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

void q8_repack_soa(std::byte* dst, const std::byte* src, int rows, int cols);
void q8_unpack_soa(std::byte* dst, const std::byte* src, int rows, int cols);

std::size_t q8_packed_bytes(int rows, int cols);

}  // namespace vesper
