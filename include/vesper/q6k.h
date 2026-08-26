#pragma once

#include <cstddef>
#include <cstdint>

namespace vesper {

inline constexpr int kQ6KBlockElems = 256;
inline constexpr int kQ6KBlockBytes = 210;
inline constexpr int kQ6KQlBytes = 128;
inline constexpr int kQ6KQhBytes = 64;
inline constexpr int kQ6KScaleBytes = 16;
inline constexpr int kQ6KQi = 32;
inline constexpr int kQ6KQr = 2;

// HIP SoA: row-major padded f16 d, then super-major scales, ql, qh.
// Official o_proj (24 supers) is the same 210 B/super. Official lm_head
// (20 supers) pads d by 8 B/row.
inline constexpr int q6k_soa_d_bytes(int supers) {
    return (supers * 2 + 15) & ~15;
}

inline constexpr std::size_t q6k_soa_row_bytes(int supers) {
    return static_cast<std::size_t>(q6k_soa_d_bytes(supers)) +
           static_cast<std::size_t>(supers) *
               (kQ6KScaleBytes + kQ6KQlBytes + kQ6KQhBytes);
}

inline constexpr std::size_t q6k_soa_bytes(int rows, int cols) {
    return static_cast<std::size_t>(rows) * q6k_soa_row_bytes(cols / kQ6KBlockElems);
}

static_assert(q6k_soa_row_bytes(24) == 24u * kQ6KBlockBytes, "official o_proj Q6 SoA is same size");
static_assert(q6k_soa_row_bytes(20) == 4208u, "official lm_head Q6 SoA pads d to 16 B");

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

void q6k_repack_soa(std::byte* dst, const std::byte* src, int rows, int cols);
void q6k_unpack_soa(std::byte* dst, const std::byte* src, int rows, int cols);

std::size_t q6k_packed_bytes(int rows, int cols);

}  // namespace vesper
