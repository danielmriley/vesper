#include "vesper/q5k.h"

#include "vesper/q8.h"
#include "vesper/types.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace vesper {
namespace {

int super_count(int cols) {
    check(cols > 0 && (cols % kQ5KBlockElems) == 0, "Q5_K cols must be a multiple of 256");
    return cols / kQ5KBlockElems;
}

void decode_scales(const std::uint8_t* packed, int* scales, int* mins) {
    for (int j = 0; j < 4; ++j) {
        scales[j] = packed[j] & 0x3f;
        mins[j] = packed[j + 4] & 0x3f;
    }
    for (int j = 4; j < 8; ++j) {
        scales[j] = (packed[j + 4] & 0x0f) | ((packed[j - 4] >> 6) << 4);
        mins[j] = (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
    }
}

void encode_scales(std::uint8_t* packed, const int* scales, const int* mins) {
    std::memset(packed, 0, 12);
    for (int j = 0; j < 4; ++j) {
        packed[j] = static_cast<std::uint8_t>(scales[j] & 0x3f);
        packed[j + 4] = static_cast<std::uint8_t>(mins[j] & 0x3f);
    }
    for (int j = 4; j < 8; ++j) {
        packed[j + 4] |= static_cast<std::uint8_t>(scales[j] & 0x0f);
        packed[j - 4] |= static_cast<std::uint8_t>((scales[j] & 0x30) << 2);
        packed[j + 4] |= static_cast<std::uint8_t>((mins[j] & 0x0f) << 4);
        packed[j] |= static_cast<std::uint8_t>((mins[j] & 0x30) << 2);
    }
}

void dequant_block(float* dst, const BlockQ5K& block) {
    int scales[8];
    int mins[8];
    decode_scales(block.scales, scales, mins);
    const float d = f16_to_f32(block.d);
    const float dmin = f16_to_f32(block.dmin);
    for (int il = 0; il < 4; ++il) {
        const int is = 2 * il;
        const float d1 = d * static_cast<float>(scales[is]);
        const float m1 = dmin * static_cast<float>(mins[is]);
        const float d2 = d * static_cast<float>(scales[is + 1]);
        const float m2 = dmin * static_cast<float>(mins[is + 1]);
        std::uint8_t hm = static_cast<std::uint8_t>(1u << (2 * il));
        const std::uint8_t* qs = block.qs + 32 * il;
        float* y = dst + 64 * il;
        for (int r = 0; r < 32; ++r) {
            const int qlo = (qs[r] & 0x0f) + ((block.qh[r] & hm) ? 16 : 0);
            y[r] = d1 * static_cast<float>(qlo) - m1;
        }
        hm = static_cast<std::uint8_t>(hm << 1);
        for (int r = 0; r < 32; ++r) {
            const int qhi = (qs[r] >> 4) + ((block.qh[r] & hm) ? 16 : 0);
            y[r + 32] = d2 * static_cast<float>(qhi) - m2;
        }
    }
}

}  // namespace

std::size_t q5k_packed_bytes(int rows, int cols) {
    check(rows > 0, "Q5_K rows must be positive");
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(super_count(cols)) *
           kQ5KBlockBytes;
}

void quantize_q5k(const float* src, std::byte* packed, int rows, int cols) {
    check(src != nullptr && packed != nullptr, "Q5_K quantize null pointer");
    const int supers = super_count(cols);
    auto* out = reinterpret_cast<BlockQ5K*>(packed);
    for (int r = 0; r < rows; ++r) {
        const float* row = src + static_cast<std::size_t>(r) * cols;
        for (int s = 0; s < supers; ++s) {
            const float* block_x = row + s * kQ5KBlockElems;
            float mins_f[8];
            float scales_f[8];
            std::uint8_t qs[kQ5KBlockElems];
            float max_scale = 0.0f;
            float max_min = 0.0f;
            for (int b = 0; b < 8; ++b) {
                const float* xs = block_x + b * 32;
                float lo = xs[0];
                float hi = xs[0];
                for (int i = 1; i < 32; ++i) {
                    lo = std::min(lo, xs[i]);
                    hi = std::max(hi, xs[i]);
                }
                mins_f[b] = lo;
                scales_f[b] = (hi - lo) / 31.0f;
                max_scale = std::max(max_scale, scales_f[b]);
                max_min = std::max(max_min, std::fabs(lo));
            }
            const float d = (max_scale > 0.0f) ? (max_scale / 63.0f) : 0.0f;
            const float dmin = (max_min > 0.0f) ? (max_min / 63.0f) : 0.0f;
            BlockQ5K block{};
            block.d = f32_to_f16(d);
            block.dmin = f32_to_f16(dmin);
            const float d_use = f16_to_f32(block.d);
            const float dmin_use = f16_to_f32(block.dmin);
            int scales_i[8];
            int mins_i[8];
            for (int b = 0; b < 8; ++b) {
                scales_i[b] = 0;
                mins_i[b] = 0;
                if (d_use > 0.0f) {
                    scales_i[b] =
                        std::max(0, std::min(63, static_cast<int>(std::lround(scales_f[b] / d_use))));
                }
                if (dmin_use > 0.0f) {
                    mins_i[b] = std::max(
                        0, std::min(63, static_cast<int>(std::lround(std::fabs(mins_f[b]) / dmin_use))));
                }
                const float sc = d_use * static_cast<float>(scales_i[b]);
                const float mn = dmin_use * static_cast<float>(mins_i[b]);
                const float* xs = block_x + b * 32;
                for (int i = 0; i < 32; ++i) {
                    float q = 0.0f;
                    if (sc > 0.0f) {
                        q = std::round((xs[i] + mn) / sc);
                    }
                    qs[b * 32 + i] =
                        static_cast<std::uint8_t>(std::max(0, std::min(31, static_cast<int>(q))));
                }
            }
            encode_scales(block.scales, scales_i, mins_i);
            std::memset(block.qh, 0, sizeof(block.qh));
            for (int il = 0; il < 4; ++il) {
                const std::uint8_t hm_lo = static_cast<std::uint8_t>(1u << (2 * il));
                const std::uint8_t hm_hi = static_cast<std::uint8_t>(hm_lo << 1);
                for (int r = 0; r < 32; ++r) {
                    const std::uint8_t qlo = qs[il * 64 + r];
                    const std::uint8_t qhi = qs[il * 64 + 32 + r];
                    block.qs[32 * il + r] =
                        static_cast<std::uint8_t>((qlo & 0x0f) | ((qhi & 0x0f) << 4));
                    if (qlo & 0x10) {
                        block.qh[r] |= hm_lo;
                    }
                    if (qhi & 0x10) {
                        block.qh[r] |= hm_hi;
                    }
                }
            }
            out[r * supers + s] = block;
        }
    }
}

void dequant_q5k_row(float* dst, const std::byte* packed, int row, int cols) {
    check(dst != nullptr && packed != nullptr, "Q5_K dequant row null pointer");
    check(row >= 0, "Q5_K dequant row index");
    const int supers = super_count(cols);
    const auto* in = reinterpret_cast<const BlockQ5K*>(packed) + row * supers;
    for (int s = 0; s < supers; ++s) {
        dequant_block(dst + s * kQ5KBlockElems, in[s]);
    }
}

void dequant_q5k(float* dst, const std::byte* packed, int rows, int cols) {
    check(dst != nullptr && packed != nullptr, "Q5_K dequant null pointer");
    for (int r = 0; r < rows; ++r) {
        dequant_q5k_row(dst + static_cast<std::size_t>(r) * cols, packed, r, cols);
    }
}

void gemv_q5k(float* y, const std::byte* packed, const float* x, int rows, int cols) {
    check(y != nullptr && packed != nullptr && x != nullptr, "Q5_K GEMV null pointer");
    const int supers = super_count(cols);
    const auto* in = reinterpret_cast<const BlockQ5K*>(packed);
    for (int r = 0; r < rows; ++r) {
        const BlockQ5K* row = in + r * supers;
        float acc = 0.0f;
        for (int s = 0; s < supers; ++s) {
            int scales[8];
            int mins[8];
            decode_scales(row[s].scales, scales, mins);
            const float d = f16_to_f32(row[s].d);
            const float dmin = f16_to_f32(row[s].dmin);
            const float* xb = x + s * kQ5KBlockElems;
            for (int il = 0; il < 4; ++il) {
                const int is = 2 * il;
                const float d1 = d * static_cast<float>(scales[is]);
                const float m1 = dmin * static_cast<float>(mins[is]);
                const float d2 = d * static_cast<float>(scales[is + 1]);
                const float m2 = dmin * static_cast<float>(mins[is + 1]);
                std::uint8_t hm = static_cast<std::uint8_t>(1u << (2 * il));
                const std::uint8_t* qs = row[s].qs + 32 * il;
                const float* xlo = xb + 64 * il;
                const float* xhi = xb + 64 * il + 32;
                for (int l = 0; l < 32; ++l) {
                    const int qlo = (qs[l] & 0x0f) + ((row[s].qh[l] & hm) ? 16 : 0);
                    acc += (d1 * static_cast<float>(qlo) - m1) * xlo[l];
                }
                hm = static_cast<std::uint8_t>(hm << 1);
                for (int l = 0; l < 32; ++l) {
                    const int qhi = (qs[l] >> 4) + ((row[s].qh[l] & hm) ? 16 : 0);
                    acc += (d2 * static_cast<float>(qhi) - m2) * xhi[l];
                }
            }
        }
        y[r] = acc;
    }
}

}  // namespace vesper
