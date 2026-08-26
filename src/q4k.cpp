#include "vesper/q4k.h"

#include "vesper/q8.h"
#include "vesper/types.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vesper {
namespace {

int super_count(int cols) {
    check(cols > 0 && (cols % kQ4KBlockElems) == 0, "Q4_K cols must be a multiple of 256");
    return cols / kQ4KBlockElems;
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

void dequant_block(float* dst, const BlockQ4K& block) {
    int scales[kQ4KSubBlocks];
    int mins[kQ4KSubBlocks];
    decode_scales(block.scales, scales, mins);
    const float d = f16_to_f32(block.d);
    const float dmin = f16_to_f32(block.dmin);
    for (int g = 0; g < 4; ++g) {
        const int lo = 2 * g;
        const int hi = 2 * g + 1;
        const float sc_lo = d * static_cast<float>(scales[lo]);
        const float sc_hi = d * static_cast<float>(scales[hi]);
        const float mn_lo = dmin * static_cast<float>(mins[lo]);
        const float mn_hi = dmin * static_cast<float>(mins[hi]);
        const std::uint8_t* qs = block.qs + g * 32;
        float* out_lo = dst + lo * kQ4KSubElems;
        float* out_hi = dst + hi * kQ4KSubElems;
        for (int l = 0; l < 32; ++l) {
            const std::uint8_t byte = qs[l];
            out_lo[l] = sc_lo * static_cast<float>(byte & 0x0f) - mn_lo;
            out_hi[l] = sc_hi * static_cast<float>(byte >> 4) - mn_hi;
        }
    }
}

}  // namespace

std::size_t q4k_packed_bytes(int rows, int cols) {
    check(rows > 0, "Q4_K rows must be positive");
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(super_count(cols)) *
           kQ4KBlockBytes;
}

void quantize_q4k(const float* src, std::byte* packed, int rows, int cols) {
    check(src != nullptr && packed != nullptr, "Q4_K quantize null pointer");
    const int supers = super_count(cols);
    auto* out = reinterpret_cast<BlockQ4K*>(packed);
    for (int r = 0; r < rows; ++r) {
        const float* row = src + static_cast<std::size_t>(r) * cols;
        for (int s = 0; s < supers; ++s) {
            const float* block_x = row + s * kQ4KBlockElems;
            float mins_f[kQ4KSubBlocks];
            float scales_f[kQ4KSubBlocks];
            std::uint8_t qs[kQ4KBlockElems];
            float max_scale = 0.0f;
            float max_min = 0.0f;
            for (int b = 0; b < kQ4KSubBlocks; ++b) {
                const float* xs = block_x + b * kQ4KSubElems;
                float lo = xs[0];
                float hi = xs[0];
                for (int i = 1; i < kQ4KSubElems; ++i) {
                    lo = std::min(lo, xs[i]);
                    hi = std::max(hi, xs[i]);
                }
                mins_f[b] = lo;
                scales_f[b] = (hi - lo) / 15.0f;
                max_scale = std::max(max_scale, scales_f[b]);
                max_min = std::max(max_min, std::fabs(lo));
            }
            const float d = (max_scale > 0.0f) ? (max_scale / 63.0f) : 0.0f;
            const float dmin = (max_min > 0.0f) ? (max_min / 63.0f) : 0.0f;
            BlockQ4K block{};
            block.d = f32_to_f16(d);
            block.dmin = f32_to_f16(dmin);
            const float d_use = f16_to_f32(block.d);
            const float dmin_use = f16_to_f32(block.dmin);
            int scales_i[kQ4KSubBlocks];
            int mins_i[kQ4KSubBlocks];
            for (int b = 0; b < kQ4KSubBlocks; ++b) {
                scales_i[b] = 0;
                mins_i[b] = 0;
                if (d_use > 0.0f) {
                    scales_i[b] = std::max(0, std::min(63, static_cast<int>(std::lround(scales_f[b] / d_use))));
                }
                if (dmin_use > 0.0f) {
                    mins_i[b] = std::max(0, std::min(63, static_cast<int>(std::lround(std::fabs(mins_f[b]) / dmin_use))));
                }
                const float sc = d_use * static_cast<float>(scales_i[b]);
                const float mn = dmin_use * static_cast<float>(mins_i[b]);
                const float* xs = block_x + b * kQ4KSubElems;
                for (int i = 0; i < kQ4KSubElems; ++i) {
                    float q = 0.0f;
                    if (sc > 0.0f) {
                        q = std::round((xs[i] + mn) / sc);
                    }
                    qs[b * kQ4KSubElems + i] =
                        static_cast<std::uint8_t>(std::max(0, std::min(15, static_cast<int>(q))));
                }
            }
            encode_scales(block.scales, scales_i, mins_i);
            for (int g = 0; g < 4; ++g) {
                const int lo = 2 * g;
                const int hi = 2 * g + 1;
                for (int l = 0; l < 32; ++l) {
                    const std::uint8_t qlo = qs[lo * 32 + l];
                    const std::uint8_t qhi = qs[hi * 32 + l];
                    block.qs[g * 32 + l] = static_cast<std::uint8_t>(qlo | (qhi << 4));
                }
            }
            out[r * supers + s] = block;
        }
    }
}

void dequant_q4k_row(float* dst, const std::byte* packed, int row, int cols) {
    check(dst != nullptr && packed != nullptr, "Q4_K dequant row null pointer");
    check(row >= 0, "Q4_K dequant row index");
    const int supers = super_count(cols);
    const auto* in = reinterpret_cast<const BlockQ4K*>(packed) + row * supers;
    for (int s = 0; s < supers; ++s) {
        dequant_block(dst + s * kQ4KBlockElems, in[s]);
    }
}

void dequant_q4k(float* dst, const std::byte* packed, int rows, int cols) {
    check(dst != nullptr && packed != nullptr, "Q4_K dequant null pointer");
    for (int r = 0; r < rows; ++r) {
        dequant_q4k_row(dst + static_cast<std::size_t>(r) * cols, packed, r, cols);
    }
}

void gemv_q4k(float* y, const std::byte* packed, const float* x, int rows, int cols) {
    check(y != nullptr && packed != nullptr && x != nullptr, "Q4_K GEMV null pointer");
    const int supers = super_count(cols);
    const auto* in = reinterpret_cast<const BlockQ4K*>(packed);
    for (int r = 0; r < rows; ++r) {
        const BlockQ4K* row = in + r * supers;
        float acc = 0.0f;
        for (int s = 0; s < supers; ++s) {
            int scales[kQ4KSubBlocks];
            int mins[kQ4KSubBlocks];
            decode_scales(row[s].scales, scales, mins);
            const float d = f16_to_f32(row[s].d);
            const float dmin = f16_to_f32(row[s].dmin);
            const float* xb = x + s * kQ4KBlockElems;
            for (int g = 0; g < 4; ++g) {
                const int lo = 2 * g;
                const int hi = 2 * g + 1;
                const float sc_lo = d * static_cast<float>(scales[lo]);
                const float sc_hi = d * static_cast<float>(scales[hi]);
                const float mn_lo = dmin * static_cast<float>(mins[lo]);
                const float mn_hi = dmin * static_cast<float>(mins[hi]);
                const std::uint8_t* qs = row[s].qs + g * 32;
                const float* xlo = xb + lo * 32;
                const float* xhi = xb + hi * 32;
                for (int l = 0; l < 32; ++l) {
                    const std::uint8_t byte = qs[l];
                    acc += (sc_lo * static_cast<float>(byte & 0x0f) - mn_lo) * xlo[l];
                    acc += (sc_hi * static_cast<float>(byte >> 4) - mn_hi) * xhi[l];
                }
            }
        }
        y[r] = acc;
    }
}

void gemv_q4k_q8x(float* y, const std::byte* packed, const std::int8_t* xq, const float* xd,
                  const float* xsum, int rows, int cols) {
    check(y != nullptr && packed != nullptr && xq != nullptr && xd != nullptr && xsum != nullptr,
          "Q4_K q8x GEMV null pointer");
    const int supers = super_count(cols);
    const auto* in = reinterpret_cast<const BlockQ4K*>(packed);
    for (int r = 0; r < rows; ++r) {
        const BlockQ4K* row = in + r * supers;
        float acc = 0.0f;
        for (int s = 0; s < supers; ++s) {
            int scales[kQ4KSubBlocks];
            int mins[kQ4KSubBlocks];
            decode_scales(row[s].scales, scales, mins);
            const float d = f16_to_f32(row[s].d);
            const float dmin = f16_to_f32(row[s].dmin);
            for (int g = 0; g < 4; ++g) {
                const int lo = 2 * g;
                const int hi = 2 * g + 1;
                const float sc_lo = d * static_cast<float>(scales[lo]);
                const float sc_hi = d * static_cast<float>(scales[hi]);
                const float mn_lo = dmin * static_cast<float>(mins[lo]);
                const float mn_hi = dmin * static_cast<float>(mins[hi]);
                const std::uint8_t* qs = row[s].qs + g * 32;
                const std::int8_t* q8lo = xq + (s * 8 + lo) * 32;
                const std::int8_t* q8hi = xq + (s * 8 + hi) * 32;
                int dot_lo = 0;
                int dot_hi = 0;
                for (int l = 0; l < 32; ++l) {
                    const std::uint8_t byte = qs[l];
                    dot_lo += static_cast<int>(byte & 0x0f) * static_cast<int>(q8lo[l]);
                    dot_hi += static_cast<int>(byte >> 4) * static_cast<int>(q8hi[l]);
                }
                acc += sc_lo * xd[s * 8 + lo] * static_cast<float>(dot_lo) - mn_lo * xsum[s * 8 + lo];
                acc += sc_hi * xd[s * 8 + hi] * static_cast<float>(dot_hi) - mn_hi * xsum[s * 8 + hi];
            }
        }
        y[r] = acc;
    }
}

}  // namespace vesper
