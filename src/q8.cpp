#include "vesper/q8.h"

#include "vesper/types.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vesper {
namespace {

int block_count(int cols) {
    check(cols > 0 && (cols % kQ8BlockElems) == 0, "Q8_0 cols must be a multiple of 32");
    return cols / kQ8BlockElems;
}

}  // namespace

std::uint16_t f32_to_f16(float value) {
    std::uint32_t x = 0;
    std::memcpy(&x, &value, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const std::uint32_t absx = x & 0x7fffffffu;
    if (absx >= 0x7f800000u) {
        const std::uint16_t nan = static_cast<std::uint16_t>(sign | 0x7e00u);
        const std::uint16_t inf = static_cast<std::uint16_t>(sign | 0x7c00u);
        return (absx > 0x7f800000u) ? nan : inf;
    }
    if (absx > 0x477fe000u) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (absx < 0x38800000u) {
        if (absx < 0x33000000u) {
            return static_cast<std::uint16_t>(sign);
        }
        const std::uint32_t man = (absx & 0x7fffffu) | 0x800000u;
        const int shift = static_cast<int>(113u - (absx >> 23));
        const std::uint32_t rounded = (man + (1u << (shift - 1))) >> shift;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const std::uint32_t bits = absx - 0x38000000u;
    return static_cast<std::uint16_t>(sign | ((bits + 0x1000u) >> 13));
}

float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000u) << 16);
    const std::uint32_t exp = (h >> 10) & 0x1fu;
    const std::uint32_t man = h & 0x3ffu;
    std::uint32_t out = 0;
    if (exp == 0) {
        if (man == 0) {
            out = sign;
        } else {
            std::uint32_t m = man;
            std::uint32_t e = 1;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x3ffu;
            out = sign | ((e + 127 - 15) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (man << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

std::size_t q8_packed_bytes(int rows, int cols) {
    check(rows > 0, "Q8_0 rows must be positive");
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(block_count(cols)) *
           kQ8BlockBytes;
}

void quantize_q8(const float* src, std::byte* packed, int rows, int cols) {
    check(src != nullptr && packed != nullptr, "Q8 quantize null pointer");
    const int blocks = block_count(cols);
    auto* out = reinterpret_cast<BlockQ80*>(packed);
    for (int r = 0; r < rows; ++r) {
        const float* row = src + static_cast<std::size_t>(r) * cols;
        for (int b = 0; b < blocks; ++b) {
            const float* xs = row + b * kQ8BlockElems;
            float amax = 0.0f;
            for (int k = 0; k < kQ8BlockElems; ++k) {
                amax = std::max(amax, std::fabs(xs[k]));
            }
            BlockQ80 block{};
            if (amax == 0.0f) {
                block.d = f32_to_f16(0.0f);
            } else {
                const float d = amax / 127.0f;
                block.d = f32_to_f16(d);
                const float inv = (d > 0.0f) ? (1.0f / f16_to_f32(block.d)) : 0.0f;
                for (int k = 0; k < kQ8BlockElems; ++k) {
                    const float q = std::round(xs[k] * inv);
                    const int iq = std::max(-127, std::min(127, static_cast<int>(q)));
                    block.qs[k] = static_cast<std::int8_t>(iq);
                }
            }
            out[r * blocks + b] = block;
        }
    }
}

void dequant_q8_row(float* dst, const std::byte* packed, int row, int cols) {
    check(dst != nullptr && packed != nullptr, "Q8 dequant row null pointer");
    check(row >= 0, "Q8 dequant row index");
    const int blocks = block_count(cols);
    const auto* in = reinterpret_cast<const BlockQ80*>(packed) + row * blocks;
    for (int b = 0; b < blocks; ++b) {
        const float d = f16_to_f32(in[b].d);
        float* out = dst + b * kQ8BlockElems;
        for (int k = 0; k < kQ8BlockElems; ++k) {
            out[k] = d * static_cast<float>(in[b].qs[k]);
        }
    }
}

void dequant_q8(float* dst, const std::byte* packed, int rows, int cols) {
    check(dst != nullptr && packed != nullptr, "Q8 dequant null pointer");
    for (int r = 0; r < rows; ++r) {
        dequant_q8_row(dst + static_cast<std::size_t>(r) * cols, packed, r, cols);
    }
}

void gemv_q8(float* y, const std::byte* packed, const float* x, int rows, int cols) {
    check(y != nullptr && packed != nullptr && x != nullptr, "Q8 GEMV null pointer");
    const int blocks = block_count(cols);
    const auto* in = reinterpret_cast<const BlockQ80*>(packed);
    for (int r = 0; r < rows; ++r) {
        const BlockQ80* row = in + r * blocks;
        float acc = 0.0f;
        for (int b = 0; b < blocks; ++b) {
            const float d = f16_to_f32(row[b].d);
            const float* xb = x + b * kQ8BlockElems;
            float local = 0.0f;
            for (int k = 0; k < kQ8BlockElems; ++k) {
                local += static_cast<float>(row[b].qs[k]) * xb[k];
            }
            acc += d * local;
        }
        y[r] = acc;
    }
}

}  // namespace vesper
