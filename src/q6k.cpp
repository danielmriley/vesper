#include "vesper/q6k.h"

#include "vesper/q8.h"
#include "vesper/types.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace vesper {
namespace {

int super_count(int cols) {
    check(cols > 0 && (cols % kQ6KBlockElems) == 0, "Q6_K cols must be a multiple of 256");
    return cols / kQ6KBlockElems;
}

void dequant_block(float* dst, const BlockQ6K& block) {
    const float d = f16_to_f32(block.d);
    for (int ip = 0; ip < 2; ++ip) {
        for (int il = 0; il < 32; ++il) {
            const int is = 8 * ip + il / 16;
            const std::uint8_t ql0 = block.ql[64 * ip + il];
            const std::uint8_t ql1 = block.ql[64 * ip + 32 + il];
            const std::uint8_t qh = block.qh[32 * ip + il];
            dst[128 * ip + il + 0] =
                d * static_cast<float>(block.scales[is + 0]) *
                static_cast<float>(static_cast<int>((ql0 & 0x0f) | (((qh >> 0) & 3) << 4)) - 32);
            dst[128 * ip + il + 32] =
                d * static_cast<float>(block.scales[is + 2]) *
                static_cast<float>(static_cast<int>((ql1 & 0x0f) | (((qh >> 2) & 3) << 4)) - 32);
            dst[128 * ip + il + 64] =
                d * static_cast<float>(block.scales[is + 4]) *
                static_cast<float>(static_cast<int>((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
            dst[128 * ip + il + 96] =
                d * static_cast<float>(block.scales[is + 6]) *
                static_cast<float>(static_cast<int>((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32);
        }
    }
}

}  // namespace

std::size_t q6k_packed_bytes(int rows, int cols) {
    check(rows > 0, "Q6_K rows must be positive");
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(super_count(cols)) *
           kQ6KBlockBytes;
}

void quantize_q6k(const float* src, std::byte* packed, int rows, int cols) {
    check(src != nullptr && packed != nullptr, "Q6_K quantize null pointer");
    const int supers = super_count(cols);
    auto* out = reinterpret_cast<BlockQ6K*>(packed);
    for (int r = 0; r < rows; ++r) {
        const float* row = src + static_cast<std::size_t>(r) * cols;
        for (int s = 0; s < supers; ++s) {
            const float* block_x = row + s * kQ6KBlockElems;
            float scales_f[16];
            std::int8_t qs[kQ6KBlockElems];
            float max_scale = 0.0f;
            for (int b = 0; b < 16; ++b) {
                const float* xs = block_x + b * 16;
                float amax = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    amax = std::max(amax, std::fabs(xs[i]));
                }
                scales_f[b] = amax / 31.0f;
                max_scale = std::max(max_scale, scales_f[b]);
            }
            const float d = (max_scale > 0.0f) ? (max_scale / 127.0f) : 0.0f;
            BlockQ6K block{};
            block.d = f32_to_f16(d);
            const float d_use = f16_to_f32(block.d);
            for (int b = 0; b < 16; ++b) {
                int sc = 0;
                if (d_use > 0.0f) {
                    sc = std::max(-128, std::min(127, static_cast<int>(std::lround(scales_f[b] / d_use))));
                }
                block.scales[b] = static_cast<std::int8_t>(sc);
                const float scale = d_use * static_cast<float>(block.scales[b]);
                const float* xs = block_x + b * 16;
                for (int i = 0; i < 16; ++i) {
                    int q = 0;
                    if (scale != 0.0f) {
                        q = static_cast<int>(std::lround(xs[i] / scale));
                    }
                    q = std::max(-32, std::min(31, q));
                    qs[b * 16 + i] = static_cast<std::int8_t>(q + 32);
                }
            }
            std::memset(block.ql, 0, sizeof(block.ql));
            std::memset(block.qh, 0, sizeof(block.qh));
            for (int ip = 0; ip < 2; ++ip) {
                for (int il = 0; il < 32; ++il) {
                    const int q0 = qs[128 * ip + il + 0] & 0x3f;
                    const int q1 = qs[128 * ip + il + 32] & 0x3f;
                    const int q2 = qs[128 * ip + il + 64] & 0x3f;
                    const int q3 = qs[128 * ip + il + 96] & 0x3f;
                    block.ql[64 * ip + il] =
                        static_cast<std::uint8_t>((q0 & 0x0f) | ((q2 & 0x0f) << 4));
                    block.ql[64 * ip + 32 + il] =
                        static_cast<std::uint8_t>((q1 & 0x0f) | ((q3 & 0x0f) << 4));
                    block.qh[32 * ip + il] = static_cast<std::uint8_t>(
                        ((q0 >> 4) & 3) | (((q1 >> 4) & 3) << 2) | (((q2 >> 4) & 3) << 4) |
                        (((q3 >> 4) & 3) << 6));
                }
            }
            out[r * supers + s] = block;
        }
    }
}

void dequant_q6k_row(float* dst, const std::byte* packed, int row, int cols) {
    check(dst != nullptr && packed != nullptr, "Q6_K dequant row null pointer");
    check(row >= 0, "Q6_K dequant row index");
    const int supers = super_count(cols);
    const auto* in = reinterpret_cast<const BlockQ6K*>(packed) + row * supers;
    for (int s = 0; s < supers; ++s) {
        dequant_block(dst + s * kQ6KBlockElems, in[s]);
    }
}

void dequant_q6k(float* dst, const std::byte* packed, int rows, int cols) {
    check(dst != nullptr && packed != nullptr, "Q6_K dequant null pointer");
    for (int r = 0; r < rows; ++r) {
        dequant_q6k_row(dst + static_cast<std::size_t>(r) * cols, packed, r, cols);
    }
}

void gemv_q6k(float* y, const std::byte* packed, const float* x, int rows, int cols) {
    check(y != nullptr && packed != nullptr && x != nullptr, "Q6_K GEMV null pointer");
    const int supers = super_count(cols);
    const auto* in = reinterpret_cast<const BlockQ6K*>(packed);
    for (int r = 0; r < rows; ++r) {
        const BlockQ6K* row = in + r * supers;
        float acc = 0.0f;
        for (int s = 0; s < supers; ++s) {
            const float d = f16_to_f32(row[s].d);
            const float* xb = x + s * kQ6KBlockElems;
            for (int ip = 0; ip < 2; ++ip) {
                for (int il = 0; il < 32; ++il) {
                    const int is = 8 * ip + il / 16;
                    const std::uint8_t ql0 = row[s].ql[64 * ip + il];
                    const std::uint8_t ql1 = row[s].ql[64 * ip + 32 + il];
                    const std::uint8_t qh = row[s].qh[32 * ip + il];
                    const int q0 =
                        static_cast<int>((ql0 & 0x0f) | (((qh >> 0) & 3) << 4)) - 32;
                    const int q1 =
                        static_cast<int>((ql1 & 0x0f) | (((qh >> 2) & 3) << 4)) - 32;
                    const int q2 =
                        static_cast<int>((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32;
                    const int q3 =
                        static_cast<int>((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32;
                    const float* xb0 = xb + 128 * ip + il;
                    acc += d * static_cast<float>(row[s].scales[is + 0]) *
                           static_cast<float>(q0) * xb0[0];
                    acc += d * static_cast<float>(row[s].scales[is + 2]) *
                           static_cast<float>(q1) * xb0[32];
                    acc += d * static_cast<float>(row[s].scales[is + 4]) *
                           static_cast<float>(q2) * xb0[64];
                    acc += d * static_cast<float>(row[s].scales[is + 6]) *
                           static_cast<float>(q3) * xb0[96];
                }
            }
        }
        y[r] = acc;
    }
}

}  // namespace vesper
