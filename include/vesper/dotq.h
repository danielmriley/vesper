#pragma once

#include <cstdint>
#include <cstring>

namespace vesper {

// 4x int8 dot. gfx1201 uses v_dot4_i32_i8; host and other compiles use scalar.
inline int dp4a_i8(int a, int b, int c) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const int a0 = static_cast<int>(static_cast<signed char>(a & 0xff));
    const int a1 = static_cast<int>(static_cast<signed char>((a >> 8) & 0xff));
    const int a2 = static_cast<int>(static_cast<signed char>((a >> 16) & 0xff));
    const int a3 = static_cast<int>(static_cast<signed char>((a >> 24) & 0xff));
    const int b0 = static_cast<int>(static_cast<signed char>(b & 0xff));
    const int b1 = static_cast<int>(static_cast<signed char>((b >> 8) & 0xff));
    const int b2 = static_cast<int>(static_cast<signed char>((b >> 16) & 0xff));
    const int b3 = static_cast<int>(static_cast<signed char>((b >> 24) & 0xff));
    return c + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
#endif
}

inline int load_i32(const void* base, int n) {
#if defined(__HIP_DEVICE_COMPILE__)
    return reinterpret_cast<const int*>(base)[n];
#else
    int value = 0;
    std::memcpy(&value, static_cast<const char*>(base) + static_cast<std::size_t>(n) * 4u, 4);
    return value;
#endif
}

// One MMVQ slice of a Q4_K super-block against Q8_1 x. iqs is even in [0, 30].
// 16 slices cover the 256 weights. Matches llama.cpp vec_dot_q4_K_q8_1.
inline float q4k_dot_q8_iqs(const unsigned char* blk, float d, float dmin, const std::int8_t* xq,
                            const float* xd, int iqs) {
    const int bq8_offset = 2 * (iqs / 8);
    const unsigned char* qs = blk + 16;
    const int q4_index = (16 * bq8_offset + 4 * ((iqs / 2) % 4)) / 4;
    const int v0 = load_i32(qs, q4_index);
    const int v1 = load_i32(qs, q4_index + 4);

    const std::uint16_t* scales = reinterpret_cast<const std::uint16_t*>(blk + 4);
    std::uint16_t aux0 = 0;
    std::uint16_t aux1 = 0;
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux0 = static_cast<std::uint16_t>(scales[j + 0] & 0x3f3f);
        aux1 = static_cast<std::uint16_t>(scales[j + 2] & 0x3f3f);
    } else {
        aux0 = static_cast<std::uint16_t>(((scales[j + 2] >> 0) & 0x0f0f) |
                                          ((scales[j - 2] & 0xc0c0) >> 2));
        aux1 = static_cast<std::uint16_t>(((scales[j + 2] >> 4) & 0x0f0f) |
                                          ((scales[j - 0] & 0xc0c0) >> 2));
    }
    const std::uint8_t sc0 = static_cast<std::uint8_t>(aux0 & 0xff);
    const std::uint8_t sc1 = static_cast<std::uint8_t>(aux0 >> 8);
    const std::uint8_t m0 = static_cast<std::uint8_t>(aux1 & 0xff);
    const std::uint8_t m1 = static_cast<std::uint8_t>(aux1 >> 8);

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    const std::uint8_t sc[2] = {sc0, sc1};
    const std::uint8_t mn[2] = {m0, m1};
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v0 >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v1 >> (4 * i)) & 0x0f0f0f0f;
        const int q8_base = ((bq8_offset + i) * 32 + 4 * ((iqs / 2) % 4)) / 4;
        const int u0 = load_i32(xq, q8_base);
        const int u1 = load_i32(xq, q8_base + 4);
        const int dot1 = dp4a_i8(v1i, u1, dp4a_i8(v0i, u0, 0));
        const int dot2 = dp4a_i8(0x01010101, u1, dp4a_i8(0x01010101, u0, 0));
        const float d8 = xd[bq8_offset + i];
        sumf_d += d8 * static_cast<float>(dot1 * static_cast<int>(sc[i]));
        sumf_m += d8 * static_cast<float>(dot2 * static_cast<int>(mn[i]));
    }
    return d * sumf_d - dmin * sumf_m;
}

inline float q4k_dot_q8_super(const unsigned char* blk, float d, float dmin, const std::int8_t* xq,
                              const float* xd) {
    float acc = 0.0f;
    for (int t = 0; t < 16; ++t) {
        acc += q4k_dot_q8_iqs(blk, d, dmin, xq, xd, 2 * t);
    }
    return acc;
}

}  // namespace vesper
