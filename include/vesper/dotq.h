#pragma once

#include <cstdint>
#include <cstring>

namespace vesper {

// 4x int8 dot. llama.cpp RDNA3/4 uses sudot4 -> v_dot4_i32_iu8.
// sdot4 is the CDNA/RDNA2 builtin and may not map on gfx1201.
inline int dp4a_i8(int a, int b, int c) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1201__) || defined(__GFX12__))
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
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

// Cached 8-byte load of Q8_1 x. iqs is even, so this is aligned.
// Weights stay on load_w32. Do not stream x.
inline void load_i32x2(const void* base, int n, int* a, int* b) {
#if defined(__HIP_DEVICE_COMPILE__)
    const int2 v = reinterpret_cast<const int2*>(static_cast<const int*>(base) + n)[0];
    *a = v.x;
    *b = v.y;
#else
    *a = load_i32(base, n);
    *b = load_i32(base, n + 1);
#endif
}

inline void load_f32x2(const float* base, int n, float* a, float* b) {
#if defined(__HIP_DEVICE_COMPILE__)
    const float2 v = reinterpret_cast<const float2*>(base + n)[0];
    *a = v.x;
    *b = v.y;
#else
    *a = base[n];
    *b = base[n + 1];
#endif
}

// Weight-side int32. gfx1201 decode GEMV reads each Q4/Q8/Q6 word once.
// A streaming load keeps Q8_1 x in L2 while 18 GB of weights pass through.
// Do not use this for x. load_i32 stays cached.
inline int load_w32(const void* base, int n) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1201__) || defined(__GFX12__)) && \
    defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
    return __builtin_nontemporal_load(reinterpret_cast<const int*>(base) + n);
#else
    return load_i32(base, n);
#endif
}

// One MMVQ slice of a Q4_K super-block against Q8_1 x. iqs is even in [0, 30].
// 16 slices cover the 256 weights. Matches llama.cpp vec_dot_q4_K_q8_1.
inline float q4k_dot_q8_iqs(const unsigned char* blk, float d, float dmin, const std::int8_t* xq,
                            const float* xd, int iqs) {
    const int bq8_offset = 2 * (iqs / 8);
    const unsigned char* qs = blk + 16;
    const int q4_index = (16 * bq8_offset + 4 * ((iqs / 2) % 4)) / 4;
    const int v0 = load_w32(qs, q4_index);
    const int v1 = load_w32(qs, q4_index + 4);

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
    float d8_0 = 0.0f;
    float d8_1 = 0.0f;
    load_f32x2(xd, bq8_offset, &d8_0, &d8_1);
    const float d8s[2] = {d8_0, d8_1};
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v0 >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v1 >> (4 * i)) & 0x0f0f0f0f;
        const int q8_base = ((bq8_offset + i) * 32 + 4 * ((iqs / 2) % 4)) / 4;
        const int u0 = load_i32(xq, q8_base);
        const int u1 = load_i32(xq, q8_base + 4);
        const int dot1 = dp4a_i8(v1i, u1, dp4a_i8(v0i, u0, 0));
        const int dot2 = dp4a_i8(0x01010101, u1, dp4a_i8(0x01010101, u0, 0));
        const float d8 = d8s[i];
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

// Q8_0 qs starts at byte 2 of a 34-byte block: 2-byte aligned, not 4.
// llama.cpp get_int_b2. Two uint16 loads beat four byte loads on gfx1201.
inline int load_i32_b2(const void* base, int i32) {
    const auto* x16 = static_cast<const std::uint16_t*>(base);
    const unsigned lo = x16[2 * i32];
    const unsigned hi = x16[2 * i32 + 1];
    return static_cast<int>(lo | (hi << 16));
}

inline int load_w32_b2(const void* base, int i32) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1201__) || defined(__GFX12__)) && \
    defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
    const auto* x16 = static_cast<const std::uint16_t*>(base);
    const unsigned lo = __builtin_nontemporal_load(x16 + 2 * i32);
    const unsigned hi = __builtin_nontemporal_load(x16 + 2 * i32 + 1);
    return static_cast<int>(lo | (hi << 16));
#else
    return load_i32_b2(base, i32);
#endif
}

// VDR=2 slice. iqs is the starting int index in {0,2,4,6}.
inline float q8_dot_q8_iqs(const std::int8_t* qs, float d, const std::int8_t* xq, float xd, int iqs) {
    int u0 = 0;
    int u1 = 0;
    load_i32x2(xq, iqs, &u0, &u1);
    const int sumi = dp4a_i8(load_w32_b2(qs, iqs + 1), u1, dp4a_i8(load_w32_b2(qs, iqs), u0, 0));
    return d * xd * static_cast<float>(sumi);
}

// One Q8_0 block (32 i8) against one Q8_1 x block. llama.cpp vec_dot_q8_0_q8_1.
inline float q8_dot_q8(const std::int8_t* qs, float d, const std::int8_t* xq, float xd) {
    int sumi = 0;
    for (int i = 0; i < 8; i += 2) {
        int u0 = 0;
        int u1 = 0;
        load_i32x2(xq, i, &u0, &u1);
        sumi = dp4a_i8(load_w32_b2(qs, i + 1), u1, dp4a_i8(load_w32_b2(qs, i), u0, sumi));
    }
    return d * xd * static_cast<float>(sumi);
}

// Per-byte -32 without a 32-bit borrow. llama.cpp HIP uses __vsubss4
// (i8x4 + elementwise_sub_sat). Q6 bytes are 0..63, so sat never fires.
inline int q6k_sub32_bytes(int q) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__has_builtin) && \
    __has_builtin(__builtin_elementwise_sub_sat) && __has_builtin(__builtin_bit_cast)
    using i8x4 = signed char __attribute__((ext_vector_type(4)));
    const i8x4 va = __builtin_bit_cast(i8x4, q);
    const i8x4 vb = i8x4{32, 32, 32, 32};
    return __builtin_bit_cast(int, __builtin_elementwise_sub_sat(va, vb));
#else
    const unsigned u = static_cast<unsigned>(q);
    const unsigned r0 = ((u & 0xffu) - 32u) & 0xffu;
    const unsigned r1 = (((u >> 8) & 0xffu) - 32u) & 0xffu;
    const unsigned r2 = (((u >> 16) & 0xffu) - 32u) & 0xffu;
    const unsigned r3 = (((u >> 24) & 0xffu) - 32u) & 0xffu;
    return static_cast<int>(r0 | (r1 << 8) | (r2 << 16) | (r3 << 24));
#endif
}

// llama.cpp vec_dot_q6_K_q8_1. QI6_K=32, QR6_K=2, VDR_MMVQ=1.
// iqs is the lane in [0, 31].
inline int q6k_pack_vi(int vl, int vh, int i) {
    const unsigned q = static_cast<unsigned>(((vl >> (4 * i)) & 0x0f0f0f0f) |
                                             (((vh >> (4 * i)) << 4) & 0x30303030));
    return q6k_sub32_bytes(static_cast<int>(q));
}

inline float q6k_dot_q8_iqs(const unsigned char* blk, float d, const std::int8_t* xq,
                            const float* xd, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_w32_b2(blk, iqs);
    const int vh = load_w32_b2(blk + 128, 8 * (iqs / 16) + (iqs % 8)) >> vh_shift;
    const signed char* scales = reinterpret_cast<const signed char*>(blk + 192);
    float sumf = 0.0f;
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[scale_offset + 4 * i];
        const int u = load_i32(xq + (bq8_offset + 2 * i) * 32, iqs % 8);
        sumf += xd[bq8_offset + 2 * i] * static_cast<float>(dp4a_i8(q6k_pack_vi(vl, vh, i), u, 0) * sc);
    }
    return d * sumf;
}

inline float q6k_dot_q8_super(const unsigned char* blk, float d, const std::int8_t* xq,
                              const float* xd) {
    float acc = 0.0f;
    for (int t = 0; t < 32; ++t) {
        acc += q6k_dot_q8_iqs(blk, d, xq, xd, t);
    }
    return acc;
}

}  // namespace vesper
