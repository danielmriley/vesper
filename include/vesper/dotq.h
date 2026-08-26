#pragma once

#include "vesper/q8.h"

#include <cstdint>
#include <cstring>

#if defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_fp16.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define VESPER_HOT inline __attribute__((always_inline))
#else
#define VESPER_HOT inline
#endif

namespace vesper {

// 4x int8 dot. llama.cpp RDNA3/4 uses sudot4 -> v_dot4_i32_iu8.
// sdot4 is the CDNA/RDNA2 builtin and may not map on gfx1201.
VESPER_HOT int dp4a_i8(int a, int b, int c) {
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

VESPER_HOT int load_i32(const void* base, int n) {
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
VESPER_HOT void load_i32x2(const void* base, int n, int* a, int* b) {
#if defined(__HIP_DEVICE_COMPILE__)
    const int2 v = reinterpret_cast<const int2*>(static_cast<const int*>(base) + n)[0];
    *a = v.x;
    *b = v.y;
#else
    *a = load_i32(base, n);
    *b = load_i32(base, n + 1);
#endif
}

// 16 B of Q8_1 x. Q8 pair and full-block halves are four consecutive ints.
VESPER_HOT void load_i32x4(const void* base, int n, int* a, int* b, int* c, int* d) {
#if defined(__HIP_DEVICE_COMPILE__)
    const int4 v = reinterpret_cast<const int4*>(static_cast<const int*>(base) + n)[0];
    *a = v.x;
    *b = v.y;
    *c = v.z;
    *d = v.w;
#else
    load_i32x2(base, n, a, b);
    load_i32x2(base, n + 2, c, d);
#endif
}

VESPER_HOT void load_f32x2(const float* base, int n, float* a, float* b) {
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
VESPER_HOT int load_w32(const void* base, int n) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1201__) || defined(__GFX12__)) && \
    defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
    return __builtin_nontemporal_load(reinterpret_cast<const int*>(base) + n);
#else
    return load_i32(base, n);
#endif
}

// Consecutive 4-byte-aligned weight ints. Q4 pair slices sit on this.
VESPER_HOT void load_w32x2(const void* base, int n, int* a, int* b) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1201__) || defined(__GFX12__))
    struct alignas(8) W64 {
        int x;
        int y;
    };
    const auto* p = reinterpret_cast<const W64*>(static_cast<const int*>(base) + n);
#if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
    const W64 v = __builtin_nontemporal_load(p);
#else
    const W64 v = p[0];
#endif
    *a = v.x;
    *b = v.y;
#else
    *a = load_w32(base, n);
    *b = load_w32(base, n + 1);
#endif
}

// 16 B of 4-byte-aligned Q4 qs. Official Q4_K is 144 B, so a super is
// 16-byte aligned. n is a multiple of 4.
VESPER_HOT void load_w32x4(const void* base, int n, int* a, int* b, int* c, int* d) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1201__) || defined(__GFX12__))
    struct alignas(16) W128 {
        int x;
        int y;
        int z;
        int w;
    };
    const auto* p = reinterpret_cast<const W128*>(static_cast<const int*>(base) + n);
#if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
    const W128 v = __builtin_nontemporal_load(p);
#else
    const W128 v = p[0];
#endif
    *a = v.x;
    *b = v.y;
    *c = v.z;
    *d = v.w;
#else
    load_w32x2(base, n, a, b);
    load_w32x2(base, n + 2, c, d);
#endif
}

// 2-byte weight load. Q6 i8 scales use load_ws8 (cached). Do not use
// this for Q8_1 x.
VESPER_HOT std::uint16_t load_w16(const void* base, int n) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1201__) || defined(__GFX12__)) && \
    defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
    return __builtin_nontemporal_load(reinterpret_cast<const std::uint16_t*>(base) + n);
#else
    std::uint16_t value = 0;
    std::memcpy(&value, static_cast<const char*>(base) + static_cast<std::size_t>(n) * 2u, 2);
    return value;
#endif
}

// Q6 i8 scales. The 16-byte table is reused by every thread on a super.
// Odd 210-byte supers are only 2-byte aligned at blk+192, so this stays
// a u16 load. Cached: NT is for ql/qh. CPU is a signed byte.
VESPER_HOT int load_ws8(const void* base, int n) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned pair = reinterpret_cast<const std::uint16_t*>(base)[n >> 1];
    const unsigned byte = (n & 1) != 0 ? (pair >> 8) : (pair & 0xffu);
    return static_cast<int>(static_cast<signed char>(byte));
#else
    return static_cast<int>(static_cast<const signed char*>(base)[n]);
#endif
}

// llama.cpp get_scale_min_k4 for one MMVQ j in {0,1,2,3}. The 12-byte
// table is the same for every thread on a super-block. Three cached
// int loads; the old u16 extract used j-dependent addresses.
// Branchless: official Q4 pair waves always mix j, and llama.cpp still
// takes if (j < 2). shift = 16*(j&1) matches both halves.
VESPER_HOT void q4k_mmvq_sc_mn_words(int w0, int w1, int w2, int j, int* sc0, int* sc1, int* m0,
                                     int* m1) {
    const int shift = 16 * (j & 1);
    const int lo0 = (w0 >> shift) & 0xff;
    const int lo1 = (w0 >> (shift + 8)) & 0xff;
    const int mid0 = (w1 >> shift) & 0xff;
    const int mid1 = (w1 >> (shift + 8)) & 0xff;
    const int hi0 = (w2 >> shift) & 0xff;
    const int hi1 = (w2 >> (shift + 8)) & 0xff;
    const int sc0_lo = lo0 & 0x3f;
    const int sc1_lo = lo1 & 0x3f;
    const int m0_lo = mid0 & 0x3f;
    const int m1_lo = mid1 & 0x3f;
    const int sc0_hi = (hi0 & 0x0f) | ((lo0 >> 6) << 4);
    const int sc1_hi = (hi1 & 0x0f) | ((lo1 >> 6) << 4);
    const int m0_hi = (hi0 >> 4) | ((mid0 >> 6) << 4);
    const int m1_hi = (hi1 >> 4) | ((mid1 >> 6) << 4);
    const int mask = -(j >> 1);
    *sc0 = (sc0_lo & ~mask) | (sc0_hi & mask);
    *sc1 = (sc1_lo & ~mask) | (sc1_hi & mask);
    *m0 = (m0_lo & ~mask) | (m0_hi & mask);
    *m1 = (m1_lo & ~mask) | (m1_hi & mask);
}

// Cached: the 12-byte table is reused by every thread on a super.
// Nontemporal is for qs. Streaming this line evicts Q8_1 x for no
// bandwidth win. CPU load_i32 matches load_w32.
VESPER_HOT void q4k_mmvq_sc_mn(const void* scales, int j, int* sc0, int* sc1, int* m0, int* m1) {
    q4k_mmvq_sc_mn_words(load_i32(scales, 0), load_i32(scales, 1), load_i32(scales, 2), j, sc0, sc1,
                         m0, m1);
}

// Adjacent j from one cached 12-byte table. Official down now owns a
// full super, so it extracts all four j from the header words instead.
VESPER_HOT void q4k_mmvq_sc_mn2(const void* scales, int j, int* sc0a, int* sc1a, int* m0a, int* m1a,
                                int* sc0b, int* sc1b, int* m0b, int* m1b) {
    const int w0 = load_i32(scales, 0);
    const int w1 = load_i32(scales, 1);
    const int w2 = load_i32(scales, 2);
    q4k_mmvq_sc_mn_words(w0, w1, w2, j, sc0a, sc1a, m0a, m1a);
    q4k_mmvq_sc_mn_words(w0, w1, w2, j + 1, sc0b, sc1b, m0b, m1b);
}

// First 16 bytes of a Q4_K super: d, dmin, 12-byte scale table.
// Super is 144 B, so this is 16-byte aligned. One cached load.
// Nontemporal is for qs. Do not stream this line.
VESPER_HOT void q4k_load_header(const unsigned char* blk, float* d, float* dmin, int* w0, int* w1,
                                int* w2) {
    int h0 = 0;
    load_i32x4(blk, 0, &h0, w0, w1, w2);
#if defined(__HIP_DEVICE_COMPILE__)
    __half2 h;
    std::memcpy(&h, &h0, sizeof(h));
    *d = __low2float(h);
    *dmin = __high2float(h);
#else
    *d = f16_to_f32(static_cast<std::uint16_t>(static_cast<unsigned>(h0) & 0xffffu));
    *dmin = f16_to_f32(static_cast<std::uint16_t>(static_cast<unsigned>(h0) >> 16));
#endif
}

// One MMVQ slice of a Q4_K super-block against Q8_1 x. iqs is even in [0, 30].
// 16 slices cover the 256 weights. Matches llama.cpp vec_dot_q4_K_q8_1.
VESPER_HOT float q4k_dot_q8_iqs(const unsigned char* blk, float d, float dmin, const std::int8_t* xq,
                                const float* xd, int iqs) {
    const int bq8_offset = 2 * (iqs / 8);
    const unsigned char* qs = blk + 16;
    const int q4_index = (16 * bq8_offset + 4 * ((iqs / 2) % 4)) / 4;
    const int v0 = load_w32(qs, q4_index);
    const int v1 = load_w32(qs, q4_index + 4);

    int sc0 = 0;
    int sc1 = 0;
    int m0 = 0;
    int m1 = 0;
    q4k_mmvq_sc_mn(blk + 4, bq8_offset / 2, &sc0, &sc1, &m0, &m1);

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    const int sc[2] = {sc0, sc1};
    const int mn[2] = {m0, m1};
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

// Two consecutive even iqs that share bq8_offset (iqs in {0,4,...,28}).
// One 16 B header load, two aligned int2 qs loads, two aligned int2 x
// loads per QR half. Same sum as q4k_dot_q8_iqs(iqs)+q4k_dot_q8_iqs(iqs+2).
VESPER_HOT float q4k_dot_q8_pair(const unsigned char* blk, const std::int8_t* xq, const float* xd,
                                 int iqs) {
    float d = 0.0f;
    float dmin = 0.0f;
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    q4k_load_header(blk, &d, &dmin, &w0, &w1, &w2);
    const int bq8_offset = 2 * (iqs / 8);
    const unsigned char* qs = blk + 16;
    const int q4_index = (16 * bq8_offset + 4 * ((iqs / 2) % 4)) / 4;
    int v0a = 0;
    int v0b = 0;
    int v1a = 0;
    int v1b = 0;
    load_w32x2(qs, q4_index, &v0a, &v0b);
    load_w32x2(qs, q4_index + 4, &v1a, &v1b);

    int sc0 = 0;
    int sc1 = 0;
    int m0 = 0;
    int m1 = 0;
    q4k_mmvq_sc_mn_words(w0, w1, w2, bq8_offset / 2, &sc0, &sc1, &m0, &m1);

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    const int sc[2] = {sc0, sc1};
    const int mn[2] = {m0, m1};
    float d8_0 = 0.0f;
    float d8_1 = 0.0f;
    load_f32x2(xd, bq8_offset, &d8_0, &d8_1);
    const float d8s[2] = {d8_0, d8_1};
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (int i = 0; i < 2; ++i) {
        const int v0ai = (v0a >> (4 * i)) & 0x0f0f0f0f;
        const int v1ai = (v1a >> (4 * i)) & 0x0f0f0f0f;
        const int v0bi = (v0b >> (4 * i)) & 0x0f0f0f0f;
        const int v1bi = (v1b >> (4 * i)) & 0x0f0f0f0f;
        const int q8_base = ((bq8_offset + i) * 32 + 4 * ((iqs / 2) % 4)) / 4;
        int u0a = 0;
        int u0b = 0;
        int u1a = 0;
        int u1b = 0;
        load_i32x2(xq, q8_base, &u0a, &u0b);
        load_i32x2(xq, q8_base + 4, &u1a, &u1b);
        const int dot1a = dp4a_i8(v1ai, u1a, dp4a_i8(v0ai, u0a, 0));
        const int dot2a = dp4a_i8(0x01010101, u1a, dp4a_i8(0x01010101, u0a, 0));
        const int dot1b = dp4a_i8(v1bi, u1b, dp4a_i8(v0bi, u0b, 0));
        const int dot2b = dp4a_i8(0x01010101, u1b, dp4a_i8(0x01010101, u0b, 0));
        const float d8 = d8s[i];
        const int sci = static_cast<int>(sc[i]);
        const int mni = static_cast<int>(mn[i]);
        sumf_d += d8 * static_cast<float>(dot1a * sci);
        sumf_d += d8 * static_cast<float>(dot1b * sci);
        sumf_m += d8 * static_cast<float>(dot2a * mni);
        sumf_m += d8 * static_cast<float>(dot2b * mni);
    }
    return d * sumf_d - dmin * sumf_m;
}

// Four even iqs that share bq8_offset (iqs in {0,8,16,24}).
// Scales are already extracted. Two 16 B qs loads, two 16 B x loads
// per QR half.
VESPER_HOT float q4k_dot_q8_quad_sc(const unsigned char* blk, float d, float dmin,
                                    const std::int8_t* xq, const float* xd, int iqs, int sc0,
                                    int sc1, int m0, int m1) {
    const int bq8_offset = 2 * (iqs / 8);
    const unsigned char* qs = blk + 16;
    const int q4_index = (16 * bq8_offset + 4 * ((iqs / 2) % 4)) / 4;
    int v0a = 0;
    int v0b = 0;
    int v0c = 0;
    int v0d = 0;
    int v1a = 0;
    int v1b = 0;
    int v1c = 0;
    int v1d = 0;
    load_w32x4(qs, q4_index, &v0a, &v0b, &v0c, &v0d);
    load_w32x4(qs, q4_index + 4, &v1a, &v1b, &v1c, &v1d);

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    const int sc[2] = {sc0, sc1};
    const int mn[2] = {m0, m1};
    float d8_0 = 0.0f;
    float d8_1 = 0.0f;
    load_f32x2(xd, bq8_offset, &d8_0, &d8_1);
    const float d8s[2] = {d8_0, d8_1};
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (int i = 0; i < 2; ++i) {
        const int v0ai = (v0a >> (4 * i)) & 0x0f0f0f0f;
        const int v1ai = (v1a >> (4 * i)) & 0x0f0f0f0f;
        const int v0bi = (v0b >> (4 * i)) & 0x0f0f0f0f;
        const int v1bi = (v1b >> (4 * i)) & 0x0f0f0f0f;
        const int v0ci = (v0c >> (4 * i)) & 0x0f0f0f0f;
        const int v1ci = (v1c >> (4 * i)) & 0x0f0f0f0f;
        const int v0di = (v0d >> (4 * i)) & 0x0f0f0f0f;
        const int v1di = (v1d >> (4 * i)) & 0x0f0f0f0f;
        const int q8_base = ((bq8_offset + i) * 32 + 4 * ((iqs / 2) % 4)) / 4;
        int u0a = 0;
        int u0b = 0;
        int u0c = 0;
        int u0d = 0;
        int u1a = 0;
        int u1b = 0;
        int u1c = 0;
        int u1d = 0;
        load_i32x4(xq, q8_base, &u0a, &u0b, &u0c, &u0d);
        load_i32x4(xq, q8_base + 4, &u1a, &u1b, &u1c, &u1d);
        const float d8 = d8s[i];
        const int sci = static_cast<int>(sc[i]);
        const int mni = static_cast<int>(mn[i]);
        const int ones = 0x01010101;
        const int dot1a = dp4a_i8(v1ai, u1a, dp4a_i8(v0ai, u0a, 0));
        const int dot2a = dp4a_i8(ones, u1a, dp4a_i8(ones, u0a, 0));
        const int dot1b = dp4a_i8(v1bi, u1b, dp4a_i8(v0bi, u0b, 0));
        const int dot2b = dp4a_i8(ones, u1b, dp4a_i8(ones, u0b, 0));
        const int dot1c = dp4a_i8(v1ci, u1c, dp4a_i8(v0ci, u0c, 0));
        const int dot2c = dp4a_i8(ones, u1c, dp4a_i8(ones, u0c, 0));
        const int dot1d = dp4a_i8(v1di, u1d, dp4a_i8(v0di, u0d, 0));
        const int dot2d = dp4a_i8(ones, u1d, dp4a_i8(ones, u0d, 0));
        sumf_d += d8 * static_cast<float>(dot1a * sci);
        sumf_d += d8 * static_cast<float>(dot1b * sci);
        sumf_d += d8 * static_cast<float>(dot1c * sci);
        sumf_d += d8 * static_cast<float>(dot1d * sci);
        sumf_m += d8 * static_cast<float>(dot2a * mni);
        sumf_m += d8 * static_cast<float>(dot2b * mni);
        sumf_m += d8 * static_cast<float>(dot2c * mni);
        sumf_m += d8 * static_cast<float>(dot2d * mni);
    }
    return d * sumf_d - dmin * sumf_m;
}

VESPER_HOT float q4k_dot_q8_quad(const unsigned char* blk, const std::int8_t* xq, const float* xd,
                                 int iqs) {
    float d = 0.0f;
    float dmin = 0.0f;
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    q4k_load_header(blk, &d, &dmin, &w0, &w1, &w2);
    const int bq8_offset = 2 * (iqs / 8);
    int sc0 = 0;
    int sc1 = 0;
    int m0 = 0;
    int m1 = 0;
    q4k_mmvq_sc_mn_words(w0, w1, w2, bq8_offset / 2, &sc0, &sc1, &m0, &m1);
    return q4k_dot_q8_quad_sc(blk, d, dmin, xq, xd, iqs, sc0, sc1, m0, m1);
}

// Two quads that share the 12-byte scale table (iqs in {0,16}).
// Header is already loaded. Same sum as q4k_dot_q8_quad(iqs)+q4k_dot_q8_quad(iqs+8).
VESPER_HOT float q4k_dot_q8_oct_sc(const unsigned char* blk, float d, float dmin, int w0, int w1,
                                   int w2, const std::int8_t* xq, const float* xd, int iqs) {
    int sc0a = 0;
    int sc1a = 0;
    int m0a = 0;
    int m1a = 0;
    int sc0b = 0;
    int sc1b = 0;
    int m0b = 0;
    int m1b = 0;
    q4k_mmvq_sc_mn_words(w0, w1, w2, iqs / 8, &sc0a, &sc1a, &m0a, &m1a);
    q4k_mmvq_sc_mn_words(w0, w1, w2, iqs / 8 + 1, &sc0b, &sc1b, &m0b, &m1b);
    return q4k_dot_q8_quad_sc(blk, d, dmin, xq, xd, iqs, sc0a, sc1a, m0a, m1a) +
           q4k_dot_q8_quad_sc(blk, d, dmin, xq, xd, iqs + 8, sc0b, sc1b, m0b, m1b);
}

VESPER_HOT float q4k_dot_q8_oct(const unsigned char* blk, const std::int8_t* xq, const float* xd,
                                int iqs) {
    float d = 0.0f;
    float dmin = 0.0f;
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    q4k_load_header(blk, &d, &dmin, &w0, &w1, &w2);
    return q4k_dot_q8_oct_sc(blk, d, dmin, w0, w1, w2, xq, xd, iqs);
}

// Official FFN down: one thread, one super. One 16 B header load, then
// both octs. Calling oct twice would reload d/dmin/scales.
VESPER_HOT float q4k_dot_q8_super(const unsigned char* blk, const std::int8_t* xq, const float* xd) {
    float d = 0.0f;
    float dmin = 0.0f;
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    q4k_load_header(blk, &d, &dmin, &w0, &w1, &w2);
    return q4k_dot_q8_oct_sc(blk, d, dmin, w0, w1, w2, xq, xd, 0) +
           q4k_dot_q8_oct_sc(blk, d, dmin, w0, w1, w2, xq, xd, 16);
}

// Q8_0 qs starts at byte 2 of a 34-byte block: 2-byte aligned, not 4.
// llama.cpp get_int_b2. Two uint16 loads beat four byte loads on gfx1201.
VESPER_HOT int load_i32_b2(const void* base, int i32) {
    const auto* x16 = static_cast<const std::uint16_t*>(base);
    const unsigned lo = x16[2 * i32];
    const unsigned hi = x16[2 * i32 + 1];
    return static_cast<int>(lo | (hi << 16));
}

VESPER_HOT int load_w32_b2(const void* base, int i32) {
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

// Q8_0 VDR=2: two consecutive 2-byte-aligned qs ints (8 bytes).
VESPER_HOT void load_w32x2_b2(const void* base, int i32, int* a, int* b) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1201__) || defined(__GFX12__)) && \
    defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
    const auto* x16 = static_cast<const std::uint16_t*>(base);
    const unsigned lo0 = __builtin_nontemporal_load(x16 + 2 * i32);
    const unsigned hi0 = __builtin_nontemporal_load(x16 + 2 * i32 + 1);
    const unsigned lo1 = __builtin_nontemporal_load(x16 + 2 * i32 + 2);
    const unsigned hi1 = __builtin_nontemporal_load(x16 + 2 * i32 + 3);
    *a = static_cast<int>(lo0 | (hi0 << 16));
    *b = static_cast<int>(lo1 | (hi1 << 16));
#else
    *a = load_w32_b2(base, i32);
    *b = load_w32_b2(base, i32 + 1);
#endif
}

// VDR=2 slice. iqs is the starting int index in {0,2,4,6}.
VESPER_HOT float q8_dot_q8_iqs(const std::int8_t* qs, float d, const std::int8_t* xq, float xd,
                               int iqs) {
    int u0 = 0;
    int u1 = 0;
    int w0 = 0;
    int w1 = 0;
    load_i32x2(xq, iqs, &u0, &u1);
    load_w32x2_b2(qs, iqs, &w0, &w1);
    const int sumi = dp4a_i8(w1, u1, dp4a_i8(w0, u0, 0));
    return d * xd * static_cast<float>(sumi);
}

// Two VDR=2 slices (16 B qs). iqs is 0 or 4 on the 2-thread-per-block map.
VESPER_HOT float q8_dot_q8_pair(const std::int8_t* qs, float d, const std::int8_t* xq, float xd,
                                int iqs) {
    int u0 = 0;
    int u1 = 0;
    int u2 = 0;
    int u3 = 0;
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    int w3 = 0;
    load_i32x4(xq, iqs, &u0, &u1, &u2, &u3);
    load_w32x2_b2(qs, iqs, &w0, &w1);
    load_w32x2_b2(qs, iqs + 2, &w2, &w3);
    const int sumi = dp4a_i8(w3, u3, dp4a_i8(w2, u2, dp4a_i8(w1, u1, dp4a_i8(w0, u0, 0))));
    return d * xd * static_cast<float>(sumi);
}

// One Q8_0 block (32 i8) against one Q8_1 x block. llama.cpp vec_dot_q8_0_q8_1.
// Two 16 B x loads, integer acc, one scale. Matches four VDR=2 slices.
VESPER_HOT float q8_dot_q8(const std::int8_t* qs, float d, const std::int8_t* xq, float xd) {
    int u0 = 0;
    int u1 = 0;
    int u2 = 0;
    int u3 = 0;
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    int w3 = 0;
    load_i32x4(xq, 0, &u0, &u1, &u2, &u3);
    load_w32x2_b2(qs, 0, &w0, &w1);
    load_w32x2_b2(qs, 2, &w2, &w3);
    int sumi = dp4a_i8(w3, u3, dp4a_i8(w2, u2, dp4a_i8(w1, u1, dp4a_i8(w0, u0, 0))));
    load_i32x4(xq, 4, &u0, &u1, &u2, &u3);
    load_w32x2_b2(qs, 4, &w0, &w1);
    load_w32x2_b2(qs, 6, &w2, &w3);
    sumi = dp4a_i8(w3, u3, dp4a_i8(w2, u2, dp4a_i8(w1, u1, dp4a_i8(w0, u0, sumi))));
    return d * xd * static_cast<float>(sumi);
}

// Per-byte -32 without a 32-bit borrow. llama.cpp HIP uses __vsubss4
// (i8x4 + elementwise_sub_sat). Q6 bytes are 0..63, so sat never fires.
VESPER_HOT int q6k_sub32_bytes(int q) {
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
VESPER_HOT int q6k_pack_vi(int vl, int vh, int i) {
    const unsigned q = static_cast<unsigned>(((vl >> (4 * i)) & 0x0f0f0f0f) |
                                             (((vh >> (4 * i)) << 4) & 0x30303030));
    return q6k_sub32_bytes(static_cast<int>(q));
}

VESPER_HOT float q6k_dot_q8_iqs(const unsigned char* blk, float d, const std::int8_t* xq,
                                const float* xd, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_w32_b2(blk, iqs);
    const int vh = load_w32_b2(blk + 128, 8 * (iqs / 16) + (iqs % 8)) >> vh_shift;
    const void* scales = blk + 192;
    float sumf = 0.0f;
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (int i = 0; i < 2; ++i) {
        const int sc = load_ws8(scales, scale_offset + 4 * i);
        const int u = load_i32(xq + (bq8_offset + 2 * i) * 32, iqs % 8);
        sumf += xd[bq8_offset + 2 * i] * static_cast<float>(dp4a_i8(q6k_pack_vi(vl, vh, i), u, 0) * sc);
    }
    return d * sumf;
}

// Two consecutive iqs (iqs even in [0, 30]) share bq8_offset, scales, and
// vh_shift. vl/vh are consecutive 2-byte-aligned ints. Same sum as
// q6k_dot_q8_iqs(iqs)+q6k_dot_q8_iqs(iqs+1).
VESPER_HOT float q6k_dot_q8_pair(const unsigned char* blk, float d, const std::int8_t* xq,
                                 const float* xd, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    int vl0 = 0;
    int vl1 = 0;
    int vh0 = 0;
    int vh1 = 0;
    load_w32x2_b2(blk, iqs, &vl0, &vl1);
    load_w32x2_b2(blk + 128, 8 * (iqs / 16) + (iqs % 8), &vh0, &vh1);
    vh0 >>= vh_shift;
    vh1 >>= vh_shift;
    const void* scales = blk + 192;
    float sumf = 0.0f;
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (int i = 0; i < 2; ++i) {
        const int sc = load_ws8(scales, scale_offset + 4 * i);
        int u0 = 0;
        int u1 = 0;
        load_i32x2(xq + (bq8_offset + 2 * i) * 32, iqs % 8, &u0, &u1);
        const float d8 = xd[bq8_offset + 2 * i];
        sumf += d8 * static_cast<float>(dp4a_i8(q6k_pack_vi(vl0, vh0, i), u0, 0) * sc);
        sumf += d8 * static_cast<float>(dp4a_i8(q6k_pack_vi(vl1, vh1, i), u1, 0) * sc);
    }
    return d * sumf;
}

// Four consecutive iqs (iqs in {0,4,...,28}) share bq8_offset, scales,
// and vh_shift. Two pair loads. Same sum as
// q6k_dot_q8_pair(iqs)+q6k_dot_q8_pair(iqs+2).
VESPER_HOT float q6k_dot_q8_quad(const unsigned char* blk, float d, const std::int8_t* xq,
                                 const float* xd, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    int vl0 = 0;
    int vl1 = 0;
    int vl2 = 0;
    int vl3 = 0;
    int vh0 = 0;
    int vh1 = 0;
    int vh2 = 0;
    int vh3 = 0;
    load_w32x2_b2(blk, iqs, &vl0, &vl1);
    load_w32x2_b2(blk, iqs + 2, &vl2, &vl3);
    const int vh_index = 8 * (iqs / 16) + (iqs % 8);
    load_w32x2_b2(blk + 128, vh_index, &vh0, &vh1);
    load_w32x2_b2(blk + 128, vh_index + 2, &vh2, &vh3);
    vh0 >>= vh_shift;
    vh1 >>= vh_shift;
    vh2 >>= vh_shift;
    vh3 >>= vh_shift;
    const void* scales = blk + 192;
    float sumf = 0.0f;
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (int i = 0; i < 2; ++i) {
        const int sc = load_ws8(scales, scale_offset + 4 * i);
        int u0 = 0;
        int u1 = 0;
        int u2 = 0;
        int u3 = 0;
        load_i32x4(xq + (bq8_offset + 2 * i) * 32, iqs % 8, &u0, &u1, &u2, &u3);
        const float d8 = xd[bq8_offset + 2 * i];
        sumf += d8 * static_cast<float>(dp4a_i8(q6k_pack_vi(vl0, vh0, i), u0, 0) * sc);
        sumf += d8 * static_cast<float>(dp4a_i8(q6k_pack_vi(vl1, vh1, i), u1, 0) * sc);
        sumf += d8 * static_cast<float>(dp4a_i8(q6k_pack_vi(vl2, vh2, i), u2, 0) * sc);
        sumf += d8 * static_cast<float>(dp4a_i8(q6k_pack_vi(vl3, vh3, i), u3, 0) * sc);
    }
    return d * sumf;
}

// Eight consecutive iqs (iqs in {0,8,16,24}). Same sum as
// q6k_dot_q8_quad(iqs)+q6k_dot_q8_quad(iqs+4).
VESPER_HOT float q6k_dot_q8_oct(const unsigned char* blk, float d, const std::int8_t* xq,
                                const float* xd, int iqs) {
    return q6k_dot_q8_quad(blk, d, xq, xd, iqs) + q6k_dot_q8_quad(blk, d, xq, xd, iqs + 4);
}

VESPER_HOT float q6k_dot_q8_super(const unsigned char* blk, float d, const std::int8_t* xq,
                                  const float* xd) {
    float acc = 0.0f;
    for (int t = 0; t < 4; ++t) {
        acc += q6k_dot_q8_oct(blk, d, xq, xd, 8 * t);
    }
    return acc;
}

}  // namespace vesper

#undef VESPER_HOT
