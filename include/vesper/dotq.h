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

// HIP GEMV loads weights and Q8_1 x in the same function. restrict lets
// the compiler keep those streams independent. CPU tests use
// -Wpedantic -Werror, so the qualifier is empty on that compile.
#if defined(__HIP_DEVICE_COMPILE__)
#define VESPER_RESTRICT __restrict__
#else
#define VESPER_RESTRICT
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

// 32 B of Q8_1 x. Two int4 loads, no ALU between them. Cached: NT is
// for weights. Official Q4 oct and Q8 pair issue both x tiles before
// dp4a so L2 latency sits in the same window as the NT qs hoist.
VESPER_HOT void load_i32x8(const void* base, int n, int* a, int* b, int* c, int* d, int* e, int* f,
                           int* g, int* h) {
    load_i32x4(base, n, a, b, c, d);
    load_i32x4(base, n + 4, e, f, g, h);
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

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1201__) || defined(__GFX12__))
// vector_size so one NT load is one b64/b128. A struct of ints can be
// alignas(16), but LLVM may scalarize __builtin_nontemporal_load of that
// struct into four dword NT loads. Official Q4 qs is 16-byte aligned.
// Q8 qs / Q6 ql sit at align 2. Still load as HipW128: a vector_size
// type with aligned(2) lets LLVM split the b128 into dwords. GFX12
// splits an unaligned b128 in hardware (Johannes #22821). A struct of
// ints cannot be alignas(2): that is weaker than alignof(int).
using HipW64 = int __attribute__((vector_size(8), aligned(8)));
using HipW128 = int __attribute__((vector_size(16), aligned(16)));
static_assert(sizeof(HipW64) == 8, "HipW64 is 8 bytes");
static_assert(sizeof(HipW128) == 16, "HipW128 is 16 bytes");
#endif

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
    const auto* p = reinterpret_cast<const HipW64*>(static_cast<const int*>(base) + n);
#if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
    const HipW64 v = __builtin_nontemporal_load(p);
#else
    const HipW64 v = p[0];
#endif
    *a = v[0];
    *b = v[1];
#else
    *a = load_w32(base, n);
    *b = load_w32(base, n + 1);
#endif
}

// 16 B of 4-byte-aligned Q4 qs. Official Q4_K is 144 B, so a super is
// 16-byte aligned. n is a multiple of 4. Same vector_size NT as Q8/Q6.
VESPER_HOT void load_w32x4(const void* base, int n, int* a, int* b, int* c, int* d) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1201__) || defined(__GFX12__))
    const auto* p = reinterpret_cast<const HipW128*>(static_cast<const int*>(base) + n);
#if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
    const HipW128 v = __builtin_nontemporal_load(p);
#else
    const HipW128 v = p[0];
#endif
    *a = v[0];
    *b = v[1];
    *c = v[2];
    *d = v[3];
#else
    load_w32x2(base, n, a, b);
    load_w32x2(base, n + 2, c, d);
#endif
}

// 32 B of 4-byte-aligned Q4/Q8 qs. Two HipW128 NT loads, no ALU
// between them. NT does not leave the rest of a 64 B tile in cache,
// so a later 16 B load of the same tile is another memory round-trip.
VESPER_HOT void load_w32x8(const void* base, int n, int* a, int* b, int* c, int* d, int* e, int* f,
                           int* g, int* h) {
    load_w32x4(base, n, a, b, c, d);
    load_w32x4(base, n + 4, e, f, g, h);
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
// Odd 210-byte supers are only 2-byte aligned at blk+192, so a single
// scale stays a u16 load. Official oct loads the whole table with
// load_i32x4_b2. Cached: NT is for ql/qh. CPU is a signed byte.
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

// Adjacent j from one cached 12-byte table. Official down is two threads
// on one super (one oct each); each oct extracts two j pairs.
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
VESPER_HOT float q4k_dot_q8_iqs(const unsigned char* VESPER_RESTRICT blk, float d, float dmin,
                                const std::int8_t* VESPER_RESTRICT xq,
                                const float* VESPER_RESTRICT xd, int iqs) {
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
// hdr/qs are GGUF blk/blk+16 or the HIP SoA split of that super.
VESPER_HOT float q4k_dot_q8_pair_parts(const unsigned char* VESPER_RESTRICT hdr,
                                       const unsigned char* VESPER_RESTRICT qs,
                                       const std::int8_t* VESPER_RESTRICT xq,
                                       const float* VESPER_RESTRICT xd, int iqs) {
    float d = 0.0f;
    float dmin = 0.0f;
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    q4k_load_header(hdr, &d, &dmin, &w0, &w1, &w2);
    const int bq8_offset = 2 * (iqs / 8);
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
        // llama.cpp vmmq: integer acc, one scale. The products fit in i32
        // and in float's 24-bit mantissa, so this matches the split muls.
        sumf_d += d8 * static_cast<float>((dot1a + dot1b) * sci);
        sumf_m += d8 * static_cast<float>((dot2a + dot2b) * mni);
    }
    return d * sumf_d - dmin * sumf_m;
}

VESPER_HOT float q4k_dot_q8_pair(const unsigned char* VESPER_RESTRICT blk,
                                 const std::int8_t* VESPER_RESTRICT xq,
                                 const float* VESPER_RESTRICT xd, int iqs) {
    return q4k_dot_q8_pair_parts(blk, blk + 16, xq, xd, iqs);
}

// One quad's Q8_1 x int index. iqs in {0,8,16,24}. i in {0,1}.
VESPER_HOT int q4k_quad_x_base(int iqs, int i) {
    return ((2 * (iqs / 8) + i) * 32 + 4 * ((iqs / 2) % 4)) / 4;
}

// One quad's Q8_1 x (64 B). Two 32 B tiles into named ints. Do not
// stage through an address-taken array: that can spill the hoist.
VESPER_HOT void q4k_load_quad_x(const std::int8_t* VESPER_RESTRICT xq, int iqs, int* xa0, int* xa1,
                                int* xa2, int* xa3, int* xa4, int* xa5, int* xa6, int* xa7,
                                int* xb0, int* xb1, int* xb2, int* xb3, int* xb4, int* xb5, int* xb6,
                                int* xb7) {
    load_i32x8(xq, q4k_quad_x_base(iqs, 0), xa0, xa1, xa2, xa3, xa4, xa5, xa6, xa7);
    load_i32x8(xq, q4k_quad_x_base(iqs, 1), xb0, xb1, xb2, xb3, xb4, xb5, xb6, xb7);
}

// One QR half of a quad. qs, x, and d8 are already in registers.
VESPER_HOT void q4k_quad_acc(int v0a, int v0b, int v0c, int v0d, int v1a, int v1b, int v1c, int v1d,
                             int shift, int u0a, int u0b, int u0c, int u0d, int u1a, int u1b, int u1c,
                             int u1d, float d8, int sci, int mni, float& sumf_d, float& sumf_m) {
    const int v0ai = (v0a >> shift) & 0x0f0f0f0f;
    const int v1ai = (v1a >> shift) & 0x0f0f0f0f;
    const int v0bi = (v0b >> shift) & 0x0f0f0f0f;
    const int v1bi = (v1b >> shift) & 0x0f0f0f0f;
    const int v0ci = (v0c >> shift) & 0x0f0f0f0f;
    const int v1ci = (v1c >> shift) & 0x0f0f0f0f;
    const int v0di = (v0d >> shift) & 0x0f0f0f0f;
    const int v1di = (v1d >> shift) & 0x0f0f0f0f;
    const int ones = 0x01010101;
    const int dot1a = dp4a_i8(v1ai, u1a, dp4a_i8(v0ai, u0a, 0));
    const int dot2a = dp4a_i8(ones, u1a, dp4a_i8(ones, u0a, 0));
    const int dot1b = dp4a_i8(v1bi, u1b, dp4a_i8(v0bi, u0b, 0));
    const int dot2b = dp4a_i8(ones, u1b, dp4a_i8(ones, u0b, 0));
    const int dot1c = dp4a_i8(v1ci, u1c, dp4a_i8(v0ci, u0c, 0));
    const int dot2c = dp4a_i8(ones, u1c, dp4a_i8(ones, u0c, 0));
    const int dot1d = dp4a_i8(v1di, u1d, dp4a_i8(v0di, u0d, 0));
    const int dot2d = dp4a_i8(ones, u1d, dp4a_i8(ones, u0d, 0));
    sumf_d += d8 * static_cast<float>((dot1a + dot1b + dot1c + dot1d) * sci);
    sumf_m += d8 * static_cast<float>((dot2a + dot2b + dot2c + dot2d) * mni);
}

// Four even iqs that share bq8_offset (iqs in {0,8,16,24}).
// qs, x, and both d8 scales are already in registers. Same sum as
// the old load-inside quad.
VESPER_HOT float q4k_dot_q8_quad_sc_vu(int v0a, int v0b, int v0c, int v0d, int v1a, int v1b, int v1c,
                                       int v1d, float d, float dmin, int sc0, int sc1, int m0, int m1,
                                       float d8_0, float d8_1, int xa0, int xa1, int xa2, int xa3,
                                       int xa4, int xa5, int xa6, int xa7, int xb0, int xb1, int xb2,
                                       int xb3, int xb4, int xb5, int xb6, int xb7) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    q4k_quad_acc(v0a, v0b, v0c, v0d, v1a, v1b, v1c, v1d, 0, xa0, xa1, xa2, xa3, xa4, xa5, xa6, xa7,
                 d8_0, sc0, m0, sumf_d, sumf_m);
    q4k_quad_acc(v0a, v0b, v0c, v0d, v1a, v1b, v1c, v1d, 4, xb0, xb1, xb2, xb3, xb4, xb5, xb6, xb7,
                 d8_1, sc1, m1, sumf_d, sumf_m);
    return d * sumf_d - dmin * sumf_m;
}

VESPER_HOT float q4k_dot_q8_quad_sc_v(int v0a, int v0b, int v0c, int v0d, int v1a, int v1b, int v1c,
                                      int v1d, float d, float dmin, const std::int8_t* VESPER_RESTRICT xq,
                                      const float* VESPER_RESTRICT xd, int iqs, int sc0, int sc1,
                                      int m0, int m1) {
    int xa0 = 0;
    int xa1 = 0;
    int xa2 = 0;
    int xa3 = 0;
    int xa4 = 0;
    int xa5 = 0;
    int xa6 = 0;
    int xa7 = 0;
    int xb0 = 0;
    int xb1 = 0;
    int xb2 = 0;
    int xb3 = 0;
    int xb4 = 0;
    int xb5 = 0;
    int xb6 = 0;
    int xb7 = 0;
    q4k_load_quad_x(xq, iqs, &xa0, &xa1, &xa2, &xa3, &xa4, &xa5, &xa6, &xa7, &xb0, &xb1, &xb2, &xb3,
                    &xb4, &xb5, &xb6, &xb7);
    float d8_0 = 0.0f;
    float d8_1 = 0.0f;
    load_f32x2(xd, 2 * (iqs / 8), &d8_0, &d8_1);
    return q4k_dot_q8_quad_sc_vu(v0a, v0b, v0c, v0d, v1a, v1b, v1c, v1d, d, dmin, sc0, sc1, m0, m1,
                                 d8_0, d8_1, xa0, xa1, xa2, xa3, xa4, xa5, xa6, xa7, xb0, xb1, xb2,
                                 xb3, xb4, xb5, xb6, xb7);
}

VESPER_HOT float q4k_dot_q8_quad_sc(const unsigned char* VESPER_RESTRICT qs, float d, float dmin,
                                    const std::int8_t* VESPER_RESTRICT xq,
                                    const float* VESPER_RESTRICT xd, int iqs, int sc0, int sc1,
                                    int m0, int m1) {
    const int bq8_offset = 2 * (iqs / 8);
    const int q4_index = (16 * bq8_offset + 4 * ((iqs / 2) % 4)) / 4;
    int v0a = 0;
    int v0b = 0;
    int v0c = 0;
    int v0d = 0;
    int v1a = 0;
    int v1b = 0;
    int v1c = 0;
    int v1d = 0;
    load_w32x8(qs, q4_index, &v0a, &v0b, &v0c, &v0d, &v1a, &v1b, &v1c, &v1d);
    return q4k_dot_q8_quad_sc_v(v0a, v0b, v0c, v0d, v1a, v1b, v1c, v1d, d, dmin, xq, xd, iqs, sc0,
                                sc1, m0, m1);
}

VESPER_HOT float q4k_dot_q8_quad(const unsigned char* VESPER_RESTRICT blk,
                                 const std::int8_t* VESPER_RESTRICT xq,
                                 const float* VESPER_RESTRICT xd, int iqs) {
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
    return q4k_dot_q8_quad_sc(blk + 16, d, dmin, xq, xd, iqs, sc0, sc1, m0, m1);
}

// Two quads that share the 12-byte scale table (iqs in {0,16}).
// Header is already loaded. Same sum as q4k_dot_q8_quad(iqs)+q4k_dot_q8_quad(iqs+8).
// NT does not cache the second 32 B of a 64 B half, so both 32 B qs
// tiles issue before any dp4a. Both 64 B x tiles and both 8 B d8
// tiles issue in the same window. vu is then pure ALU. q4_base is 0
// on HIP half-remap (iqs_r=0) and 16 on a GGUF 128 B super at iqs=16.
// launch_bounds 160 is the VGPR budget.
VESPER_HOT float q4k_dot_q8_oct_sc_qs(const unsigned char* VESPER_RESTRICT qs, float d, float dmin,
                                      int w0, int w1, int w2, const std::int8_t* VESPER_RESTRICT xq,
                                      const float* VESPER_RESTRICT xd, int iqs) {
    const int q4_base = 16 * (iqs / 16);
    int a0 = 0;
    int a1 = 0;
    int a2 = 0;
    int a3 = 0;
    int b0 = 0;
    int b1 = 0;
    int b2 = 0;
    int b3 = 0;
    int c0 = 0;
    int c1 = 0;
    int c2 = 0;
    int c3 = 0;
    int d0 = 0;
    int d1 = 0;
    int d2 = 0;
    int d3 = 0;
    int ua0 = 0;
    int ua1 = 0;
    int ua2 = 0;
    int ua3 = 0;
    int ua4 = 0;
    int ua5 = 0;
    int ua6 = 0;
    int ua7 = 0;
    int ua8 = 0;
    int ua9 = 0;
    int ua10 = 0;
    int ua11 = 0;
    int ua12 = 0;
    int ua13 = 0;
    int ua14 = 0;
    int ua15 = 0;
    int ub0 = 0;
    int ub1 = 0;
    int ub2 = 0;
    int ub3 = 0;
    int ub4 = 0;
    int ub5 = 0;
    int ub6 = 0;
    int ub7 = 0;
    int ub8 = 0;
    int ub9 = 0;
    int ub10 = 0;
    int ub11 = 0;
    int ub12 = 0;
    int ub13 = 0;
    int ub14 = 0;
    int ub15 = 0;
    load_w32x8(qs, q4_base, &a0, &a1, &a2, &a3, &b0, &b1, &b2, &b3);
    load_w32x8(qs, q4_base + 8, &c0, &c1, &c2, &c3, &d0, &d1, &d2, &d3);
    q4k_load_quad_x(xq, iqs, &ua0, &ua1, &ua2, &ua3, &ua4, &ua5, &ua6, &ua7, &ua8, &ua9, &ua10,
                    &ua11, &ua12, &ua13, &ua14, &ua15);
    q4k_load_quad_x(xq, iqs + 8, &ub0, &ub1, &ub2, &ub3, &ub4, &ub5, &ub6, &ub7, &ub8, &ub9, &ub10,
                    &ub11, &ub12, &ub13, &ub14, &ub15);
    int sc0a = 0;
    int sc1a = 0;
    int m0a = 0;
    int m1a = 0;
    int sc0b = 0;
    int sc1b = 0;
    int m0b = 0;
    int m1b = 0;
    float d8a0 = 0.0f;
    float d8a1 = 0.0f;
    float d8b0 = 0.0f;
    float d8b1 = 0.0f;
    load_f32x2(xd, 2 * (iqs / 8), &d8a0, &d8a1);
    load_f32x2(xd, 2 * ((iqs + 8) / 8), &d8b0, &d8b1);
    q4k_mmvq_sc_mn_words(w0, w1, w2, iqs / 8, &sc0a, &sc1a, &m0a, &m1a);
    q4k_mmvq_sc_mn_words(w0, w1, w2, iqs / 8 + 1, &sc0b, &sc1b, &m0b, &m1b);
    const float s0 = q4k_dot_q8_quad_sc_vu(a0, a1, a2, a3, b0, b1, b2, b3, d, dmin, sc0a, sc1a, m0a,
                                           m1a, d8a0, d8a1, ua0, ua1, ua2, ua3, ua4, ua5, ua6, ua7,
                                           ua8, ua9, ua10, ua11, ua12, ua13, ua14, ua15);
    const float s1 = q4k_dot_q8_quad_sc_vu(c0, c1, c2, c3, d0, d1, d2, d3, d, dmin, sc0b, sc1b, m0b,
                                           m1b, d8b0, d8b1, ub0, ub1, ub2, ub3, ub4, ub5, ub6, ub7,
                                           ub8, ub9, ub10, ub11, ub12, ub13, ub14, ub15);
    return s0 + s1;
}

VESPER_HOT float q4k_dot_q8_oct_sc(const unsigned char* VESPER_RESTRICT blk, float d, float dmin,
                                   int w0, int w1, int w2, const std::int8_t* VESPER_RESTRICT xq,
                                   const float* VESPER_RESTRICT xd, int iqs) {
    return q4k_dot_q8_oct_sc_qs(blk + 16, d, dmin, w0, w1, w2, xq, xd, iqs);
}

VESPER_HOT float q4k_dot_q8_oct(const unsigned char* VESPER_RESTRICT blk,
                                const std::int8_t* VESPER_RESTRICT xq,
                                const float* VESPER_RESTRICT xd, int iqs) {
    float d = 0.0f;
    float dmin = 0.0f;
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    q4k_load_header(blk, &d, &dmin, &w0, &w1, &w2);
    return q4k_dot_q8_oct_sc_qs(blk + 16, d, dmin, w0, w1, w2, xq, xd, iqs);
}

// HIP Q4 SoA: super-major 16 B headers, then half-major 64 B qs.
// Official oct (down) and quad (SwiGLU) each own one 64 B half, so
// neighboring WGs hit consecutive 64 B instead of a 128 B super stride.
VESPER_HOT const unsigned char* q4k_soa_hdr(const unsigned char* packed, int rows, int supers,
                                            int row, int s) {
    (void)supers;
    return packed + (static_cast<std::size_t>(s) * static_cast<std::size_t>(rows) +
                     static_cast<std::size_t>(row)) *
                        16u;
}

VESPER_HOT const unsigned char* q4k_soa_qs(const unsigned char* packed, int rows, int supers,
                                           int row, int s, int half) {
    return packed + static_cast<std::size_t>(rows) * static_cast<std::size_t>(supers) * 16u +
           ((static_cast<std::size_t>(s) * 2u + static_cast<std::size_t>(half)) *
                static_cast<std::size_t>(rows) +
            static_cast<std::size_t>(row)) *
               64u;
}

VESPER_HOT float q4k_dot_q8_quad_parts(const unsigned char* VESPER_RESTRICT hdr,
                                       const unsigned char* VESPER_RESTRICT qs,
                                       const std::int8_t* VESPER_RESTRICT xq,
                                       const float* VESPER_RESTRICT xd, int iqs) {
    float d = 0.0f;
    float dmin = 0.0f;
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    q4k_load_header(hdr, &d, &dmin, &w0, &w1, &w2);
    const int bq8_offset = 2 * (iqs / 8);
    int sc0 = 0;
    int sc1 = 0;
    int m0 = 0;
    int m1 = 0;
    q4k_mmvq_sc_mn_words(w0, w1, w2, bq8_offset / 2, &sc0, &sc1, &m0, &m1);
    return q4k_dot_q8_quad_sc(qs, d, dmin, xq, xd, iqs, sc0, sc1, m0, m1);
}

VESPER_HOT float q4k_dot_q8_oct_parts(const unsigned char* VESPER_RESTRICT hdr,
                                      const unsigned char* VESPER_RESTRICT qs,
                                      const std::int8_t* VESPER_RESTRICT xq,
                                      const float* VESPER_RESTRICT xd, int iqs) {
    float d = 0.0f;
    float dmin = 0.0f;
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    q4k_load_header(hdr, &d, &dmin, &w0, &w1, &w2);
    return q4k_dot_q8_oct_sc_qs(qs, d, dmin, w0, w1, w2, xq, xd, iqs);
}

VESPER_HOT float q4k_dot_q8_super_parts(const unsigned char* VESPER_RESTRICT hdr,
                                        const unsigned char* VESPER_RESTRICT qs,
                                        const std::int8_t* VESPER_RESTRICT xq,
                                        const float* VESPER_RESTRICT xd) {
    float d = 0.0f;
    float dmin = 0.0f;
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    q4k_load_header(hdr, &d, &dmin, &w0, &w1, &w2);
    return q4k_dot_q8_oct_sc_qs(qs, d, dmin, w0, w1, w2, xq, xd, 0) +
           q4k_dot_q8_oct_sc_qs(qs, d, dmin, w0, w1, w2, xq, xd, 16);
}

// HIP SoA qs is two 64 B halves. iqs=16 addressing assumes a 128 B super,
// so the second oct is remapped to iqs=0 on half 1.
VESPER_HOT float q4k_dot_q8_super_halves(const unsigned char* VESPER_RESTRICT hdr,
                                         const unsigned char* VESPER_RESTRICT qs0,
                                         const unsigned char* VESPER_RESTRICT qs1,
                                         const std::int8_t* VESPER_RESTRICT xq,
                                         const float* VESPER_RESTRICT xd) {
    float d = 0.0f;
    float dmin = 0.0f;
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    q4k_load_header(hdr, &d, &dmin, &w0, &w1, &w2);
    return q4k_dot_q8_oct_sc_qs(qs0, d, dmin, w0, w1, w2, xq, xd, 0) +
           q4k_dot_q8_oct_sc_qs(qs1, d, dmin, w0, w1, w2, xq + 128, xd + 4, 0);
}

// Wide Q4 (129+ supers): one thread, one super. One 16 B header load,
// then both octs. Official down (68 supers) uses q4k_dot_q8_oct instead.
// Calling oct twice would reload d/dmin/scales.
VESPER_HOT float q4k_dot_q8_super(const unsigned char* VESPER_RESTRICT blk,
                                  const std::int8_t* VESPER_RESTRICT xq,
                                  const float* VESPER_RESTRICT xd) {
    return q4k_dot_q8_super_parts(blk, blk + 16, xq, xd);
}

// GGUF Q8_0 qs starts at byte 2 of a 34-byte block (2-byte aligned).
// HIP upload repacks to SoA so qs is 16-byte aligned and uses load_w32x4.
// Q6_K HIP upload is also SoA (aligned ql/qh). GGUF 210-byte supers stay
// on disk and on the CPU mmap path.
// llama.cpp get_int_b2. gfx1201 issues one 8-byte or 16-byte NT load as
// HipW64/HipW128 (Johannes #22821). The hardware splits if the address
// is not naturally aligned. A vector_size type with aligned(2) can make
// LLVM emit dwords instead. Four scalar u16 NT loads for one Q8 half
// were the old path.
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

// Q8_0 VDR=2 / Q6 pair: two consecutive 2-byte-aligned qs ints (8 bytes).
VESPER_HOT void load_w32x2_b2(const void* base, int i32, int* a, int* b) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1201__) || defined(__GFX12__)) && \
    defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
    const auto* p =
        reinterpret_cast<const HipW64*>(static_cast<const std::uint16_t*>(base) + 2 * i32);
    const HipW64 v = __builtin_nontemporal_load(p);
    *a = v[0];
    *b = v[1];
#else
    *a = load_w32_b2(base, i32);
    *b = load_w32_b2(base, i32 + 1);
#endif
}

// Official Q8 block half and Q6 quad: 16 B of qs at a 2-byte address.
// Load as HipW128 so LLVM emits one b128. GFX12 splits if needed.
VESPER_HOT void load_w32x4_b2(const void* base, int i32, int* a, int* b, int* c, int* d) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1201__) || defined(__GFX12__)) && \
    defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
    const auto* p =
        reinterpret_cast<const HipW128*>(static_cast<const std::uint16_t*>(base) + 2 * i32);
    const HipW128 v = __builtin_nontemporal_load(p);
    *a = v[0];
    *b = v[1];
    *c = v[2];
    *d = v[3];
#else
    load_w32x2_b2(base, i32, a, b);
    load_w32x2_b2(base, i32 + 2, c, d);
#endif
}

// Cached 16 B at a 2-byte address. Q6 i8 scales sit at blk+192 on a
// 210-byte super, so a 4-byte-aligned int4 load is illegal as C++.
// Official lm_head / o_proj reuse this table for both quads of an oct.
// Do not stream it: four threads on a super reread the same 16 bytes,
// and NT would evict Q8_1 x for that. Same HipW128 as the NT qs path
// so LLVM does not split on an aligned(2) vector type.
VESPER_HOT void load_i32x4_b2(const void* base, int i32, int* a, int* b, int* c, int* d) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1201__) || defined(__GFX12__))
    const auto* p =
        reinterpret_cast<const HipW128*>(static_cast<const std::uint16_t*>(base) + 2 * i32);
    const HipW128 v = p[0];
    *a = v[0];
    *b = v[1];
    *c = v[2];
    *d = v[3];
#else
    *a = load_i32_b2(base, i32);
    *b = load_i32_b2(base, i32 + 1);
    *c = load_i32_b2(base, i32 + 2);
    *d = load_i32_b2(base, i32 + 3);
#endif
}

// One signed scale from a 16-byte Q6 table already in four ints.
// n is 0..15. Matches load_ws8 on that table.
VESPER_HOT int q6k_scale_byte(int w0, int w1, int w2, int w3, int n) {
    const int w = (n & 4) != 0 ? ((n & 8) != 0 ? w3 : w1) : ((n & 8) != 0 ? w2 : w0);
    return static_cast<int>(
        static_cast<signed char>((static_cast<unsigned>(w) >> (8 * (n & 3))) & 0xffu));
}

VESPER_HOT void q6k_load_scales(const unsigned char* blk, int* w0, int* w1, int* w2, int* w3) {
    load_i32x4_b2(blk + 192, 0, w0, w1, w2, w3);
}

VESPER_HOT void q6k_load_scales_a(const unsigned char* scales, int* w0, int* w1, int* w2, int* w3) {
    load_i32x4(scales, 0, w0, w1, w2, w3);
}

// VDR=2 slice. iqs is the starting int index in {0,2,4,6}.
VESPER_HOT float q8_dot_q8_iqs(const std::int8_t* VESPER_RESTRICT qs, float d,
                               const std::int8_t* VESPER_RESTRICT xq, float xd, int iqs) {
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
VESPER_HOT float q8_dot_q8_pair(const std::int8_t* VESPER_RESTRICT qs, float d,
                                const std::int8_t* VESPER_RESTRICT xq, float xd, int iqs) {
    int u0 = 0;
    int u1 = 0;
    int u2 = 0;
    int u3 = 0;
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    int w3 = 0;
    load_i32x4(xq, iqs, &u0, &u1, &u2, &u3);
    load_w32x4_b2(qs, iqs, &w0, &w1, &w2, &w3);
    // Two VDR=2 dots. Independent acc so the second pair does not wait
    // on the first chain's sumi.
    const int sum0 = dp4a_i8(w1, u1, dp4a_i8(w0, u0, 0));
    const int sum1 = dp4a_i8(w3, u3, dp4a_i8(w2, u2, 0));
    return d * xd * static_cast<float>(sum0 + sum1);
}

// One Q8_0 block against one Q8_1 x block. qs and x already in
// registers. Two independent half-dots, then one scale.
// llama.cpp vec_dot_q8_0_q8_1.
VESPER_HOT float q8_dot_q8_wu(int w0, int w1, int w2, int w3, int w4, int w5, int w6, int w7, int u0,
                              int u1, int u2, int u3, int u4, int u5, int u6, int u7, float d,
                              float xd) {
    const int sum0 = dp4a_i8(w3, u3, dp4a_i8(w2, u2, dp4a_i8(w1, u1, dp4a_i8(w0, u0, 0))));
    const int sum1 = dp4a_i8(w7, u7, dp4a_i8(w6, u6, dp4a_i8(w5, u5, dp4a_i8(w4, u4, 0))));
    return d * xd * static_cast<float>(sum0 + sum1);
}

VESPER_HOT float q8_dot_q8_w(int w0, int w1, int w2, int w3, int w4, int w5, int w6, int w7,
                             const std::int8_t* VESPER_RESTRICT xq, float d, float xd) {
    int u0 = 0;
    int u1 = 0;
    int u2 = 0;
    int u3 = 0;
    int u4 = 0;
    int u5 = 0;
    int u6 = 0;
    int u7 = 0;
    load_i32x8(xq, 0, &u0, &u1, &u2, &u3, &u4, &u5, &u6, &u7);
    return q8_dot_q8_wu(w0, w1, w2, w3, w4, w5, w6, w7, u0, u1, u2, u3, u4, u5, u6, u7, d, xd);
}

// One Q8_0 block (32 i8) against one Q8_1 x block.
// Both 16 B qs halves issue before any dp4a. AlignedQs is HIP SoA.
// GGUF keeps the 2-byte load.
template<bool AlignedQs>
VESPER_HOT float q8_dot_q8_t(const std::int8_t* VESPER_RESTRICT qs, float d,
                             const std::int8_t* VESPER_RESTRICT xq, float xd) {
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    int w3 = 0;
    int w4 = 0;
    int w5 = 0;
    int w6 = 0;
    int w7 = 0;
    if constexpr (AlignedQs) {
        load_w32x8(qs, 0, &w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);
    } else {
        load_w32x4_b2(qs, 0, &w0, &w1, &w2, &w3);
        load_w32x4_b2(qs, 4, &w4, &w5, &w6, &w7);
    }
    return q8_dot_q8_w(w0, w1, w2, w3, w4, w5, w6, w7, xq, d, xd);
}

// Two consecutive Q8_0 blocks that share a 64 B SoA pair. NT both 32 B
// qs tiles and both 32 B x tiles before ALU. x stays cached. Official
// attn/GDN Tight path. launch_bounds 96 is the VGPR budget.
template<bool AlignedQs>
VESPER_HOT float q8_dot_q8_2_t(const std::int8_t* VESPER_RESTRICT qs0, float d0,
                               const std::int8_t* VESPER_RESTRICT xq0, float xd0,
                               const std::int8_t* VESPER_RESTRICT qs1, float d1,
                               const std::int8_t* VESPER_RESTRICT xq1, float xd1) {
    int w0 = 0;
    int w1 = 0;
    int w2 = 0;
    int w3 = 0;
    int w4 = 0;
    int w5 = 0;
    int w6 = 0;
    int w7 = 0;
    int w8 = 0;
    int w9 = 0;
    int w10 = 0;
    int w11 = 0;
    int w12 = 0;
    int w13 = 0;
    int w14 = 0;
    int w15 = 0;
    int u0 = 0;
    int u1 = 0;
    int u2 = 0;
    int u3 = 0;
    int u4 = 0;
    int u5 = 0;
    int u6 = 0;
    int u7 = 0;
    int u8 = 0;
    int u9 = 0;
    int u10 = 0;
    int u11 = 0;
    int u12 = 0;
    int u13 = 0;
    int u14 = 0;
    int u15 = 0;
    if constexpr (AlignedQs) {
        load_w32x8(qs0, 0, &w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);
        load_w32x8(qs1, 0, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15);
    } else {
        load_w32x4_b2(qs0, 0, &w0, &w1, &w2, &w3);
        load_w32x4_b2(qs0, 4, &w4, &w5, &w6, &w7);
        load_w32x4_b2(qs1, 0, &w8, &w9, &w10, &w11);
        load_w32x4_b2(qs1, 4, &w12, &w13, &w14, &w15);
    }
    load_i32x8(xq0, 0, &u0, &u1, &u2, &u3, &u4, &u5, &u6, &u7);
    load_i32x8(xq1, 0, &u8, &u9, &u10, &u11, &u12, &u13, &u14, &u15);
    const float a0 =
        q8_dot_q8_wu(w0, w1, w2, w3, w4, w5, w6, w7, u0, u1, u2, u3, u4, u5, u6, u7, d0, xd0);
    const float a1 =
        q8_dot_q8_wu(w8, w9, w10, w11, w12, w13, w14, w15, u8, u9, u10, u11, u12, u13, u14, u15, d1,
                     xd1);
    return a0 + a1;
}

VESPER_HOT float q8_dot_q8(const std::int8_t* VESPER_RESTRICT qs, float d,
                           const std::int8_t* VESPER_RESTRICT xq, float xd) {
    return q8_dot_q8_t<false>(qs, d, xq, xd);
}

VESPER_HOT int q8_soa_scale_bytes_n(int nblocks) {
    return ((nblocks + 1) / 2) * 4;
}

// HIP Q8 SoA: pair-major 4 B scales (cached), then pair-major 64 B qs.
// OneTrip threads do two blocks; neighboring WGs hit consecutive 4 B
// scales and 64 B qs instead of a 320 B scale-row stride.
VESPER_HOT const unsigned char* q8_soa_scale(const unsigned char* packed, int rows, int nblocks, int row,
                                             int b) {
    (void)nblocks;
    const int pair = b / 2;
    const int half = b & 1;
    return packed +
           (static_cast<std::size_t>(pair) * static_cast<std::size_t>(rows) +
            static_cast<std::size_t>(row)) *
               4u +
           static_cast<std::size_t>(half) * 2u;
}

VESPER_HOT const unsigned char* q8_soa_qs(const unsigned char* packed, int rows, int nblocks, int row,
                                          int b) {
    const int pair = b / 2;
    const int half = b & 1;
    return packed + static_cast<std::size_t>(rows) * static_cast<std::size_t>(q8_soa_scale_bytes_n(nblocks)) +
           (static_cast<std::size_t>(pair) * static_cast<std::size_t>(rows) +
            static_cast<std::size_t>(row)) *
               64u +
           static_cast<std::size_t>(half) * 32u;
}

VESPER_HOT std::size_t q8_soa_row_bytes_n(int nblocks) {
    return static_cast<std::size_t>(q8_soa_scale_bytes_n(nblocks)) +
           static_cast<std::size_t>(nblocks) * 32u;
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

VESPER_HOT float q6k_dot_q8_iqs(const unsigned char* VESPER_RESTRICT blk, float d,
                                const std::int8_t* VESPER_RESTRICT xq,
                                const float* VESPER_RESTRICT xd, int iqs) {
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
VESPER_HOT float q6k_dot_q8_pair(const unsigned char* VESPER_RESTRICT blk, float d,
                                 const std::int8_t* VESPER_RESTRICT xq,
                                 const float* VESPER_RESTRICT xd, int iqs) {
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
        const int acc = dp4a_i8(q6k_pack_vi(vl0, vh0, i), u0, 0) +
                        dp4a_i8(q6k_pack_vi(vl1, vh1, i), u1, 0);
        sumf += d8 * static_cast<float>(acc * sc);
    }
    return d * sumf;
}

// Four consecutive iqs (iqs in {0,4,...,28}) share bq8_offset, scales,
// and vh_shift. Two pair loads. Same sum as
// q6k_dot_q8_pair(iqs)+q6k_dot_q8_pair(iqs+2). Scales are already loaded.
// Aligned is the HIP SoA path (16-byte ql/qh). GGUF keeps the 2-byte load.
// HalfQl is the HIP 64 B ql tile: iqs>=16 is remapped to iqs-16 on half 1.
// qh / x / scales still use the full-super iqs.
template<bool Aligned, bool HalfQl = false>
VESPER_HOT float q6k_dot_q8_quad_sc_t(const unsigned char* VESPER_RESTRICT ql,
                                      const unsigned char* VESPER_RESTRICT qh, float d,
                                      const std::int8_t* VESPER_RESTRICT xq,
                                      const float* VESPER_RESTRICT xd, int iqs, int sw0, int sw1,
                                      int sw2, int sw3) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int ql_iqs = HalfQl ? (iqs & 15) : iqs;
    int vl0 = 0;
    int vl1 = 0;
    int vl2 = 0;
    int vl3 = 0;
    int vh0 = 0;
    int vh1 = 0;
    int vh2 = 0;
    int vh3 = 0;
    if constexpr (Aligned) {
        load_w32x4(ql, ql_iqs, &vl0, &vl1, &vl2, &vl3);
        const int vh_index = 8 * (iqs / 16) + (iqs % 8);
        load_w32x4(qh, vh_index, &vh0, &vh1, &vh2, &vh3);
    } else {
        load_w32x4_b2(ql, iqs, &vl0, &vl1, &vl2, &vl3);
        const int vh_index = 8 * (iqs / 16) + (iqs % 8);
        load_w32x4_b2(qh, vh_index, &vh0, &vh1, &vh2, &vh3);
    }
    vh0 >>= vh_shift;
    vh1 >>= vh_shift;
    vh2 >>= vh_shift;
    vh3 >>= vh_shift;
    float sumf = 0.0f;
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (int i = 0; i < 2; ++i) {
        const int sc = q6k_scale_byte(sw0, sw1, sw2, sw3, scale_offset + 4 * i);
        int u0 = 0;
        int u1 = 0;
        int u2 = 0;
        int u3 = 0;
        load_i32x4(xq + (bq8_offset + 2 * i) * 32, iqs % 8, &u0, &u1, &u2, &u3);
        const float d8 = xd[bq8_offset + 2 * i];
        const int acc = dp4a_i8(q6k_pack_vi(vl0, vh0, i), u0, 0) +
                        dp4a_i8(q6k_pack_vi(vl1, vh1, i), u1, 0) +
                        dp4a_i8(q6k_pack_vi(vl2, vh2, i), u2, 0) +
                        dp4a_i8(q6k_pack_vi(vl3, vh3, i), u3, 0);
        // Official lm_head / o_proj. One scale after the four dots.
        sumf += d8 * static_cast<float>(acc * sc);
    }
    return d * sumf;
}

VESPER_HOT float q6k_dot_q8_quad_sc_v(int vl0, int vl1, int vl2, int vl3, int vh0, int vh1, int vh2,
                                      int vh3, float d, const std::int8_t* VESPER_RESTRICT xq,
                                      const float* VESPER_RESTRICT xd, int iqs, int sw0, int sw1,
                                      int sw2, int sw3) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    vh0 >>= vh_shift;
    vh1 >>= vh_shift;
    vh2 >>= vh_shift;
    vh3 >>= vh_shift;
    float sumf = 0.0f;
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (int i = 0; i < 2; ++i) {
        const int sc = q6k_scale_byte(sw0, sw1, sw2, sw3, scale_offset + 4 * i);
        int u0 = 0;
        int u1 = 0;
        int u2 = 0;
        int u3 = 0;
        load_i32x4(xq + (bq8_offset + 2 * i) * 32, iqs % 8, &u0, &u1, &u2, &u3);
        const float d8 = xd[bq8_offset + 2 * i];
        const int acc = dp4a_i8(q6k_pack_vi(vl0, vh0, i), u0, 0) +
                        dp4a_i8(q6k_pack_vi(vl1, vh1, i), u1, 0) +
                        dp4a_i8(q6k_pack_vi(vl2, vh2, i), u2, 0) +
                        dp4a_i8(q6k_pack_vi(vl3, vh3, i), u3, 0);
        sumf += d8 * static_cast<float>(acc * sc);
    }
    return d * sumf;
}

VESPER_HOT float q6k_dot_q8_quad_sc(const unsigned char* VESPER_RESTRICT blk, float d,
                                    const std::int8_t* VESPER_RESTRICT xq,
                                    const float* VESPER_RESTRICT xd, int iqs, int sw0, int sw1,
                                    int sw2, int sw3) {
    return q6k_dot_q8_quad_sc_t<false>(blk, blk + 128, d, xq, xd, iqs, sw0, sw1, sw2, sw3);
}

VESPER_HOT float q6k_dot_q8_quad(const unsigned char* VESPER_RESTRICT blk, float d,
                                 const std::int8_t* VESPER_RESTRICT xq,
                                 const float* VESPER_RESTRICT xd, int iqs) {
    int sw0 = 0;
    int sw1 = 0;
    int sw2 = 0;
    int sw3 = 0;
    q6k_load_scales(blk, &sw0, &sw1, &sw2, &sw3);
    return q6k_dot_q8_quad_sc(blk, d, xq, xd, iqs, sw0, sw1, sw2, sw3);
}

// Eight consecutive iqs (iqs in {0,8,16,24}). Same sum as
// q6k_dot_q8_quad(iqs)+q6k_dot_q8_quad(iqs+4). One cached 16 B scale load.
VESPER_HOT float q6k_dot_q8_oct(const unsigned char* VESPER_RESTRICT blk, float d,
                                const std::int8_t* VESPER_RESTRICT xq,
                                const float* VESPER_RESTRICT xd, int iqs) {
    int sw0 = 0;
    int sw1 = 0;
    int sw2 = 0;
    int sw3 = 0;
    q6k_load_scales(blk, &sw0, &sw1, &sw2, &sw3);
    return q6k_dot_q8_quad_sc(blk, d, xq, xd, iqs, sw0, sw1, sw2, sw3) +
           q6k_dot_q8_quad_sc(blk, d, xq, xd, iqs + 4, sw0, sw1, sw2, sw3);
}

VESPER_HOT float q6k_dot_q8_super(const unsigned char* VESPER_RESTRICT blk, float d,
                                  const std::int8_t* VESPER_RESTRICT xq,
                                  const float* VESPER_RESTRICT xd) {
    int sw0 = 0;
    int sw1 = 0;
    int sw2 = 0;
    int sw3 = 0;
    q6k_load_scales(blk, &sw0, &sw1, &sw2, &sw3);
    float acc = 0.0f;
    for (int t = 0; t < 4; ++t) {
        acc += q6k_dot_q8_quad_sc(blk, d, xq, xd, 8 * t, sw0, sw1, sw2, sw3) +
               q6k_dot_q8_quad_sc(blk, d, xq, xd, 8 * t + 4, sw0, sw1, sw2, sw3);
    }
    return acc;
}

VESPER_HOT int q6k_soa_d_bytes_n(int supers) {
    return (supers * 2 + 15) & ~15;
}

VESPER_HOT std::size_t q6k_soa_row_bytes_n(int supers) {
    return static_cast<std::size_t>(q6k_soa_d_bytes_n(supers)) +
           static_cast<std::size_t>(supers) * (16u + 128u + 64u);
}

// HIP Q6 SoA: row-major padded f16 d (cached), super-major scales,
// half-major 64 B ql, super-major qh. Official oct (iqs 0/8 vs 16/24)
// owns one ql half, so neighboring WGs hit consecutive 64 B instead of
// a 128 B super stride.
VESPER_HOT const unsigned char* q6k_soa_d_plane(const unsigned char* packed, int rows, int supers) {
    (void)rows;
    (void)supers;
    return packed;
}

VESPER_HOT const unsigned char* q6k_soa_scale_plane(const unsigned char* packed, int rows,
                                                    int supers) {
    return packed + static_cast<std::size_t>(rows) * static_cast<std::size_t>(q6k_soa_d_bytes_n(supers));
}

VESPER_HOT const unsigned char* q6k_soa_ql_plane(const unsigned char* packed, int rows, int supers) {
    return q6k_soa_scale_plane(packed, rows, supers) +
           static_cast<std::size_t>(rows) * static_cast<std::size_t>(supers) * 16u;
}

VESPER_HOT const unsigned char* q6k_soa_qh_plane(const unsigned char* packed, int rows, int supers) {
    return q6k_soa_ql_plane(packed, rows, supers) +
           static_cast<std::size_t>(rows) * static_cast<std::size_t>(supers) * 128u;
}

VESPER_HOT const unsigned char* q6k_soa_d(const unsigned char* packed, int rows, int supers, int row,
                                          int s) {
    return q6k_soa_d_plane(packed, rows, supers) +
           static_cast<std::size_t>(row) * static_cast<std::size_t>(q6k_soa_d_bytes_n(supers)) +
           static_cast<std::size_t>(s) * 2u;
}

VESPER_HOT const unsigned char* q6k_soa_scales(const unsigned char* packed, int rows, int supers,
                                               int row, int s) {
    return q6k_soa_scale_plane(packed, rows, supers) +
           (static_cast<std::size_t>(s) * static_cast<std::size_t>(rows) +
            static_cast<std::size_t>(row)) *
               16u;
}

VESPER_HOT const unsigned char* q6k_soa_ql(const unsigned char* packed, int rows, int supers, int row,
                                           int s, int half) {
    return q6k_soa_ql_plane(packed, rows, supers) +
           ((static_cast<std::size_t>(s) * 2u + static_cast<std::size_t>(half)) *
                static_cast<std::size_t>(rows) +
            static_cast<std::size_t>(row)) *
               64u;
}

VESPER_HOT const unsigned char* q6k_soa_qh(const unsigned char* packed, int rows, int supers, int row,
                                           int s) {
    return q6k_soa_qh_plane(packed, rows, supers) +
           (static_cast<std::size_t>(s) * static_cast<std::size_t>(rows) +
            static_cast<std::size_t>(row)) *
               64u;
}

VESPER_HOT float q6k_dot_q8_oct_parts(const unsigned char* VESPER_RESTRICT ql,
                                      const unsigned char* VESPER_RESTRICT qh,
                                      const unsigned char* VESPER_RESTRICT scales, float d,
                                      const std::int8_t* VESPER_RESTRICT xq,
                                      const float* VESPER_RESTRICT xd, int iqs) {
    int sw0 = 0;
    int sw1 = 0;
    int sw2 = 0;
    int sw3 = 0;
    q6k_load_scales_a(scales, &sw0, &sw1, &sw2, &sw3);
    return q6k_dot_q8_quad_sc_t<true>(ql, qh, d, xq, xd, iqs, sw0, sw1, sw2, sw3) +
           q6k_dot_q8_quad_sc_t<true>(ql, qh, d, xq, xd, iqs + 4, sw0, sw1, sw2, sw3);
}

// HIP SoA ql is two 64 B halves. iqs>=16 addressing assumes a 128 B ql
// super, so the second oct pair is remapped to iqs-16 on half 1.
// NT both 32 B ql and 32 B qh of the oct before ALU. Two serial
// 16 B quads would pay a second memory round-trip.
VESPER_HOT float q6k_dot_q8_oct_half(const unsigned char* VESPER_RESTRICT ql,
                                     const unsigned char* VESPER_RESTRICT qh,
                                     const unsigned char* VESPER_RESTRICT scales, float d,
                                     const std::int8_t* VESPER_RESTRICT xq,
                                     const float* VESPER_RESTRICT xd, int iqs) {
    int sw0 = 0;
    int sw1 = 0;
    int sw2 = 0;
    int sw3 = 0;
    q6k_load_scales_a(scales, &sw0, &sw1, &sw2, &sw3);
    const int ql_iqs = iqs & 15;
    const int vh_index = 8 * (iqs / 16) + (iqs % 8);
    int vl0 = 0;
    int vl1 = 0;
    int vl2 = 0;
    int vl3 = 0;
    int vl4 = 0;
    int vl5 = 0;
    int vl6 = 0;
    int vl7 = 0;
    int vh0 = 0;
    int vh1 = 0;
    int vh2 = 0;
    int vh3 = 0;
    int vh4 = 0;
    int vh5 = 0;
    int vh6 = 0;
    int vh7 = 0;
    load_w32x8(ql, ql_iqs, &vl0, &vl1, &vl2, &vl3, &vl4, &vl5, &vl6, &vl7);
    load_w32x8(qh, vh_index, &vh0, &vh1, &vh2, &vh3, &vh4, &vh5, &vh6, &vh7);
    const float s0 =
        q6k_dot_q8_quad_sc_v(vl0, vl1, vl2, vl3, vh0, vh1, vh2, vh3, d, xq, xd, iqs, sw0, sw1, sw2,
                             sw3);
    const float s1 =
        q6k_dot_q8_quad_sc_v(vl4, vl5, vl6, vl7, vh4, vh5, vh6, vh7, d, xq, xd, iqs + 4, sw0, sw1,
                             sw2, sw3);
    return s0 + s1;
}

VESPER_HOT float q6k_dot_q8_super_parts(const unsigned char* VESPER_RESTRICT ql,
                                        const unsigned char* VESPER_RESTRICT qh,
                                        const unsigned char* VESPER_RESTRICT scales, float d,
                                        const std::int8_t* VESPER_RESTRICT xq,
                                        const float* VESPER_RESTRICT xd) {
    int sw0 = 0;
    int sw1 = 0;
    int sw2 = 0;
    int sw3 = 0;
    q6k_load_scales_a(scales, &sw0, &sw1, &sw2, &sw3);
    float acc = 0.0f;
    for (int t = 0; t < 4; ++t) {
        acc += q6k_dot_q8_quad_sc_t<true>(ql, qh, d, xq, xd, 8 * t, sw0, sw1, sw2, sw3) +
               q6k_dot_q8_quad_sc_t<true>(ql, qh, d, xq, xd, 8 * t + 4, sw0, sw1, sw2, sw3);
    }
    return acc;
}

VESPER_HOT float q6k_dot_q8_super_halves(const unsigned char* VESPER_RESTRICT ql0,
                                         const unsigned char* VESPER_RESTRICT ql1,
                                         const unsigned char* VESPER_RESTRICT qh,
                                         const unsigned char* VESPER_RESTRICT scales, float d,
                                         const std::int8_t* VESPER_RESTRICT xq,
                                         const float* VESPER_RESTRICT xd) {
    return q6k_dot_q8_oct_half(ql0, qh, scales, d, xq, xd, 0) +
           q6k_dot_q8_oct_half(ql0, qh, scales, d, xq, xd, 8) +
           q6k_dot_q8_oct_half(ql1, qh, scales, d, xq, xd, 16) +
           q6k_dot_q8_oct_half(ql1, qh, scales, d, xq, xd, 24);
}

}  // namespace vesper

#undef VESPER_RESTRICT
#undef VESPER_HOT
