#include "vesper/rdna4.h"

#include "vesper/dotq.h"
#include "vesper/hip.h"
#include "vesper/multi_row.h"
#include "vesper/q4k.h"
#include "vesper/q5k.h"
#include "vesper/q6k.h"
#include "vesper/q8.h"
#include "vesper/q8x.h"
#include "vesper/target.h"
#include "vesper/types.h"
#include "vesper/weight.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace vesper {
namespace rdna4 {
namespace {

#define VESPER_HIP_CHECK(cmd)                                                              \
    do {                                                                                   \
        const hipError_t err_ = (cmd);                                                     \
        if (err_ != hipSuccess) {                                                          \
            fail(std::string("HIP: ") + hipGetErrorString(err_) + " (" + #cmd + ")");      \
        }                                                                                  \
    } while (0)

// Wave32 block reduce: LDS down to one wave, then shuffle 16…1.
// Do not copy CDNA wave64 offsets (32…1).
__device__ float block_reduce_sum(float value, float* smem) {
    const int tid = static_cast<int>(threadIdx.x);
    smem[tid] = value;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride >= kWavefront; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float v = (tid < kWavefront) ? smem[tid] : 0.0f;
    if (tid < kWavefront) {
        v += __shfl_down(v, 16, kWavefront);
        v += __shfl_down(v, 8, kWavefront);
        v += __shfl_down(v, 4, kWavefront);
        v += __shfl_down(v, 2, kWavefront);
        v += __shfl_down(v, 1, kWavefront);
        if (tid == 0) {
            smem[0] = v;
        }
    }
    __syncthreads();
    return smem[0];
}

__device__ float block_reduce_max(float value, float* smem) {
    const int tid = static_cast<int>(threadIdx.x);
    smem[tid] = value;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride >= kWavefront; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    float v = (tid < kWavefront) ? smem[tid] : -INFINITY;
    if (tid < kWavefront) {
        v = fmaxf(v, __shfl_down(v, 16, kWavefront));
        v = fmaxf(v, __shfl_down(v, 8, kWavefront));
        v = fmaxf(v, __shfl_down(v, 4, kWavefront));
        v = fmaxf(v, __shfl_down(v, 2, kWavefront));
        v = fmaxf(v, __shfl_down(v, 1, kWavefront));
        if (tid == 0) {
            smem[0] = v;
        }
    }
    __syncthreads();
    return smem[0];
}

__device__ int decode_i32(int val, const int* ptr) {
    return ptr != nullptr ? *ptr : val;
}

// Same IEEE half decode as src/q8.cpp so CPU and HIP share bits.
__device__ float f16_to_f32_dev(std::uint16_t h) {
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
    return __uint_as_float(out);
}

__device__ float wave32_sum(float v) {
    v += __shfl_down(v, 16, kWavefront);
    v += __shfl_down(v, 8, kWavefront);
    v += __shfl_down(v, 4, kWavefront);
    v += __shfl_down(v, 2, kWavefront);
    v += __shfl_down(v, 1, kWavefront);
    return v;
}

__device__ float wave32_max(float v) {
    v = fmaxf(v, __shfl_down(v, 16, kWavefront));
    v = fmaxf(v, __shfl_down(v, 8, kWavefront));
    v = fmaxf(v, __shfl_down(v, 4, kWavefront));
    v = fmaxf(v, __shfl_down(v, 2, kWavefront));
    v = fmaxf(v, __shfl_down(v, 1, kWavefront));
    return v;
}

// llama.cpp RDNA4 MMVQ: 8 waves, 1 output row. Q8_1 x lives in a process scratch.
struct Q8XDevice {
    std::int8_t* qs = nullptr;
    float* d = nullptr;
    float* sum = nullptr;
    int cap = 0;
};

Q8XDevice g_q8x;

void q8x_ensure(int cols) {
    if (cols <= g_q8x.cap) {
        return;
    }
    if (g_q8x.qs != nullptr) {
        hip_free(g_q8x.qs);
        hip_free(g_q8x.d);
        hip_free(g_q8x.sum);
    }
    const int blocks = cols / kQ8XBlockElems;
    g_q8x.qs = static_cast<std::int8_t*>(hip_alloc(static_cast<std::size_t>(cols)));
    g_q8x.d = static_cast<float*>(hip_alloc(sizeof(float) * static_cast<std::size_t>(blocks)));
    g_q8x.sum = static_cast<float*>(hip_alloc(sizeof(float) * static_cast<std::size_t>(blocks)));
    g_q8x.cap = cols;
}

__global__ void __launch_bounds__(256)
quantize_q8x_kernel(std::int8_t* __restrict__ qs, float* __restrict__ d, float* __restrict__ sum,
                    const float* __restrict__ x, int nblocks) {
    const int b = static_cast<int>(blockIdx.x) * kQuantizeRowsPerWg +
                  static_cast<int>(threadIdx.x) / kWavefront;
    const int lane = static_cast<int>(threadIdx.x) % kWavefront;
    if (b >= nblocks) {
        return;
    }
    const float v = x[b * kQ8XBlockElems + lane];
    const float amax = wave32_max(fabsf(v));
    const float db = amax / 127.0f;
    const float inv = (db > 0.0f) ? (1.0f / db) : 0.0f;
    int q = static_cast<int>(roundf(v * inv));
    if (q > 127) {
        q = 127;
    }
    if (q < -127) {
        q = -127;
    }
    qs[b * kQ8XBlockElems + lane] = static_cast<std::int8_t>(q);
    const float qsum = wave32_sum(static_cast<float>(q));
    if (lane == 0) {
        d[b] = db;
        sum[b] = db * qsum;
    }
}

void q8x_quantize(const float* x, int cols) {
    check(cols > 0 && (cols % kQ8XBlockElems) == 0, "q8x: cols must be a multiple of 32");
    q8x_ensure(cols);
    const int nblocks = cols / kQ8XBlockElems;
    const int grid = (nblocks + kQuantizeRowsPerWg - 1) / kQuantizeRowsPerWg;
    hipLaunchKernelGGL(quantize_q8x_kernel, dim3(grid), dim3(kGemvWorkgroup), 0, hip_stream(), g_q8x.qs,
                       g_q8x.d, g_q8x.sum, x, nblocks);
    VESPER_HIP_CHECK(hipGetLastError());
}

__device__ void write_gemv(float* y, int row, float acc, const float* addend) {
    if (addend != nullptr) {
        acc += addend[row];
    }
    y[row] = acc;
}

__device__ int gemv_row() {
    return static_cast<int>(blockIdx.x);
}

__device__ int gemv_tid() {
    return static_cast<int>(threadIdx.x);
}

__device__ int gemv_lane() {
    return static_cast<int>(threadIdx.x) % kWavefront;
}

__device__ int gemv_warp() {
    return static_cast<int>(threadIdx.x) / kWavefront;
}

__device__ float reduce_wg(float acc) {
    __shared__ float smem[kGemvWorkgroup];
    return block_reduce_sum(acc, smem);
}

static_assert((kQ8VdrMmvq * kGemvWorkgroup) % kQ8Qi == 0,
              "Q8 MMVQ thread mapping needs workgroup divisible by QI/VDR");

__device__ void q4k_decode_scales(const unsigned char* packed, int* scales, int* mins);
__device__ float silu_dev(float g);

__device__ float q5k_lane_dot_q8(const unsigned char* blk, const std::int8_t* xq, const float* xd,
                                int lane) {
    const std::uint16_t dbits =
        static_cast<std::uint16_t>(blk[0]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[1]) << 8);
    const std::uint16_t dminbits =
        static_cast<std::uint16_t>(blk[2]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[3]) << 8);
    const float d = f16_to_f32_dev(dbits);
    const float dmin = f16_to_f32_dev(dminbits);
    int scales[8];
    int mins[8];
    q4k_decode_scales(blk + 4, scales, mins);
    const unsigned char* qh = blk + 16;
    const unsigned char* qs = blk + 48;
    float acc = 0.0f;
    for (int il = 0; il < 4; ++il) {
        const int is = 2 * il;
        unsigned char hm = static_cast<unsigned char>(1u << (2 * il));
        const unsigned char byte = qs[32 * il + lane];
        const int qlo = (byte & 0x0f) + ((qh[lane] & hm) ? 16 : 0);
        hm = static_cast<unsigned char>(hm << 1);
        const int qhi = (byte >> 4) + ((qh[lane] & hm) ? 16 : 0);
        const float xlo = xd[is] * static_cast<float>(xq[64 * il + lane]);
        const float xhi = xd[is + 1] * static_cast<float>(xq[64 * il + 32 + lane]);
        acc += (d * static_cast<float>(scales[is]) * static_cast<float>(qlo) -
                dmin * static_cast<float>(mins[is])) *
               xlo;
        acc += (d * static_cast<float>(scales[is + 1]) * static_cast<float>(qhi) -
                dmin * static_cast<float>(mins[is + 1])) *
               xhi;
    }
    return acc;
}

__device__ float q6k_lane_dot_q8(const unsigned char* blk, const std::int8_t* xq, const float* xd,
                                int lane) {
    const std::uint16_t dbits =
        static_cast<std::uint16_t>(blk[208]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[209]) << 8);
    return q6k_dot_q8_iqs(blk, f16_to_f32_dev(dbits), xq, xd, lane);
}

__host__ __device__ int q8x_lds_bytes(int cols) {
    return ((cols + 3) & ~3) + (cols / 32) * static_cast<int>(sizeof(float));
}

__device__ void maybe_stage_q8x(const std::int8_t*& xq, const float*& xd, int cols) {
    if (q8x_lds_bytes(cols) > kLdsQ8xMaxBytes) {
        return;
    }
    extern __shared__ unsigned char sm[];
    std::int8_t* qs_s = reinterpret_cast<std::int8_t*>(sm);
    float* xd_s = reinterpret_cast<float*>(sm + ((cols + 3) & ~3));
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    for (int i = tid; i < cols; i += nthreads) {
        qs_s[i] = xq[i];
    }
    const int nxd = cols / 32;
    for (int i = tid; i < nxd; i += nthreads) {
        xd_s[i] = xd[i];
    }
    __syncthreads();
    xq = qs_s;
    xd = xd_s;
}

__device__ bool bind_multi_row(int r0, int r1, int r2, int r3, float* y0, float* y1, float* y2,
                               float* y3, const unsigned char* p0, const unsigned char* p1,
                               const unsigned char* p2, const unsigned char* p3, float*& y,
                               const unsigned char*& packed, int& local) {
    vesper::MultiRowPick pick;
    if (!vesper::pick_multi_row(gemv_row(), r0, r1, r2, r3, &pick)) {
        return false;
    }
    float* ys[4] = {y0, y1, y2, y3};
    const unsigned char* ps[4] = {p0, p1, p2, p3};
    y = ys[pick.slot];
    packed = ps[pick.slot];
    local = pick.local;
    return true;
}

__device__ float mmvq_q8_partial(const unsigned char* packed, const std::int8_t* xqs,
                                 const float* xds, int row, int cols) {
    const int tid = gemv_tid();
    const int nblocks = cols / kQ8BlockElems;
    const unsigned char* rowp =
        packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(nblocks) *
                     static_cast<std::size_t>(kQ8BlockBytes);
    constexpr int per_iter = kQ8VdrMmvq * kGemvWorkgroup / kQ8Qi;
    float acc = 0.0f;
    for (int kbx = tid / (kQ8Qi / kQ8VdrMmvq); kbx < nblocks; kbx += per_iter) {
        const int iqs = kQ8VdrMmvq * (tid % (kQ8Qi / kQ8VdrMmvq));
        const unsigned char* blk = rowp + kbx * kQ8BlockBytes;
        const std::uint16_t dbits =
            static_cast<std::uint16_t>(blk[0]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[1]) << 8);
        acc += q8_dot_q8_iqs(reinterpret_cast<const std::int8_t*>(blk + 2), f16_to_f32_dev(dbits),
                             xqs + kbx * kQ8BlockElems, xds[kbx], iqs);
    }
    return acc;
}

__device__ void mmvq_q8_row(float* y, const unsigned char* packed, const std::int8_t* xqs,
                            const float* xds, int row, int cols, const float* addend) {
    const float acc = reduce_wg(mmvq_q8_partial(packed, xqs, xds, row, cols));
    if (gemv_tid() == 0) {
        write_gemv(y, row, acc, addend);
    }
}

__device__ void mmvq_q4k_row(float* y, const unsigned char* packed, const std::int8_t* xqs,
                             const float* xds, int row, int cols, const float* addend) {
    const int tid = gemv_tid();
    const int supers = cols / 256;
    const unsigned char* rowp =
        packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 144u;
    const int iqs = 2 * (tid % 16);
    float acc = 0.0f;
    for (int s = tid / 16; s < supers; s += 16) {
        const unsigned char* blk = rowp + s * 144;
        const std::uint16_t dbits = static_cast<std::uint16_t>(blk[0]) |
                                    static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[1]) << 8);
        const std::uint16_t dminbits =
            static_cast<std::uint16_t>(blk[2]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[3]) << 8);
        acc += q4k_dot_q8_iqs(blk, f16_to_f32_dev(dbits), f16_to_f32_dev(dminbits), xqs + s * 256,
                              xds + s * 8, iqs);
    }
    acc = reduce_wg(acc);
    if (tid == 0) {
        write_gemv(y, row, acc, addend);
    }
}

__device__ void mmvq_q5k_row(float* y, const unsigned char* packed, const std::int8_t* xqs,
                             const float* xds, int row, int cols, const float* addend) {
    const int tid = gemv_tid();
    const int lane = gemv_lane();
    const int warp = gemv_warp();
    const int supers = cols / 256;
    const unsigned char* rowp =
        packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 176u;
    float acc = 0.0f;
    for (int s = warp; s < supers; s += kGemvWaves) {
        acc += q5k_lane_dot_q8(rowp + s * 176, xqs + s * 256, xds + s * 8, lane);
    }
    acc = reduce_wg(acc);
    if (tid == 0) {
        write_gemv(y, row, acc, addend);
    }
}

__device__ void mmvq_q6k_row(float* y, const unsigned char* packed, const std::int8_t* xqs,
                             const float* xds, int row, int cols, const float* addend) {
    const int tid = gemv_tid();
    const int lane = gemv_lane();
    const int warp = gemv_warp();
    const int supers = cols / 256;
    const unsigned char* rowp =
        packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 210u;
    float acc = 0.0f;
    for (int s = warp; s < supers; s += kGemvWaves) {
        acc += q6k_lane_dot_q8(rowp + s * 210, xqs + s * 256, xds + s * 8, lane);
    }
    acc = reduce_wg(acc);
    if (tid == 0) {
        write_gemv(y, row, acc, addend);
    }
}

// llama.cpp RDNA4: 8 waves, 1 output row. Official Q4_K_M attn/SSM is Q8_0.
__global__ void __launch_bounds__(256)
gemv_q8_mmvq_kernel(float* __restrict__ y, const unsigned char* __restrict__ packed,
                    const std::int8_t* __restrict__ xq, const float* __restrict__ xd, int rows,
                    int cols, const float* __restrict__ addend) {
    const std::int8_t* xqs = xq;
    const float* xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    if (row >= rows) {
        return;
    }
    mmvq_q8_row(y, packed, xqs, xds, row, cols, addend);
}

int gemv_block(int in_features) {
    if (in_features <= 64) {
        return 64;
    }
    if (in_features <= 256) {
        return 128;
    }
    return kGemvWorkgroup;
}

// One workgroup per output row. float4 loads cover 16 B; 16 threads cover a
// 256 B gfx1201 cacheline. Several independent FP32 accumulators.
__global__ void __launch_bounds__(256)
gemv_f32_kernel(float* __restrict__ y, const float* __restrict__ weight,
                const float* __restrict__ x, int out_features, int in_features,
                const float* __restrict__ addend) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= out_features) {
        return;
    }

    const float* row_w = weight + static_cast<std::size_t>(row) * in_features;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    // float4 only when every row is 16-byte aligned (in % 4 == 0).
    if ((in_features & 3) == 0) {
        const int n4 = in_features >> 2;
        const float4* w4 = reinterpret_cast<const float4*>(row_w);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int j = tid; j < n4; j += nthreads) {
            const float4 wv = w4[j];
            const float4 xv = x4[j];
            acc0 += wv.x * xv.x;
            acc1 += wv.y * xv.y;
            acc2 += wv.z * xv.z;
            acc3 += wv.w * xv.w;
        }
    } else {
        for (int j = tid; j < in_features; j += nthreads) {
            acc0 += row_w[j] * x[j];
        }
    }
    const float acc = acc0 + acc1 + acc2 + acc3;

    __shared__ float smem[kGemvWorkgroup];
    const float sum = block_reduce_sum(acc, smem);
    if (tid == 0) {
        write_gemv(y, row, sum, addend);
    }
}

__global__ void __launch_bounds__(256)
rmsnorm_f32_kernel(float* __restrict__ out, const float* __restrict__ x,
                   const float* __restrict__ weight, int n, float eps) {
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    float ss = 0.0f;
    for (int i = tid; i < n; i += nthreads) {
        const float v = x[i];
        ss += v * v;
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss / static_cast<float>(n) + eps);
    for (int i = tid; i < n; i += nthreads) {
        out[i] = x[i] * inv * weight[i];
    }
}

__global__ void rope_neox_kernel(float* __restrict__ q, float* __restrict__ k, int n_q_heads,
                                 int n_kv_heads, int head_dim, int rotary_dim, int pos_val,
                                 const int* pos_ptr, float theta) {
    const int pos = decode_i32(pos_val, pos_ptr);
    const int half = rotary_dim / 2;
    const int total = (n_q_heads + n_kv_heads) * half;
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) {
        return;
    }

    const bool on_q = idx < n_q_heads * half;
    const int local = on_q ? idx : idx - n_q_heads * half;
    const int head = local / half;
    const int i = local - head * half;
    float* vec = on_q ? q : k;
    float* pair = vec + head * head_dim;

    const float freq =
        1.0f / powf(theta, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
    const float angle = static_cast<float>(pos) * freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float x0 = pair[i];
    const float x1 = pair[i + half];
    pair[i] = x0 * c - x1 * s;
    pair[i + half] = x0 * s + x1 * c;
}

__global__ void __launch_bounds__(256)
rope_neox_k_norm_kernel(float* __restrict__ q, float* __restrict__ k,
                        const float* __restrict__ k_weight, int n_q_heads, int n_kv_heads,
                        int head_dim, int rotary_dim, int pos_val, const int* pos_ptr, float theta,
                        float eps) {
    const int pos = decode_i32(pos_val, pos_ptr);
    const int h = static_cast<int>(blockIdx.x);
    const int total_heads = n_q_heads + n_kv_heads;
    if (h >= total_heads) {
        return;
    }
    const bool on_q = h < n_q_heads;
    const int local = on_q ? h : h - n_q_heads;
    float* head = (on_q ? q : k) + local * head_dim;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    if (!on_q) {
        __shared__ float smem[kGemvWorkgroup];
        float ss = 0.0f;
        for (int i = tid; i < head_dim; i += nthreads) {
            const float v = head[i];
            ss += v * v;
        }
        ss = block_reduce_sum(ss, smem);
        const float inv = rsqrtf(ss / static_cast<float>(head_dim) + eps);
        for (int i = tid; i < head_dim; i += nthreads) {
            head[i] = head[i] * inv * k_weight[i];
        }
        __syncthreads();
    }
    const int half = rotary_dim / 2;
    for (int i = tid; i < half; i += nthreads) {
        const float freq =
            1.0f / powf(theta, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
        const float angle = static_cast<float>(pos) * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float x0 = head[i];
        const float x1 = head[i + half];
        head[i] = x0 * c - x1 * s;
        head[i + half] = x0 * s + x1 * c;
    }
}

__global__ void swiglu_kernel(float* __restrict__ out, const float* __restrict__ gate,
                              const float* __restrict__ up, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const float g = gate[i];
    const float silu = g / (1.0f + expf(-g));
    out[i] = silu * up[i];
}

__global__ void __launch_bounds__(256)
softmax_kernel(float* __restrict__ x, int n) {
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    float max_v = -INFINITY;
    for (int i = tid; i < n; i += nthreads) {
        max_v = fmaxf(max_v, x[i]);
    }
    max_v = block_reduce_max(max_v, smem);

    float sum = 0.0f;
    for (int i = tid; i < n; i += nthreads) {
        const float e = expf(x[i] - max_v);
        x[i] = e;
        sum += e;
    }
    sum = block_reduce_sum(sum, smem);
    const float inv = 1.0f / sum;
    for (int i = tid; i < n; i += nthreads) {
        x[i] *= inv;
    }
}

__global__ void attn_scores_kernel(float* __restrict__ scores, const float* __restrict__ q,
                                   const float* __restrict__ k, int seq, int n_kv_heads,
                                   int kv_head, int head_dim) {
    const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= seq) {
        return;
    }
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float* key =
        k + (static_cast<std::size_t>(t) * n_kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        dot += q[i] * key[i];
    }
    scores[t] = dot * scale;
}

__global__ void attn_mix_kernel(float* __restrict__ out, const float* __restrict__ scores,
                                const float* __restrict__ v, int seq, int n_kv_heads,
                                int kv_head, int head_dim) {
    const int i = static_cast<int>(threadIdx.x);
    if (i >= head_dim) {
        return;
    }
    float acc = 0.0f;
    for (int t = 0; t < seq; ++t) {
        const float* val =
            v + (static_cast<std::size_t>(t) * n_kv_heads + kv_head) * head_dim;
        acc += scores[t] * val[i];
    }
    out[i] = acc;
}

__global__ void add_inplace_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                   int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] += src[i];
    }
}

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__device__ void q4k_decode_scales(const unsigned char* packed, int* scales, int* mins) {
    for (int j = 0; j < 4; ++j) {
        scales[j] = packed[j] & 0x3f;
        mins[j] = packed[j + 4] & 0x3f;
    }
    for (int j = 4; j < 8; ++j) {
        scales[j] = (packed[j + 4] & 0x0f) | ((packed[j - 4] >> 6) << 4);
        mins[j] = (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
    }
}

// 1 row / WG, 8 waves on K. 16 iqs slices, 16 supers in flight.
__global__ void __launch_bounds__(256)
gemv_q4k_mmvq_kernel(float* __restrict__ y, const unsigned char* __restrict__ packed,
                     const std::int8_t* __restrict__ xq, const float* __restrict__ xd, int rows,
                     int cols, const float* __restrict__ addend) {
    const std::int8_t* xqs = xq;
    const float* xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    if (row >= rows) {
        return;
    }
    mmvq_q4k_row(y, packed, xqs, xds, row, cols, addend);
}

__global__ void __launch_bounds__(256)
gemv_q5k_mmvq_kernel(float* __restrict__ y, const unsigned char* __restrict__ packed,
                     const std::int8_t* __restrict__ xq, const float* __restrict__ xd, int rows,
                     int cols, const float* __restrict__ addend) {
    const std::int8_t* xqs = xq;
    const float* xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    if (row >= rows) {
        return;
    }
    mmvq_q5k_row(y, packed, xqs, xds, row, cols, addend);
}

__global__ void __launch_bounds__(256)
gemv_q6k_mmvq_kernel(float* __restrict__ y, const unsigned char* __restrict__ packed,
                     const std::int8_t* __restrict__ xq, const float* __restrict__ xd, int rows,
                     int cols, const float* __restrict__ addend) {
    const std::int8_t* xqs = xq;
    const float* xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    if (row >= rows) {
        return;
    }
    mmvq_q6k_row(y, packed, xqs, xds, row, cols, addend);
}

// One grid over Q/K/V or GDN qkv/z/beta/alpha. Stage Q8_1 x once per WG, then
// pick the output matrix from blockIdx. Host already quantized x once.
__global__ void __launch_bounds__(256)
gemv_q8_mmvq_multi_kernel(float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
                          float* __restrict__ y3, const unsigned char* __restrict__ p0,
                          const unsigned char* __restrict__ p1, const unsigned char* __restrict__ p2,
                          const unsigned char* __restrict__ p3, const std::int8_t* __restrict__ xq,
                          const float* __restrict__ xd, int r0, int r1, int r2, int r3, int cols) {
    const std::int8_t* xqs = xq;
    const float* xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    float* y = nullptr;
    const unsigned char* packed = nullptr;
    int local = 0;
    if (!bind_multi_row(r0, r1, r2, r3, y0, y1, y2, y3, p0, p1, p2, p3, y, packed, local)) {
        return;
    }
    mmvq_q8_row(y, packed, xqs, xds, local, cols, nullptr);
}

__global__ void __launch_bounds__(256)
gemv_q4k_mmvq_multi_kernel(float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
                           float* __restrict__ y3, const unsigned char* __restrict__ p0,
                           const unsigned char* __restrict__ p1, const unsigned char* __restrict__ p2,
                           const unsigned char* __restrict__ p3, const std::int8_t* __restrict__ xq,
                           const float* __restrict__ xd, int r0, int r1, int r2, int r3, int cols) {
    const std::int8_t* xqs = xq;
    const float* xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    float* y = nullptr;
    const unsigned char* packed = nullptr;
    int local = 0;
    if (!bind_multi_row(r0, r1, r2, r3, y0, y1, y2, y3, p0, p1, p2, p3, y, packed, local)) {
        return;
    }
    mmvq_q4k_row(y, packed, xqs, xds, local, cols, nullptr);
}

__global__ void __launch_bounds__(256)
gemv_q5k_mmvq_multi_kernel(float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
                           float* __restrict__ y3, const unsigned char* __restrict__ p0,
                           const unsigned char* __restrict__ p1, const unsigned char* __restrict__ p2,
                           const unsigned char* __restrict__ p3, const std::int8_t* __restrict__ xq,
                           const float* __restrict__ xd, int r0, int r1, int r2, int r3, int cols) {
    const std::int8_t* xqs = xq;
    const float* xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    float* y = nullptr;
    const unsigned char* packed = nullptr;
    int local = 0;
    if (!bind_multi_row(r0, r1, r2, r3, y0, y1, y2, y3, p0, p1, p2, p3, y, packed, local)) {
        return;
    }
    mmvq_q5k_row(y, packed, xqs, xds, local, cols, nullptr);
}

__global__ void __launch_bounds__(256)
gemv_q6k_mmvq_multi_kernel(float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
                           float* __restrict__ y3, const unsigned char* __restrict__ p0,
                           const unsigned char* __restrict__ p1, const unsigned char* __restrict__ p2,
                           const unsigned char* __restrict__ p3, const std::int8_t* __restrict__ xq,
                           const float* __restrict__ xd, int r0, int r1, int r2, int r3, int cols) {
    const std::int8_t* xqs = xq;
    const float* xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    float* y = nullptr;
    const unsigned char* packed = nullptr;
    int local = 0;
    if (!bind_multi_row(r0, r1, r2, r3, y0, y1, y2, y3, p0, p1, p2, p3, y, packed, local)) {
        return;
    }
    mmvq_q6k_row(y, packed, xqs, xds, local, cols, nullptr);
}

__global__ void sigmoid_kernel(float* __restrict__ x, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const float v = x[i];
    if (v >= 0.0f) {
        const float z = expf(-v);
        x[i] = 1.0f / (1.0f + z);
    } else {
        const float z = expf(v);
        x[i] = z / (1.0f + z);
    }
}

__global__ void silu_kernel(float* __restrict__ x, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        const float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

__global__ void softplus_kernel(float* __restrict__ x, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const float v = x[i];
    if (v > 20.0f) {
        return;
    }
    if (v < -20.0f) {
        x[i] = expf(v);
        return;
    }
    x[i] = log1pf(expf(v));
}

__global__ void exp_kernel(float* __restrict__ x, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        x[i] = expf(x[i]);
    }
}

__global__ void mul_inplace_kernel(float* __restrict__ dst, const float* __restrict__ src, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] *= src[i];
    }
}

__global__ void scale_inplace_kernel(float* __restrict__ x, float scale, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        x[i] *= scale;
    }
}

__global__ void l2_normalize_rows_kernel(float* __restrict__ x, int rows, int dim, float eps) {
    const int r = static_cast<int>(blockIdx.x);
    if (r >= rows) {
        return;
    }
    float* row = x + r * dim;
    float ss = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < dim; i += static_cast<int>(blockDim.x)) {
        ss += row[i] * row[i];
    }
    ss = wave32_sum(ss);
    __shared__ float inv;
    if (threadIdx.x == 0) {
        inv = rsqrtf(ss + eps);
    }
    __syncthreads();
    for (int i = static_cast<int>(threadIdx.x); i < dim; i += static_cast<int>(blockDim.x)) {
        row[i] *= inv;
    }
}

__global__ void gdn_conv_update_kernel(float* __restrict__ y, float* __restrict__ state,
                                      const float* __restrict__ x, const float* __restrict__ weight,
                                      int conv_dim, int kernel) {
    const int d = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (d >= conv_dim) {
        return;
    }
    const int hist = kernel - 1;
    float acc = 0.0f;
    for (int t = 0; t < hist; ++t) {
        acc += weight[d * kernel + t] * state[d * hist + t];
    }
    acc += weight[d * kernel + hist] * x[d];
    y[d] = acc / (1.0f + expf(-acc));
    for (int t = 0; t < hist - 1; ++t) {
        state[d * hist + t] = state[d * hist + t + 1];
    }
    state[d * hist + (hist - 1)] = x[d];
}

// llama.cpp gated_delta_net.cu: one warp owns one S column in registers.
// rec is [n_v][dv][dk] so column j is contiguous. dim <= kGdnDeltaMaxDim.
__global__ void __launch_bounds__(kWavefront * kGdnDeltaWarps)
gdn_delta_rule_kernel(float* __restrict__ y, float* __restrict__ rec,
                      const float* __restrict__ q, const float* __restrict__ k,
                      const float* __restrict__ v, const float* __restrict__ decay,
                      const float* __restrict__ beta, int n_heads, int dim) {
    const int h = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.y) * kGdnDeltaWarps + warp;
    if (h >= n_heads || col >= dim) {
        return;
    }
    float* scol = rec + (static_cast<std::size_t>(h) * dim + col) * dim;
    const float* qh = q + h * dim;
    const float* kh = k + h * dim;
    float s_shard[kGdnDeltaRowsPerLane];
    float k_reg[kGdnDeltaRowsPerLane];
    float q_reg[kGdnDeltaRowsPerLane];
    const int rows_per_lane = (dim + kWavefront - 1) / kWavefront;
    for (int r = 0; r < kGdnDeltaRowsPerLane; ++r) {
        const int i = r * kWavefront + lane;
        const bool on = r < rows_per_lane && i < dim;
        s_shard[r] = on ? scol[i] : 0.0f;
        k_reg[r] = on ? kh[i] : 0.0f;
        q_reg[r] = on ? qh[i] : 0.0f;
    }
    float kv = 0.0f;
    for (int r = 0; r < kGdnDeltaRowsPerLane; ++r) {
        kv += s_shard[r] * k_reg[r];
    }
    kv = wave32_sum(kv);
    const float g = decay[h];
    const float delta = beta[h] * (v[h * dim + col] - g * kv);
    float attn = 0.0f;
    for (int r = 0; r < kGdnDeltaRowsPerLane; ++r) {
        s_shard[r] = g * s_shard[r] + k_reg[r] * delta;
        attn += s_shard[r] * q_reg[r];
    }
    attn = wave32_sum(attn);
    if (lane == 0) {
        y[h * dim + col] = attn;
    }
    for (int r = 0; r < kGdnDeltaRowsPerLane; ++r) {
        const int i = r * kWavefront + lane;
        if (r < rows_per_lane && i < dim) {
            scol[i] = s_shard[r];
        }
    }
}

__global__ void __launch_bounds__(256)
rmsnorm_rows_kernel(float* __restrict__ x, const float* __restrict__ weight, int rows, int dim,
                    float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    float* y = x + row * dim;
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float ss = 0.0f;
    for (int i = tid; i < dim; i += nthreads) {
        const float v = y[i];
        ss += v * v;
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss / static_cast<float>(dim) + eps);
    for (int i = tid; i < dim; i += nthreads) {
        y[i] = y[i] * inv * weight[i];
    }
}

__global__ void split_gated_q_kernel(float* __restrict__ q, float* __restrict__ gate,
                                     const float* __restrict__ q_full, int n_heads, int head_dim) {
    const int h = static_cast<int>(blockIdx.x);
    if (h >= n_heads) {
        return;
    }
    const float* src = q_full + h * 2 * head_dim;
    for (int i = static_cast<int>(threadIdx.x); i < head_dim; i += static_cast<int>(blockDim.x)) {
        q[h * head_dim + i] = src[i];
        gate[h * head_dim + i] = src[head_dim + i];
    }
}

__global__ void __launch_bounds__(256)
split_gated_q_norm_kernel(float* __restrict__ q, float* __restrict__ gate,
                          const float* __restrict__ q_full, const float* __restrict__ weight,
                          int n_heads, int head_dim, float eps) {
    const int h = static_cast<int>(blockIdx.x);
    if (h >= n_heads) {
        return;
    }
    const float* src = q_full + h * 2 * head_dim;
    float* qh = q + h * head_dim;
    float* gh = gate + h * head_dim;
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float ss = 0.0f;
    for (int i = tid; i < head_dim; i += nthreads) {
        const float v = src[i];
        qh[i] = v;
        gh[i] = src[head_dim + i];
        ss += v * v;
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss / static_cast<float>(head_dim) + eps);
    for (int i = tid; i < head_dim; i += nthreads) {
        qh[i] = qh[i] * inv * weight[i];
    }
}

__global__ void __launch_bounds__(256)
rmsnorm_silu_mul_kernel(float* __restrict__ y, const float* __restrict__ z,
                        const float* __restrict__ weight, int rows, int dim, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    float* yr = y + row * dim;
    const float* zr = z + row * dim;
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float ss = 0.0f;
    for (int i = tid; i < dim; i += nthreads) {
        const float v = yr[i];
        ss += v * v;
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss / static_cast<float>(dim) + eps);
    for (int i = tid; i < dim; i += nthreads) {
        yr[i] = yr[i] * inv * weight[i] * silu_dev(zr[i]);
    }
}

__global__ void gdn_conv_split_kernel(float* __restrict__ q, float* __restrict__ k,
                                      float* __restrict__ v, float* __restrict__ state,
                                      const float* __restrict__ x, const float* __restrict__ weight,
                                      int key_dim, int value_dim, int kernel) {
    const int qkv_dim = 2 * key_dim + value_dim;
    const int d = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (d >= qkv_dim) {
        return;
    }
    const int hist = kernel - 1;
    float acc = 0.0f;
    for (int t = 0; t < hist; ++t) {
        acc += weight[d * kernel + t] * state[d * hist + t];
    }
    acc += weight[d * kernel + hist] * x[d];
    const float y = acc / (1.0f + expf(-acc));
    if (d < key_dim) {
        q[d] = y;
    } else if (d < 2 * key_dim) {
        k[d - key_dim] = y;
    } else {
        v[d - 2 * key_dim] = y;
    }
    for (int t = 0; t < hist - 1; ++t) {
        state[d * hist + t] = state[d * hist + t + 1];
    }
    state[d * hist + (hist - 1)] = x[d];
}

__global__ void tile_heads_kernel(float* __restrict__ dst, const float* __restrict__ src, int n_dst,
                                  int n_src, int dim) {
    const int d = static_cast<int>(blockIdx.x);
    if (d >= n_dst) {
        return;
    }
    const int s = d % n_src;
    for (int i = static_cast<int>(threadIdx.x); i < dim; i += static_cast<int>(blockDim.x)) {
        dst[d * dim + i] = src[s * dim + i];
    }
}

__global__ void __launch_bounds__(256)
attn_decode_kernel(float* __restrict__ out, const float* __restrict__ q, const float* __restrict__ k,
                   const float* __restrict__ v, const float* __restrict__ gate, int seq_val,
                   const int* pos_ptr, int n_q_heads, int n_kv_heads, int head_dim) {
    const int seq = pos_ptr != nullptr ? (*pos_ptr + 1) : seq_val;
    const int qh = static_cast<int>(blockIdx.x);
    if (qh >= n_q_heads) {
        return;
    }
    const int group = n_q_heads / n_kv_heads;
    const int kvh = qh / group;
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float smem[kGemvWorkgroup];
    __shared__ float q_s[kGemvWorkgroup];
    const float* qh_ptr = q + qh * head_dim;
    if (tid < head_dim) {
        q_s[tid] = qh_ptr[tid];
    }
    __syncthreads();
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float acc = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    const bool owns = tid < head_dim;
    for (int t = 0; t < seq; ++t) {
        const float* kt = k + (static_cast<std::size_t>(t) * n_kv_heads + kvh) * head_dim;
        const float* vt = v + (static_cast<std::size_t>(t) * n_kv_heads + kvh) * head_dim;
        float partial = 0.0f;
        for (int i = tid; i < head_dim; i += static_cast<int>(blockDim.x)) {
            partial += q_s[i] * kt[i];
        }
        const float score = block_reduce_sum(partial, smem) * scale;
        const float m_new = fmaxf(m, score);
        const float alpha = expf(m - m_new);
        const float exp_s = expf(score - m_new);
        l = l * alpha + exp_s;
        if (owns) {
            acc = acc * alpha + exp_s * vt[tid];
        }
        m = m_new;
    }
    if (owns) {
        float y = acc / l;
        if (gate != nullptr) {
            const float g = gate[qh * head_dim + tid];
            const float s = (g >= 0.0f) ? (1.0f / (1.0f + expf(-g))) : (expf(g) / (1.0f + expf(g)));
            y *= s;
        }
        out[qh * head_dim + tid] = y;
    }
}

__device__ float silu_dev(float g) {
    return g / (1.0f + expf(-g));
}

__global__ void __launch_bounds__(256)
add_rmsnorm_kernel(float* __restrict__ x, float* __restrict__ residual, const float* __restrict__ weight,
                   int n, float eps) {
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    for (int i = tid; i < n; i += nthreads) {
        residual[i] += x[i];
    }
    __syncthreads();
    float ss = 0.0f;
    for (int i = tid; i < n; i += nthreads) {
        const float v = residual[i];
        ss += v * v;
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss / static_cast<float>(n) + eps);
    for (int i = tid; i < n; i += nthreads) {
        x[i] = residual[i] * inv * weight[i];
    }
}

__global__ void __launch_bounds__(256)
copy_rmsnorm_kernel(float* __restrict__ x, float* __restrict__ residual,
                    const float* __restrict__ weight, int n, float eps) {
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    for (int i = tid; i < n; i += nthreads) {
        residual[i] = x[i];
    }
    __syncthreads();
    float ss = 0.0f;
    for (int i = tid; i < n; i += nthreads) {
        const float v = residual[i];
        ss += v * v;
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss / static_cast<float>(n) + eps);
    for (int i = tid; i < n; i += nthreads) {
        x[i] = residual[i] * inv * weight[i];
    }
}

__global__ void silu_mul_kernel(float* __restrict__ y, const float* __restrict__ z, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const float g = z[i];
    y[i] *= g / (1.0f + expf(-g));
}

__global__ void gdn_gates_kernel(float* __restrict__ decay, float* __restrict__ beta,
                                 const float* __restrict__ alpha, const float* __restrict__ dt,
                                 const float* __restrict__ a, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    float t = alpha[i] + dt[i];
    if (t > 20.0f) {
        // softplus(t) ~= t
    } else if (t < -20.0f) {
        t = expf(t);
    } else {
        t = log1pf(expf(t));
    }
    decay[i] = expf(a[i] * t);
    const float b = beta[i];
    if (b >= 0.0f) {
        beta[i] = 1.0f / (1.0f + expf(-b));
    } else {
        const float z = expf(b);
        beta[i] = z / (1.0f + z);
    }
}

__global__ void split_qkv_kernel(float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
                                 const float* __restrict__ qkv, int key_dim, int value_dim) {
    const int total = 2 * key_dim + value_dim;
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) {
        return;
    }
    if (i < key_dim) {
        q[i] = qkv[i];
        return;
    }
    if (i < 2 * key_dim) {
        k[i - key_dim] = qkv[i];
        return;
    }
    v[i - 2 * key_dim] = qkv[i];
}

__global__ void __launch_bounds__(256)
gemv_swiglu_f32_kernel(float* __restrict__ hidden, const float* __restrict__ gate,
                       const float* __restrict__ up, const float* __restrict__ x, int rows,
                       int cols) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const float* grow = gate + static_cast<std::size_t>(row) * cols;
    const float* urow = up + static_cast<std::size_t>(row) * cols;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float ag = 0.0f;
    float au = 0.0f;
    if ((cols & 3) == 0) {
        const int n4 = cols >> 2;
        const float4* g4 = reinterpret_cast<const float4*>(grow);
        const float4* u4 = reinterpret_cast<const float4*>(urow);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int j = tid; j < n4; j += nthreads) {
            const float4 gv = g4[j];
            const float4 uv = u4[j];
            const float4 xv = x4[j];
            ag += gv.x * xv.x + gv.y * xv.y + gv.z * xv.z + gv.w * xv.w;
            au += uv.x * xv.x + uv.y * xv.y + uv.z * xv.z + uv.w * xv.w;
        }
    } else {
        for (int j = tid; j < cols; j += nthreads) {
            ag += grow[j] * x[j];
            au += urow[j] * x[j];
        }
    }
    __shared__ float smem[kGemvWorkgroup];
    ag = block_reduce_sum(ag, smem);
    au = block_reduce_sum(au, smem);
    if (tid == 0) {
        hidden[row] = silu_dev(ag) * au;
    }
}

__global__ void __launch_bounds__(256)
gemv_swiglu_q8_mmvq_kernel(float* __restrict__ hidden, const unsigned char* __restrict__ gate,
                           const unsigned char* __restrict__ up, const std::int8_t* __restrict__ xq,
                           const float* __restrict__ xd, int rows, int cols) {
    const std::int8_t* xqs = xq;
    const float* xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    if (row >= rows) {
        return;
    }
    float ag = mmvq_q8_partial(gate, xqs, xds, row, cols);
    float au = mmvq_q8_partial(up, xqs, xds, row, cols);
    ag = reduce_wg(ag);
    au = reduce_wg(au);
    if (gemv_tid() == 0) {
        hidden[row] = silu_dev(ag) * au;
    }
}

__global__ void __launch_bounds__(256)
gemv_swiglu_q4k_mmvq_kernel(float* __restrict__ hidden, const unsigned char* __restrict__ gate,
                            const unsigned char* __restrict__ up, const std::int8_t* __restrict__ xq,
                            const float* __restrict__ xd, int rows, int cols) {
    const std::int8_t* xqs = xq;
    const float* xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    const int tid = gemv_tid();
    if (row >= rows) {
        return;
    }
    const int supers = cols / 256;
    const unsigned char* gp =
        gate + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 144u;
    const unsigned char* upw =
        up + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 144u;
    const int iqs = 2 * (tid % 16);
    float ag = 0.0f;
    float au = 0.0f;
    for (int s = tid / 16; s < supers; s += 16) {
        const unsigned char* gb = gp + s * 144;
        const unsigned char* ub = upw + s * 144;
        const std::uint16_t gd = static_cast<std::uint16_t>(gb[0]) |
                                 static_cast<std::uint16_t>(static_cast<std::uint16_t>(gb[1]) << 8);
        const std::uint16_t gmin = static_cast<std::uint16_t>(gb[2]) |
                                   static_cast<std::uint16_t>(static_cast<std::uint16_t>(gb[3]) << 8);
        const std::uint16_t ud = static_cast<std::uint16_t>(ub[0]) |
                                 static_cast<std::uint16_t>(static_cast<std::uint16_t>(ub[1]) << 8);
        const std::uint16_t umin = static_cast<std::uint16_t>(ub[2]) |
                                   static_cast<std::uint16_t>(static_cast<std::uint16_t>(ub[3]) << 8);
        ag += q4k_dot_q8_iqs(gb, f16_to_f32_dev(gd), f16_to_f32_dev(gmin), xqs + s * 256, xds + s * 8,
                             iqs);
        au += q4k_dot_q8_iqs(ub, f16_to_f32_dev(ud), f16_to_f32_dev(umin), xqs + s * 256, xds + s * 8,
                             iqs);
    }
    ag = reduce_wg(ag);
    au = reduce_wg(au);
    if (tid == 0) {
        hidden[row] = silu_dev(ag) * au;
    }
}

__global__ void __launch_bounds__(256)
gemv_swiglu_q5k_mmvq_kernel(float* __restrict__ hidden, const unsigned char* __restrict__ gate,
                            const unsigned char* __restrict__ up, const std::int8_t* __restrict__ xq,
                            const float* __restrict__ xd, int rows, int cols) {
    const std::int8_t* xqs = xq;
    const float* xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    const int tid = gemv_tid();
    const int lane = gemv_lane();
    const int warp = gemv_warp();
    if (row >= rows) {
        return;
    }
    const int supers = cols / 256;
    const unsigned char* gp =
        gate + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 176u;
    const unsigned char* upw =
        up + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 176u;
    float ag = 0.0f;
    float au = 0.0f;
    for (int s = warp; s < supers; s += kGemvWaves) {
        ag += q5k_lane_dot_q8(gp + s * 176, xqs + s * 256, xds + s * 8, lane);
        au += q5k_lane_dot_q8(upw + s * 176, xqs + s * 256, xds + s * 8, lane);
    }
    ag = reduce_wg(ag);
    au = reduce_wg(au);
    if (tid == 0) {
        hidden[row] = silu_dev(ag) * au;
    }
}

__global__ void __launch_bounds__(256)
gemv_swiglu_q6k_mmvq_kernel(float* __restrict__ hidden, const unsigned char* __restrict__ gate,
                            const unsigned char* __restrict__ up, const std::int8_t* __restrict__ xq,
                            const float* __restrict__ xd, int rows, int cols) {
    const std::int8_t* xqs = xq;
    const float* xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    const int tid = gemv_tid();
    const int lane = gemv_lane();
    const int warp = gemv_warp();
    if (row >= rows) {
        return;
    }
    const int supers = cols / 256;
    const unsigned char* gp =
        gate + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 210u;
    const unsigned char* upw =
        up + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 210u;
    float ag = 0.0f;
    float au = 0.0f;
    for (int s = warp; s < supers; s += kGemvWaves) {
        ag += q6k_lane_dot_q8(gp + s * 210, xqs + s * 256, xds + s * 8, lane);
        au += q6k_lane_dot_q8(upw + s * 210, xqs + s * 256, xds + s * 8, lane);
    }
    ag = reduce_wg(ag);
    au = reduce_wg(au);
    if (tid == 0) {
        hidden[row] = silu_dev(ag) * au;
    }
}

__global__ void __launch_bounds__(256)
gemv_f32_multi_kernel(float* __restrict__ y0, const float* __restrict__ w0, int r0,
                      float* __restrict__ y1, const float* __restrict__ w1, int r1,
                      float* __restrict__ y2, const float* __restrict__ w2, int r2,
                      float* __restrict__ y3, const float* __restrict__ w3, int r3,
                      const float* __restrict__ x, int cols) {
    const int row = static_cast<int>(blockIdx.x);
    const int total = r0 + r1 + r2 + r3;
    if (row >= total) {
        return;
    }
    float* y = y0;
    const float* weight = w0;
    int local = row;
    if (row >= r0 + r1 + r2) {
        y = y3;
        weight = w3;
        local = row - (r0 + r1 + r2);
    } else if (row >= r0 + r1) {
        y = y2;
        weight = w2;
        local = row - (r0 + r1);
    } else if (row >= r0) {
        y = y1;
        weight = w1;
        local = row - r0;
    }
    const float* row_w = weight + static_cast<std::size_t>(local) * cols;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float acc = 0.0f;
    if ((cols & 3) == 0) {
        const int n4 = cols >> 2;
        const float4* w4 = reinterpret_cast<const float4*>(row_w);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int j = tid; j < n4; j += nthreads) {
            const float4 wv = w4[j];
            const float4 xv = x4[j];
            acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
    } else {
        for (int j = tid; j < cols; j += nthreads) {
            acc += row_w[j] * x[j];
        }
    }
    __shared__ float smem[kGemvWorkgroup];
    acc = block_reduce_sum(acc, smem);
    if (tid == 0) {
        y[local] = acc;
    }
}

__global__ void tile_l2_scale_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                     int n_dst, int n_src, int dim, float eps, float scale) {
    const int d = static_cast<int>(blockIdx.x);
    if (d >= n_dst) {
        return;
    }
    const int s = d % n_src;
    float* row = dst + d * dim;
    const float* in = src + s * dim;
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float ss = 0.0f;
    for (int i = tid; i < dim; i += nthreads) {
        const float v = in[i];
        row[i] = v;
        ss += v * v;
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss + eps) * scale;
    for (int i = tid; i < dim; i += nthreads) {
        row[i] *= inv;
    }
}

__global__ void tile_l2_pair_kernel(float* __restrict__ q_dst, const float* __restrict__ q_src,
                                    float* __restrict__ k_dst, const float* __restrict__ k_src,
                                    int n_dst, int n_src, int dim, float eps, float q_scale,
                                    float k_scale) {
    const int idx = static_cast<int>(blockIdx.x);
    if (idx >= 2 * n_dst) {
        return;
    }
    const bool on_q = idx < n_dst;
    const int d = on_q ? idx : idx - n_dst;
    const int s = d % n_src;
    float* row = (on_q ? q_dst : k_dst) + d * dim;
    const float* in = (on_q ? q_src : k_src) + s * dim;
    const float scale = on_q ? q_scale : k_scale;
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float ss = 0.0f;
    for (int i = tid; i < dim; i += nthreads) {
        const float v = in[i];
        row[i] = v;
        ss += v * v;
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss + eps) * scale;
    for (int i = tid; i < dim; i += nthreads) {
        row[i] *= inv;
    }
}

int gemv_grid(int rows) {
    return (rows + kGemvRowsPerWg - 1) / kGemvRowsPerWg;
}

unsigned q8x_shared(int cols) {
    const int n = q8x_lds_bytes(cols);
    if (n <= 0 || n > kLdsQ8xMaxBytes) {
        return 0;
    }
    return static_cast<unsigned>(n);
}

struct ArgMaxScratch {
    float* vals = nullptr;
    int* idxs = nullptr;
    int* out = nullptr;
    int cap = 0;
};

ArgMaxScratch g_argmax;

void argmax_ensure(int nblocks) {
    if (nblocks <= g_argmax.cap) {
        return;
    }
    if (g_argmax.vals != nullptr) {
        hip_free(g_argmax.vals);
        hip_free(g_argmax.idxs);
        hip_free(g_argmax.out);
    }
    g_argmax.vals = static_cast<float*>(hip_alloc(sizeof(float) * static_cast<std::size_t>(nblocks)));
    g_argmax.idxs = static_cast<int*>(hip_alloc(sizeof(int) * static_cast<std::size_t>(nblocks)));
    g_argmax.out = static_cast<int*>(hip_alloc(sizeof(int)));
    g_argmax.cap = nblocks;
}

__device__ void better_max_idx(float& v, int& i, float ov, int oi) {
    if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

__device__ void block_reduce_max_idx(float& v, int& idx, float* sv, int* si) {
    const int tid = static_cast<int>(threadIdx.x);
    sv[tid] = v;
    si[tid] = idx;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride >= kWavefront; stride >>= 1) {
        if (tid < stride) {
            better_max_idx(sv[tid], si[tid], sv[tid + stride], si[tid + stride]);
        }
        __syncthreads();
    }
    if (tid < kWavefront) {
        float wv = sv[tid];
        int wi = si[tid];
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down(wv, off, kWavefront);
            const int oi = __shfl_down(wi, off, kWavefront);
            if (tid + off < kWavefront) {
                better_max_idx(wv, wi, ov, oi);
            }
        }
        if (tid == 0) {
            sv[0] = wv;
            si[0] = wi;
        }
    }
    __syncthreads();
    v = sv[0];
    idx = si[0];
}

__global__ void __launch_bounds__(256)
argmax_partial_kernel(const float* __restrict__ x, int n, float* __restrict__ out_v,
                      int* __restrict__ out_i) {
    const int tid = static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    float best = -INFINITY;
    int bi = 0;
    for (int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid; i < n;
         i += stride) {
        const float v = x[i];
        if (v > best) {
            best = v;
            bi = i;
        }
    }
    __shared__ float sv[kGemvWorkgroup];
    __shared__ int si[kGemvWorkgroup];
    block_reduce_max_idx(best, bi, sv, si);
    if (tid == 0) {
        out_v[blockIdx.x] = best;
        out_i[blockIdx.x] = bi;
    }
}

__global__ void __launch_bounds__(256)
argmax_final_kernel(const float* __restrict__ vals, const int* __restrict__ idxs, int n,
                    int* __restrict__ out) {
    const int tid = static_cast<int>(threadIdx.x);
    float best = -INFINITY;
    int bi = 0;
    for (int i = tid; i < n; i += static_cast<int>(blockDim.x)) {
        better_max_idx(best, bi, vals[i], idxs[i]);
    }
    __shared__ float sv[kGemvWorkgroup];
    __shared__ int si[kGemvWorkgroup];
    block_reduce_max_idx(best, bi, sv, si);
    if (tid == 0) {
        *out = bi;
    }
}

}  // namespace

void gemv(float* y, const float* weight, const float* x, int out_features, int in_features,
          const float* addend) {
    hip_init();
    check(out_features > 0 && in_features > 0, "gemv: empty shape");
    const int block = gemv_block(in_features);
    hipLaunchKernelGGL(gemv_f32_kernel, dim3(out_features), dim3(block), 0, hip_stream(), y, weight, x,
                       out_features, in_features, addend);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_q8(float* y, const std::byte* packed, const float* x, int rows, int cols,
             const float* addend) {
    hip_init();
    check(rows > 0 && cols > 0 && (cols % 32) == 0, "gemv_q8: cols must be a multiple of 32");
    q8x_quantize(x, cols);
    const unsigned char* p = reinterpret_cast<const unsigned char*>(packed);
    hipLaunchKernelGGL(gemv_q8_mmvq_kernel, dim3(gemv_grid(rows)), dim3(kGemvWorkgroup), q8x_shared(cols), hip_stream(), y, p,
                       g_q8x.qs, g_q8x.d, rows, cols, addend);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_q4k(float* y, const std::byte* packed, const float* x, int rows, int cols,
              const float* addend) {
    hip_init();
    check(rows > 0 && cols > 0 && (cols % 256) == 0, "gemv_q4k: cols must be a multiple of 256");
    q8x_quantize(x, cols);
    const unsigned char* p = reinterpret_cast<const unsigned char*>(packed);
    hipLaunchKernelGGL(gemv_q4k_mmvq_kernel, dim3(gemv_grid(rows)), dim3(kGemvWorkgroup), q8x_shared(cols), hip_stream(), y, p,
                       g_q8x.qs, g_q8x.d, rows, cols, addend);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_q5k(float* y, const std::byte* packed, const float* x, int rows, int cols,
              const float* addend) {
    hip_init();
    check(rows > 0 && cols > 0 && (cols % 256) == 0, "gemv_q5k: cols must be a multiple of 256");
    q8x_quantize(x, cols);
    const unsigned char* p = reinterpret_cast<const unsigned char*>(packed);
    hipLaunchKernelGGL(gemv_q5k_mmvq_kernel, dim3(gemv_grid(rows)), dim3(kGemvWorkgroup), q8x_shared(cols), hip_stream(), y, p,
                       g_q8x.qs, g_q8x.d, rows, cols, addend);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_q6k(float* y, const std::byte* packed, const float* x, int rows, int cols,
              const float* addend) {
    hip_init();
    check(rows > 0 && cols > 0 && (cols % 256) == 0, "gemv_q6k: cols must be a multiple of 256");
    q8x_quantize(x, cols);
    const unsigned char* p = reinterpret_cast<const unsigned char*>(packed);
    hipLaunchKernelGGL(gemv_q6k_mmvq_kernel, dim3(gemv_grid(rows)), dim3(kGemvWorkgroup), q8x_shared(cols), hip_stream(), y, p,
                       g_q8x.qs, g_q8x.d, rows, cols, addend);
    VESPER_HIP_CHECK(hipGetLastError());
}

void launch_gemv_multi(float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1,
                       float* y2, const WeightMatrix& w2, float* y3, const WeightMatrix* w3,
                       const float* x) {
    hip_init();
    const int r0 = w0.rows();
    const int r1 = w1.rows();
    const int r2 = w2.rows();
    const int r3 = (w3 != nullptr) ? w3->rows() : 0;
    const int cols = w0.cols();
    const int total = r0 + r1 + r2 + r3;
    check(total > 0 && cols > 0, "gemv_multi empty shape");
    check(w0.kind() == w1.kind() && w1.kind() == w2.kind(), "gemv_multi kind mismatch");
    if (w3 != nullptr) {
        check(w3->kind() == w0.kind(), "gemv_multi kind mismatch");
    }
    switch (w0.kind()) {
        case WeightKind::F32: {
            const float* f3 = (w3 != nullptr) ? w3->f32_data() : nullptr;
            hipLaunchKernelGGL(gemv_f32_multi_kernel, dim3(total), dim3(gemv_block(cols)), 0, hip_stream(), y0,
                               w0.f32_data(), r0, y1, w1.f32_data(), r1, y2, w2.f32_data(), r2, y3,
                               f3, r3, x, cols);
            break;
        }
        case WeightKind::Q8_0:
        case WeightKind::Q4_K:
        case WeightKind::Q5_K:
        case WeightKind::Q6_K: {
            if (w0.kind() == WeightKind::Q8_0) {
                check((cols % 32) == 0, "gemv_multi q8 cols");
            } else {
                check((cols % 256) == 0, "gemv_multi k-quant cols");
            }
            q8x_quantize(x, cols);
            const unsigned char* p0 = reinterpret_cast<const unsigned char*>(w0.packed());
            const unsigned char* p1 = reinterpret_cast<const unsigned char*>(w1.packed());
            const unsigned char* p2 = reinterpret_cast<const unsigned char*>(w2.packed());
            const unsigned char* p3 =
                (w3 != nullptr) ? reinterpret_cast<const unsigned char*>(w3->packed()) : nullptr;
            const dim3 grid(gemv_grid(total));
            const dim3 block(kGemvWorkgroup);
            const unsigned shmem = q8x_shared(cols);
            switch (w0.kind()) {
                case WeightKind::Q8_0:
                    hipLaunchKernelGGL(gemv_q8_mmvq_multi_kernel, grid, block, shmem, hip_stream(), y0, y1, y2,
                                       y3, p0, p1, p2, p3, g_q8x.qs, g_q8x.d, r0, r1, r2, r3, cols);
                    break;
                case WeightKind::Q4_K:
                    hipLaunchKernelGGL(gemv_q4k_mmvq_multi_kernel, grid, block, shmem, hip_stream(), y0, y1, y2,
                                       y3, p0, p1, p2, p3, g_q8x.qs, g_q8x.d, r0, r1, r2, r3, cols);
                    break;
                case WeightKind::Q5_K:
                    hipLaunchKernelGGL(gemv_q5k_mmvq_multi_kernel, grid, block, shmem, hip_stream(), y0, y1, y2,
                                       y3, p0, p1, p2, p3, g_q8x.qs, g_q8x.d, r0, r1, r2, r3, cols);
                    break;
                case WeightKind::Q6_K:
                    hipLaunchKernelGGL(gemv_q6k_mmvq_multi_kernel, grid, block, shmem, hip_stream(), y0, y1, y2,
                                       y3, p0, p1, p2, p3, g_q8x.qs, g_q8x.d, r0, r1, r2, r3, cols);
                    break;
                case WeightKind::F32:
                    throw std::logic_error("gemv_multi: F32 in packed launch");
            }
            break;
        }
    }
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv3(float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1, float* y2,
           const WeightMatrix& w2, const float* x) {
    launch_gemv_multi(y0, w0, y1, w1, y2, w2, nullptr, nullptr, x);
}

void gemv4(float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1, float* y2,
           const WeightMatrix& w2, float* y3, const WeightMatrix& w3, const float* x) {
    launch_gemv_multi(y0, w0, y1, w1, y2, w2, y3, &w3, x);
}

void tile_l2_scale(float* dst, const float* src, int n_dst, int n_src, int dim, float eps,
                   float scale) {
    hip_init();
    check(n_dst > 0 && n_src > 0 && dim > 0, "tile_l2_scale empty shape");
    hipLaunchKernelGGL(tile_l2_scale_kernel, dim3(n_dst), dim3(kGemvWorkgroup), 0, hip_stream(), dst, src,
                       n_dst, n_src, dim, eps, scale);
    VESPER_HIP_CHECK(hipGetLastError());
}

void tile_l2_pair(float* q_dst, const float* q_src, float* k_dst, const float* k_src, int n_dst,
                  int n_src, int dim, float eps, float q_scale, float k_scale) {
    hip_init();
    check(n_dst > 0 && n_src > 0 && dim > 0, "tile_l2_pair empty shape");
    hipLaunchKernelGGL(tile_l2_pair_kernel, dim3(2 * n_dst), dim3(kGemvWorkgroup), 0, hip_stream(), q_dst,
                       q_src, k_dst, k_src, n_dst, n_src, dim, eps, q_scale, k_scale);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
    hip_init();
    check(n > 0, "rmsnorm: empty");
    hipLaunchKernelGGL(rmsnorm_f32_kernel, dim3(1), dim3(kGemvWorkgroup), 0, hip_stream(), out, x, weight,
                       n, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rmsnorm_rows(float* x, const float* weight, int rows, int dim, float eps) {
    hip_init();
    check(rows > 0 && dim > 0, "rmsnorm_rows: empty");
    hipLaunchKernelGGL(rmsnorm_rows_kernel, dim3(rows), dim3(kGemvWorkgroup), 0, hip_stream(), x, weight,
                       rows, dim, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void split_gated_q(float* q, float* gate, const float* q_full, int n_heads, int head_dim) {
    hip_init();
    check(n_heads > 0 && head_dim > 0, "split_gated_q empty shape");
    hipLaunchKernelGGL(split_gated_q_kernel, dim3(n_heads), dim3(kGemvWorkgroup), 0, hip_stream(), q, gate,
                       q_full, n_heads, head_dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

void split_gated_q_norm(float* q, float* gate, const float* q_full, const float* weight, int n_heads,
                        int head_dim, float eps) {
    hip_init();
    check(n_heads > 0 && head_dim > 0, "split_gated_q_norm empty shape");
    hipLaunchKernelGGL(split_gated_q_norm_kernel, dim3(n_heads), dim3(kGemvWorkgroup), 0, hip_stream(), q, gate,
                       q_full, weight, n_heads, head_dim, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rmsnorm_silu_mul(float* y, const float* z, const float* weight, int rows, int dim, float eps) {
    hip_init();
    check(rows > 0 && dim > 0, "rmsnorm_silu_mul empty shape");
    hipLaunchKernelGGL(rmsnorm_silu_mul_kernel, dim3(rows), dim3(kGemvWorkgroup), 0, hip_stream(), y, z, weight,
                       rows, dim, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void tile_heads(float* dst, const float* src, int n_dst, int n_src, int dim) {
    hip_init();
    check(n_dst > 0 && n_src > 0 && dim > 0, "tile_heads empty shape");
    hipLaunchKernelGGL(tile_heads_kernel, dim3(n_dst), dim3(kGemvWorkgroup), 0, hip_stream(), dst, src, n_dst,
                       n_src, dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

void attn_decode(float* out, const float* q, const float* k, const float* v, const float* gate,
                 int seq, int n_q_heads, int n_kv_heads, int head_dim) {
    hip_init();
    check(seq > 0 && n_q_heads > 0 && n_kv_heads > 0 && head_dim > 0, "attn_decode empty shape");
    check(n_q_heads % n_kv_heads == 0, "attn_decode GQA");
    check(head_dim <= kGemvWorkgroup, "attn_decode: head_dim exceeds workgroup");
    hipLaunchKernelGGL(attn_decode_kernel, dim3(n_q_heads), dim3(kGemvWorkgroup), 0, hip_stream(), out, q, k,
                       v, gate, seq, nullptr, n_q_heads, n_kv_heads, head_dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

void attn_decode(float* out, const float* q, const float* k, const float* v, const float* gate,
                 const int* pos, int n_q_heads, int n_kv_heads, int head_dim) {
    hip_init();
    check(pos != nullptr && n_q_heads > 0 && n_kv_heads > 0 && head_dim > 0,
          "attn_decode empty shape");
    check(n_q_heads % n_kv_heads == 0, "attn_decode GQA");
    check(head_dim <= kGemvWorkgroup, "attn_decode: head_dim exceeds workgroup");
    hipLaunchKernelGGL(attn_decode_kernel, dim3(n_q_heads), dim3(kGemvWorkgroup), 0, hip_stream(), out, q, k,
                       v, gate, 0, pos, n_q_heads, n_kv_heads, head_dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

void add_rmsnorm(float* x, float* residual, const float* weight, int n, float eps) {
    hip_init();
    check(n > 0, "add_rmsnorm: empty");
    hipLaunchKernelGGL(add_rmsnorm_kernel, dim3(1), dim3(kGemvWorkgroup), 0, hip_stream(), x, residual, weight,
                       n, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void copy_rmsnorm(float* x, float* residual, const float* weight, int n, float eps) {
    hip_init();
    check(n > 0, "copy_rmsnorm: empty");
    hipLaunchKernelGGL(copy_rmsnorm_kernel, dim3(1), dim3(kGemvWorkgroup), 0, hip_stream(), x, residual, weight,
                       n, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void silu_mul(float* y, const float* z, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(silu_mul_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), y, z, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gdn_gates(float* decay, float* beta, const float* alpha, const float* dt, const float* a,
               int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(gdn_gates_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), decay, beta,
                       alpha, dt, a, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void split_qkv(float* q, float* k, float* v, const float* qkv, int key_dim, int value_dim) {
    hip_init();
    check(key_dim > 0 && value_dim > 0, "split_qkv empty shape");
    const int total = 2 * key_dim + value_dim;
    const int block = 256;
    hipLaunchKernelGGL(split_qkv_kernel, dim3(ceil_div(total, block)), dim3(block), 0, hip_stream(), q, k, v,
                       qkv, key_dim, value_dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_swiglu(float* hidden, const WeightMatrix& gate, const WeightMatrix& up, const float* x) {
    hip_init();
    check(gate.kind() == up.kind(), "gemv_swiglu kind mismatch");
    check(gate.rows() == up.rows() && gate.cols() == up.cols(), "gemv_swiglu shape mismatch");
    const int rows = gate.rows();
    const int cols = gate.cols();
    check(rows > 0 && cols > 0, "gemv_swiglu empty shape");
    switch (gate.kind()) {
        case WeightKind::F32:
            hipLaunchKernelGGL(gemv_swiglu_f32_kernel, dim3(rows), dim3(gemv_block(cols)), 0, hip_stream(),
                               hidden, gate.f32_data(), up.f32_data(), x, rows, cols);
            break;
        case WeightKind::Q8_0: {
            check((cols % 32) == 0, "gemv_swiglu q8: cols must be a multiple of 32");
            q8x_quantize(x, cols);
            hipLaunchKernelGGL(gemv_swiglu_q8_mmvq_kernel, dim3(gemv_grid(rows)),
                               dim3(kGemvWorkgroup), q8x_shared(cols), hip_stream(), hidden,
                               reinterpret_cast<const unsigned char*>(gate.packed()),
                               reinterpret_cast<const unsigned char*>(up.packed()), g_q8x.qs,
                               g_q8x.d, rows, cols);
            break;
        }
        case WeightKind::Q4_K: {
            check((cols % 256) == 0, "gemv_swiglu q4k: cols must be a multiple of 256");
            q8x_quantize(x, cols);
            hipLaunchKernelGGL(gemv_swiglu_q4k_mmvq_kernel, dim3(gemv_grid(rows)),
                               dim3(kGemvWorkgroup), q8x_shared(cols), hip_stream(), hidden,
                               reinterpret_cast<const unsigned char*>(gate.packed()),
                               reinterpret_cast<const unsigned char*>(up.packed()), g_q8x.qs,
                               g_q8x.d, rows, cols);
            break;
        }
        case WeightKind::Q5_K: {
            check((cols % 256) == 0, "gemv_swiglu q5k: cols must be a multiple of 256");
            q8x_quantize(x, cols);
            hipLaunchKernelGGL(gemv_swiglu_q5k_mmvq_kernel, dim3(gemv_grid(rows)),
                               dim3(kGemvWorkgroup), q8x_shared(cols), hip_stream(), hidden,
                               reinterpret_cast<const unsigned char*>(gate.packed()),
                               reinterpret_cast<const unsigned char*>(up.packed()), g_q8x.qs,
                               g_q8x.d, rows, cols);
            break;
        }
        case WeightKind::Q6_K: {
            check((cols % 256) == 0, "gemv_swiglu q6k: cols must be a multiple of 256");
            q8x_quantize(x, cols);
            hipLaunchKernelGGL(gemv_swiglu_q6k_mmvq_kernel, dim3(gemv_grid(rows)),
                               dim3(kGemvWorkgroup), q8x_shared(cols), hip_stream(), hidden,
                               reinterpret_cast<const unsigned char*>(gate.packed()),
                               reinterpret_cast<const unsigned char*>(up.packed()), g_q8x.qs,
                               g_q8x.d, rows, cols);
            break;
        }
    }
    VESPER_HIP_CHECK(hipGetLastError());
}

void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
               int pos, float theta) {
    hip_init();
    check(rotary_dim > 0 && (rotary_dim % 2) == 0 && rotary_dim <= head_dim,
          "rope rotary_dim must be even and <= head_dim");
    const int total = (n_q_heads + n_kv_heads) * (rotary_dim / 2);
    if (total <= 0) {
        return;
    }
    const int block = 64;
    hipLaunchKernelGGL(rope_neox_kernel, dim3(ceil_div(total, block)), dim3(block), 0, hip_stream(), q, k,
                       n_q_heads, n_kv_heads, head_dim, rotary_dim, pos, nullptr, theta);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
               const int* pos, float theta) {
    hip_init();
    check(pos != nullptr, "rope pos");
    check(rotary_dim > 0 && (rotary_dim % 2) == 0 && rotary_dim <= head_dim,
          "rope rotary_dim must be even and <= head_dim");
    const int total = (n_q_heads + n_kv_heads) * (rotary_dim / 2);
    if (total <= 0) {
        return;
    }
    const int block = 64;
    hipLaunchKernelGGL(rope_neox_kernel, dim3(ceil_div(total, block)), dim3(block), 0, hip_stream(), q, k,
                       n_q_heads, n_kv_heads, head_dim, rotary_dim, 0, pos, theta);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rope_neox_k_norm(float* q, float* k, const float* k_weight, int n_q_heads, int n_kv_heads,
                      int head_dim, int rotary_dim, int pos, float theta, float eps) {
    hip_init();
    check(rotary_dim > 0 && (rotary_dim % 2) == 0 && rotary_dim <= head_dim,
          "rope rotary_dim must be even and <= head_dim");
    check(n_q_heads > 0 && n_kv_heads > 0 && head_dim > 0, "rope_neox_k_norm empty shape");
    hipLaunchKernelGGL(rope_neox_k_norm_kernel, dim3(n_q_heads + n_kv_heads), dim3(kGemvWorkgroup),
                       0, hip_stream(), q, k, k_weight, n_q_heads, n_kv_heads, head_dim, rotary_dim,
                       pos, nullptr, theta, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rope_neox_k_norm(float* q, float* k, const float* k_weight, int n_q_heads, int n_kv_heads,
                      int head_dim, int rotary_dim, const int* pos, float theta, float eps) {
    hip_init();
    check(pos != nullptr, "rope pos");
    check(rotary_dim > 0 && (rotary_dim % 2) == 0 && rotary_dim <= head_dim,
          "rope rotary_dim must be even and <= head_dim");
    check(n_q_heads > 0 && n_kv_heads > 0 && head_dim > 0, "rope_neox_k_norm empty shape");
    hipLaunchKernelGGL(rope_neox_k_norm_kernel, dim3(n_q_heads + n_kv_heads), dim3(kGemvWorkgroup),
                       0, hip_stream(), q, k, k_weight, n_q_heads, n_kv_heads, head_dim, rotary_dim,
                       0, pos, theta, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void swiglu(float* out, const float* gate, const float* up, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(swiglu_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), out, gate,
                       up, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void softmax_inplace(float* x, int n) {
    hip_init();
    check(n > 0, "softmax: empty");
    hipLaunchKernelGGL(softmax_kernel, dim3(1), dim3(kGemvWorkgroup), 0, hip_stream(), x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

__global__ void embed_f32_kernel(float* __restrict__ out, const float* __restrict__ table,
                                const int* __restrict__ token, int hidden) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= hidden) {
        return;
    }
    out[i] = table[static_cast<std::size_t>(*token) * static_cast<std::size_t>(hidden) +
                   static_cast<std::size_t>(i)];
}

__global__ void scatter_row_kernel(float* __restrict__ base, const float* __restrict__ row,
                                  const int* __restrict__ pos, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    base[static_cast<std::size_t>(*pos) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)] =
        row[i];
}

__global__ void embed_q8_kernel(float* __restrict__ out, const unsigned char* __restrict__ packed,
                               int token_val, const int* token_ptr, int cols) {
    const int token = decode_i32(token_val, token_ptr);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= cols) {
        return;
    }
    const int nblocks = cols / kQ8BlockElems;
    const int b = i / kQ8BlockElems;
    const int e = i % kQ8BlockElems;
    const unsigned char* blk =
        packed + (static_cast<std::size_t>(token) * nblocks + b) * kQ8BlockBytes;
    const std::uint16_t dbits =
        static_cast<std::uint16_t>(blk[0]) | static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[1]) << 8);
    out[i] = f16_to_f32_dev(dbits) * static_cast<float>(static_cast<std::int8_t>(blk[2 + e]));
}

__global__ void embed_q4k_kernel(float* __restrict__ out, const unsigned char* __restrict__ packed,
                                int token_val, const int* token_ptr, int cols) {
    const int token = decode_i32(token_val, token_ptr);
    const int supers = cols / kQ4KBlockElems;
    const int s = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (s >= supers) {
        return;
    }
    const unsigned char* blk =
        packed + (static_cast<std::size_t>(token) * supers + s) * kQ4KBlockBytes;
    const std::uint16_t dbits =
        static_cast<std::uint16_t>(blk[0]) | static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[1]) << 8);
    const std::uint16_t dminbits =
        static_cast<std::uint16_t>(blk[2]) | static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[3]) << 8);
    int scales[8];
    int mins[8];
    q4k_decode_scales(blk + 4, scales, mins);
    const int group = tid / 64;
    const int r = tid % 32;
    const int hi = (tid / 32) & 1;
    const int is = 2 * group + hi;
    const unsigned char byte = blk[16 + group * 32 + r];
    const int q = hi ? (byte >> 4) : (byte & 0x0f);
    out[s * kQ4KBlockElems + tid] =
        f16_to_f32_dev(dbits) * static_cast<float>(scales[is]) * static_cast<float>(q) -
        f16_to_f32_dev(dminbits) * static_cast<float>(mins[is]);
}

__global__ void embed_q5k_kernel(float* __restrict__ out, const unsigned char* __restrict__ packed,
                                int token_val, const int* token_ptr, int cols) {
    const int token = decode_i32(token_val, token_ptr);
    const int supers = cols / kQ5KBlockElems;
    const int s = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (s >= supers) {
        return;
    }
    const unsigned char* blk =
        packed + (static_cast<std::size_t>(token) * supers + s) * kQ5KBlockBytes;
    const std::uint16_t dbits =
        static_cast<std::uint16_t>(blk[0]) | static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[1]) << 8);
    const std::uint16_t dminbits =
        static_cast<std::uint16_t>(blk[2]) | static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[3]) << 8);
    int scales[8];
    int mins[8];
    q4k_decode_scales(blk + 4, scales, mins);
    const int il = tid / 64;
    const int r = tid % 32;
    const int hi = (tid / 32) & 1;
    const int is = 2 * il + hi;
    const unsigned char hm = static_cast<unsigned char>(1u << (2 * il + hi));
    const unsigned char byte = blk[48 + il * 32 + r];
    int q = hi ? (byte >> 4) : (byte & 0x0f);
    if (blk[16 + r] & hm) {
        q += 16;
    }
    out[s * kQ5KBlockElems + tid] =
        f16_to_f32_dev(dbits) * static_cast<float>(scales[is]) * static_cast<float>(q) -
        f16_to_f32_dev(dminbits) * static_cast<float>(mins[is]);
}

__global__ void embed_q6k_kernel(float* __restrict__ out, const unsigned char* __restrict__ packed,
                                int token_val, const int* token_ptr, int cols) {
    const int token = decode_i32(token_val, token_ptr);
    const int supers = cols / kQ6KBlockElems;
    const int s = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (s >= supers) {
        return;
    }
    const unsigned char* blk =
        packed + (static_cast<std::size_t>(token) * supers + s) * kQ6KBlockBytes;
    const std::uint16_t dbits =
        static_cast<std::uint16_t>(blk[208]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[209]) << 8);
    const int ip = tid / 128;
    const int off = tid % 128;
    const int il = off % 32;
    const int slot = off / 32;
    const int is = 8 * ip + il / 16;
    const unsigned char ql0 = blk[64 * ip + il];
    const unsigned char ql1 = blk[64 * ip + 32 + il];
    const unsigned char qh = blk[128 + 32 * ip + il];
    int q = 0;
    int sc = 0;
    if (slot == 0) {
        q = (ql0 & 0x0f) | static_cast<int>((qh >> 0) & 3) << 4;
        sc = static_cast<signed char>(blk[192 + is + 0]);
    } else if (slot == 1) {
        q = (ql1 & 0x0f) | static_cast<int>((qh >> 2) & 3) << 4;
        sc = static_cast<signed char>(blk[192 + is + 2]);
    } else if (slot == 2) {
        q = (ql0 >> 4) | static_cast<int>((qh >> 4) & 3) << 4;
        sc = static_cast<signed char>(blk[192 + is + 4]);
    } else {
        q = (ql1 >> 4) | static_cast<int>((qh >> 6) & 3) << 4;
        sc = static_cast<signed char>(blk[192 + is + 6]);
    }
    out[s * kQ6KBlockElems + tid] =
        f16_to_f32_dev(dbits) * static_cast<float>(sc) * static_cast<float>(q - 32);
}

void embed_row(float* out, const float* table, int token, int hidden) {
    hip_init();
    hip_copy_d2d(out, table + static_cast<std::size_t>(token) * hidden,
                 static_cast<std::size_t>(hidden) * sizeof(float));
}

void embed_row(float* out, const float* table, const int* token, int hidden) {
    hip_init();
    check(token != nullptr && hidden > 0, "embed empty");
    const int block = 256;
    hipLaunchKernelGGL(embed_f32_kernel, dim3(ceil_div(hidden, block)), dim3(block), 0, hip_stream(), out,
                       table, token, hidden);
    VESPER_HIP_CHECK(hipGetLastError());
}

void scatter_row(float* base, const float* row, const int* pos, int n) {
    hip_init();
    check(base != nullptr && row != nullptr && pos != nullptr, "scatter_row null");
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(scatter_row_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), base, row,
                       pos, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

namespace {

void embed_packed(float* out, const WeightMatrix& table, int token_val, const int* token_ptr) {
    const int cols = table.cols();
    switch (table.kind()) {
        case WeightKind::F32:
            if (token_ptr != nullptr) {
                embed_row(out, table.f32_data(), token_ptr, cols);
            } else {
                embed_row(out, table.f32_data(), token_val, cols);
            }
            return;
        case WeightKind::Q8_0:
            check(cols > 0 && (cols % kQ8BlockElems) == 0, "embed Q8_0 cols");
            hipLaunchKernelGGL(embed_q8_kernel, dim3(ceil_div(cols, kGemvWorkgroup)),
                               dim3(kGemvWorkgroup), 0, hip_stream(), out,
                               reinterpret_cast<const unsigned char*>(table.packed()), token_val,
                               token_ptr, cols);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        case WeightKind::Q4_K:
            check(cols > 0 && (cols % kQ4KBlockElems) == 0, "embed Q4_K cols");
            hipLaunchKernelGGL(embed_q4k_kernel, dim3(cols / kQ4KBlockElems), dim3(kQ4KBlockElems),
                               0, hip_stream(), out, reinterpret_cast<const unsigned char*>(table.packed()),
                               token_val, token_ptr, cols);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        case WeightKind::Q5_K:
            check(cols > 0 && (cols % kQ5KBlockElems) == 0, "embed Q5_K cols");
            hipLaunchKernelGGL(embed_q5k_kernel, dim3(cols / kQ5KBlockElems), dim3(kQ5KBlockElems),
                               0, hip_stream(), out, reinterpret_cast<const unsigned char*>(table.packed()),
                               token_val, token_ptr, cols);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        case WeightKind::Q6_K:
            check(cols > 0 && (cols % kQ6KBlockElems) == 0, "embed Q6_K cols");
            hipLaunchKernelGGL(embed_q6k_kernel, dim3(cols / kQ6KBlockElems), dim3(kQ6KBlockElems),
                               0, hip_stream(), out, reinterpret_cast<const unsigned char*>(table.packed()),
                               token_val, token_ptr, cols);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
    }
    throw std::logic_error("unhandled WeightKind");
}

}  // namespace

void embed_row(float* out, const WeightMatrix& table, int token) {
    hip_init();
    check(token >= 0 && token < table.rows(), "embed token out of range");
    embed_packed(out, table, token, nullptr);
}

void embed_row(float* out, const WeightMatrix& table, const int* token) {
    hip_init();
    check(token != nullptr, "embed token");
    embed_packed(out, table, 0, token);
}

void attn_scores(float* scores, const float* q, const float* k, int seq, int n_kv_heads,
                 int kv_head, int head_dim) {
    hip_init();
    if (seq <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(attn_scores_kernel, dim3(ceil_div(seq, block)), dim3(block), 0, hip_stream(),
                       scores, q, k, seq, n_kv_heads, kv_head, head_dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

void attn_mix(float* out, const float* scores, const float* v, int seq, int n_kv_heads,
              int kv_head, int head_dim) {
    hip_init();
    check(head_dim <= kGemvWorkgroup, "attn_mix: head_dim exceeds workgroup");
    hipLaunchKernelGGL(attn_mix_kernel, dim3(1), dim3(head_dim), 0, hip_stream(), out, scores, v, seq,
                       n_kv_heads, kv_head, head_dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

void add_inplace(float* dst, const float* src, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(add_inplace_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), dst,
                       src, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void copy(float* dst, const float* src, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    hip_copy_d2d(dst, src, static_cast<std::size_t>(n) * sizeof(float));
}

void sigmoid_inplace(float* x, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(sigmoid_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void silu_inplace(float* x, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(silu_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void softplus_inplace(float* x, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(softplus_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void exp_inplace(float* x, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(exp_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void mul_inplace(float* dst, const float* src, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(mul_inplace_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), dst, src,
                       n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void scale_inplace(float* x, float scale, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(scale_inplace_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), x,
                       scale, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void l2_normalize_rows(float* x, int rows, int dim, float eps) {
    hip_init();
    if (rows <= 0 || dim <= 0) {
        return;
    }
    hipLaunchKernelGGL(l2_normalize_rows_kernel, dim3(rows), dim3(kWavefront), 0, hip_stream(), x, rows,
                       dim, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gdn_conv_update(float* y, float* state, const float* x, const float* weight, int conv_dim,
                     int kernel) {
    hip_init();
    check(kernel >= 2 && conv_dim > 0, "gdn conv shape");
    const int block = 256;
    hipLaunchKernelGGL(gdn_conv_update_kernel, dim3(ceil_div(conv_dim, block)), dim3(block), 0, hip_stream(),
                       y, state, x, weight, conv_dim, kernel);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gdn_conv_split(float* q, float* k, float* v, float* state, const float* x, const float* weight,
                    int key_dim, int value_dim, int kernel) {
    hip_init();
    check(kernel >= 2 && key_dim > 0 && value_dim > 0, "gdn conv split shape");
    const int qkv_dim = 2 * key_dim + value_dim;
    const int block = 256;
    hipLaunchKernelGGL(gdn_conv_split_kernel, dim3(ceil_div(qkv_dim, block)), dim3(block), 0, hip_stream(), q,
                       k, v, state, x, weight, key_dim, value_dim, kernel);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gdn_delta_rule(float* y, float* rec, const float* q, const float* k, const float* v,
                    const float* decay, const float* beta, int n_heads, int dim) {
    hip_init();
    check(n_heads > 0 && dim > 0 && dim <= kGdnDeltaMaxDim, "gdn delta shape");
    const int col_blocks = (dim + kGdnDeltaWarps - 1) / kGdnDeltaWarps;
    hipLaunchKernelGGL(gdn_delta_rule_kernel, dim3(n_heads, col_blocks),
                       dim3(kWavefront, kGdnDeltaWarps), 0, hip_stream(), y, rec, q, k, v, decay, beta,
                       n_heads, dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

int argmax(const float* x, int n) {
    hip_init();
    check(n > 0, "argmax: empty");
    const int block = kGemvWorkgroup;
    int nblocks = ceil_div(n, block);
    if (nblocks > 1024) {
        nblocks = 1024;
    }
    argmax_ensure(nblocks);
    hipLaunchKernelGGL(argmax_partial_kernel, dim3(nblocks), dim3(block), 0, hip_stream(), x, n, g_argmax.vals,
                       g_argmax.idxs);
    VESPER_HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(argmax_final_kernel, dim3(1), dim3(block), 0, hip_stream(), g_argmax.vals, g_argmax.idxs,
                       nblocks, g_argmax.out);
    VESPER_HIP_CHECK(hipGetLastError());
    int best = 0;
    hip_copy_d2h(&best, g_argmax.out, sizeof(int));
    return best;
}

}  // namespace rdna4
}  // namespace vesper

