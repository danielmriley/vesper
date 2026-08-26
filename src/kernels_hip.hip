#include "vesper/rdna4.h"

#include "vesper/hip.h"
#include "vesper/target.h"
#include "vesper/types.h"

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

// One wave32 workgroup per output row. GGUF Q8_0 stays packed at 34 B/block.
__global__ void __launch_bounds__(32)
gemv_q8_kernel(float* __restrict__ y, const unsigned char* __restrict__ packed,
               const float* __restrict__ x, int rows, int cols) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int nblocks = cols / 32;
    const unsigned char* rowp =
        packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(nblocks) * 34u;
    float acc = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        const unsigned char* blk = rowp + b * 34;
        const std::uint16_t dbits =
            static_cast<std::uint16_t>(blk[0]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[1]) << 8);
        const float d = f16_to_f32_dev(dbits);
        const float local =
            static_cast<float>(static_cast<std::int8_t>(blk[2 + tid])) * x[b * 32 + tid];
        const float sum = wave32_sum(local);
        if (tid == 0) {
            acc += d * sum;
        }
    }
    if (tid == 0) {
        y[row] = acc;
    }
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
                const float* __restrict__ x, int out_features, int in_features) {
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
        y[row] = sum;
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
                                 int n_kv_heads, int head_dim, int rotary_dim, int pos,
                                 float theta) {
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

__global__ void __launch_bounds__(32)
gemv_q4k_kernel(float* __restrict__ y, const unsigned char* __restrict__ packed,
                const float* __restrict__ x, int rows, int cols) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int supers = cols / 256;
    const unsigned char* rowp =
        packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 144u;
    float acc = 0.0f;
    for (int s = 0; s < supers; ++s) {
        const unsigned char* blk = rowp + s * 144;
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
        const unsigned char* qs = blk + 16;
        const float* xb = x + s * 256;
        for (int g = 0; g < 4; ++g) {
            const int lo = 2 * g;
            const int hi = 2 * g + 1;
            const float sc_lo = d * static_cast<float>(scales[lo]);
            const float sc_hi = d * static_cast<float>(scales[hi]);
            const float mn_lo = dmin * static_cast<float>(mins[lo]);
            const float mn_hi = dmin * static_cast<float>(mins[hi]);
            const unsigned char byte = qs[g * 32 + tid];
            acc += (sc_lo * static_cast<float>(byte & 0x0f) - mn_lo) * xb[lo * 32 + tid];
            acc += (sc_hi * static_cast<float>(byte >> 4) - mn_hi) * xb[hi * 32 + tid];
        }
    }
    acc = wave32_sum(acc);
    if (tid == 0) {
        y[row] = acc;
    }
}

__global__ void __launch_bounds__(32)
gemv_q5k_kernel(float* __restrict__ y, const unsigned char* __restrict__ packed,
                const float* __restrict__ x, int rows, int cols) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int supers = cols / 256;
    const unsigned char* rowp =
        packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 176u;
    float acc = 0.0f;
    for (int s = 0; s < supers; ++s) {
        const unsigned char* blk = rowp + s * 176;
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
        const float* xb = x + s * 256;
        for (int il = 0; il < 4; ++il) {
            const int is = 2 * il;
            const float d1 = d * static_cast<float>(scales[is]);
            const float m1 = dmin * static_cast<float>(mins[is]);
            const float d2 = d * static_cast<float>(scales[is + 1]);
            const float m2 = dmin * static_cast<float>(mins[is + 1]);
            unsigned char hm = static_cast<unsigned char>(1u << (2 * il));
            const unsigned char byte = qs[32 * il + tid];
            const int qlo = (byte & 0x0f) + ((qh[tid] & hm) ? 16 : 0);
            hm = static_cast<unsigned char>(hm << 1);
            const int qhi = (byte >> 4) + ((qh[tid] & hm) ? 16 : 0);
            acc += (d1 * static_cast<float>(qlo) - m1) * xb[64 * il + tid];
            acc += (d2 * static_cast<float>(qhi) - m2) * xb[64 * il + 32 + tid];
        }
    }
    acc = wave32_sum(acc);
    if (tid == 0) {
        y[row] = acc;
    }
}

__global__ void __launch_bounds__(32)
gemv_q6k_kernel(float* __restrict__ y, const unsigned char* __restrict__ packed,
                const float* __restrict__ x, int rows, int cols) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int supers = cols / 256;
    const unsigned char* rowp =
        packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 210u;
    float acc = 0.0f;
    for (int s = 0; s < supers; ++s) {
        const unsigned char* blk = rowp + s * 210;
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const signed char* scales = reinterpret_cast<const signed char*>(blk + 192);
        const std::uint16_t dbits =
            static_cast<std::uint16_t>(blk[208]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(blk[209]) << 8);
        const float d = f16_to_f32_dev(dbits);
        const float* xb = x + s * 256;
        for (int ip = 0; ip < 2; ++ip) {
            const int is = 8 * ip + tid / 16;
            const unsigned char ql0 = ql[64 * ip + tid];
            const unsigned char ql1 = ql[64 * ip + 32 + tid];
            const unsigned char qhv = qh[32 * ip + tid];
            const int q0 = static_cast<int>((ql0 & 0x0f) | (((qhv >> 0) & 3) << 4)) - 32;
            const int q1 = static_cast<int>((ql1 & 0x0f) | (((qhv >> 2) & 3) << 4)) - 32;
            const int q2 = static_cast<int>((ql0 >> 4) | (((qhv >> 4) & 3) << 4)) - 32;
            const int q3 = static_cast<int>((ql1 >> 4) | (((qhv >> 6) & 3) << 4)) - 32;
            const float* xb0 = xb + 128 * ip + tid;
            acc += d * static_cast<float>(scales[is + 0]) * static_cast<float>(q0) * xb0[0];
            acc += d * static_cast<float>(scales[is + 2]) * static_cast<float>(q1) * xb0[32];
            acc += d * static_cast<float>(scales[is + 4]) * static_cast<float>(q2) * xb0[64];
            acc += d * static_cast<float>(scales[is + 6]) * static_cast<float>(q3) * xb0[96];
        }
    }
    acc = wave32_sum(acc);
    if (tid == 0) {
        y[row] = acc;
    }
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

__global__ void gdn_delta_rule_kernel(float* __restrict__ y, float* __restrict__ rec,
                                     const float* __restrict__ q, const float* __restrict__ k,
                                     const float* __restrict__ v, const float* __restrict__ decay,
                                     const float* __restrict__ beta, int n_heads, int dim) {
    const int h = static_cast<int>(blockIdx.x);
    const int j = static_cast<int>(threadIdx.x);
    if (h >= n_heads || j >= dim) {
        return;
    }
    float* S = rec + static_cast<std::size_t>(h) * dim * dim;
    const float g = decay[h];
    const float b = beta[h];
    for (int i = 0; i < dim; ++i) {
        S[i * dim + j] *= g;
    }
    float retrieved = 0.0f;
    for (int i = 0; i < dim; ++i) {
        retrieved += S[i * dim + j] * k[h * dim + i];
    }
    const float delta = b * (v[h * dim + j] - retrieved);
    for (int i = 0; i < dim; ++i) {
        S[i * dim + j] += k[h * dim + i] * delta;
    }
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) {
        acc += S[i * dim + j] * q[h * dim + i];
    }
    y[h * dim + j] = acc;
}

}  // namespace

void gemv(float* y, const float* weight, const float* x, int out_features, int in_features) {
    hip_init();
    check(out_features > 0 && in_features > 0, "gemv: empty shape");
    const int block = gemv_block(in_features);
    hipLaunchKernelGGL(gemv_f32_kernel, dim3(out_features), dim3(block), 0, 0, y, weight, x,
                       out_features, in_features);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_q8(float* y, const std::byte* packed, const float* x, int rows, int cols) {
    hip_init();
    check(rows > 0 && cols > 0 && (cols % 32) == 0, "gemv_q8: cols must be a multiple of 32");
    hipLaunchKernelGGL(gemv_q8_kernel, dim3(rows), dim3(kWavefront), 0, 0, y,
                       reinterpret_cast<const unsigned char*>(packed), x, rows, cols);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_q4k(float* y, const std::byte* packed, const float* x, int rows, int cols) {
    hip_init();
    check(rows > 0 && cols > 0 && (cols % 256) == 0, "gemv_q4k: cols must be a multiple of 256");
    hipLaunchKernelGGL(gemv_q4k_kernel, dim3(rows), dim3(kWavefront), 0, 0, y,
                       reinterpret_cast<const unsigned char*>(packed), x, rows, cols);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_q5k(float* y, const std::byte* packed, const float* x, int rows, int cols) {
    hip_init();
    check(rows > 0 && cols > 0 && (cols % 256) == 0, "gemv_q5k: cols must be a multiple of 256");
    hipLaunchKernelGGL(gemv_q5k_kernel, dim3(rows), dim3(kWavefront), 0, 0, y,
                       reinterpret_cast<const unsigned char*>(packed), x, rows, cols);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_q6k(float* y, const std::byte* packed, const float* x, int rows, int cols) {
    hip_init();
    check(rows > 0 && cols > 0 && (cols % 256) == 0, "gemv_q6k: cols must be a multiple of 256");
    hipLaunchKernelGGL(gemv_q6k_kernel, dim3(rows), dim3(kWavefront), 0, 0, y,
                       reinterpret_cast<const unsigned char*>(packed), x, rows, cols);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
    hip_init();
    check(n > 0, "rmsnorm: empty");
    hipLaunchKernelGGL(rmsnorm_f32_kernel, dim3(1), dim3(kGemvWorkgroup), 0, 0, out, x, weight,
                       n, eps);
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
    hipLaunchKernelGGL(rope_neox_kernel, dim3(ceil_div(total, block)), dim3(block), 0, 0, q, k,
                       n_q_heads, n_kv_heads, head_dim, rotary_dim, pos, theta);
    VESPER_HIP_CHECK(hipGetLastError());
}

void swiglu(float* out, const float* gate, const float* up, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(swiglu_kernel, dim3(ceil_div(n, block)), dim3(block), 0, 0, out, gate,
                       up, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void softmax_inplace(float* x, int n) {
    hip_init();
    check(n > 0, "softmax: empty");
    hipLaunchKernelGGL(softmax_kernel, dim3(1), dim3(kGemvWorkgroup), 0, 0, x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void embed_row(float* out, const float* table, int token, int hidden) {
    hip_init();
    hip_copy_d2d(out, table + static_cast<std::size_t>(token) * hidden,
                 static_cast<std::size_t>(hidden) * sizeof(float));
}

void attn_scores(float* scores, const float* q, const float* k, int seq, int n_kv_heads,
                 int kv_head, int head_dim) {
    hip_init();
    if (seq <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(attn_scores_kernel, dim3(ceil_div(seq, block)), dim3(block), 0, 0,
                       scores, q, k, seq, n_kv_heads, kv_head, head_dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

void attn_mix(float* out, const float* scores, const float* v, int seq, int n_kv_heads,
              int kv_head, int head_dim) {
    hip_init();
    check(head_dim <= kGemvWorkgroup, "attn_mix: head_dim exceeds workgroup");
    hipLaunchKernelGGL(attn_mix_kernel, dim3(1), dim3(head_dim), 0, 0, out, scores, v, seq,
                       n_kv_heads, kv_head, head_dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

void add_inplace(float* dst, const float* src, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(add_inplace_kernel, dim3(ceil_div(n, block)), dim3(block), 0, 0, dst,
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
    hipLaunchKernelGGL(sigmoid_kernel, dim3(ceil_div(n, block)), dim3(block), 0, 0, x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void silu_inplace(float* x, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(silu_kernel, dim3(ceil_div(n, block)), dim3(block), 0, 0, x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void softplus_inplace(float* x, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(softplus_kernel, dim3(ceil_div(n, block)), dim3(block), 0, 0, x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void exp_inplace(float* x, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(exp_kernel, dim3(ceil_div(n, block)), dim3(block), 0, 0, x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void mul_inplace(float* dst, const float* src, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(mul_inplace_kernel, dim3(ceil_div(n, block)), dim3(block), 0, 0, dst, src,
                       n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void scale_inplace(float* x, float scale, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(scale_inplace_kernel, dim3(ceil_div(n, block)), dim3(block), 0, 0, x,
                       scale, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void l2_normalize_rows(float* x, int rows, int dim, float eps) {
    hip_init();
    if (rows <= 0 || dim <= 0) {
        return;
    }
    hipLaunchKernelGGL(l2_normalize_rows_kernel, dim3(rows), dim3(kWavefront), 0, 0, x, rows,
                       dim, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gdn_conv_update(float* y, float* state, const float* x, const float* weight, int conv_dim,
                     int kernel) {
    hip_init();
    check(kernel >= 2 && conv_dim > 0, "gdn conv shape");
    const int block = 256;
    hipLaunchKernelGGL(gdn_conv_update_kernel, dim3(ceil_div(conv_dim, block)), dim3(block), 0, 0,
                       y, state, x, weight, conv_dim, kernel);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gdn_delta_rule(float* y, float* rec, const float* q, const float* k, const float* v,
                    const float* decay, const float* beta, int n_heads, int dim) {
    hip_init();
    check(n_heads > 0 && dim > 0 && dim <= 256, "gdn delta shape");
    hipLaunchKernelGGL(gdn_delta_rule_kernel, dim3(n_heads), dim3(dim), 0, 0, y, rec, q, k, v,
                       decay, beta, n_heads, dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

}  // namespace rdna4
}  // namespace vesper

