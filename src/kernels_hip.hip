#include "vesper/rdna4.h"

#include "vesper/hip.h"
#include "vesper/target.h"
#include "vesper/types.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
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
                                 int n_kv_heads, int head_dim, int pos, float theta) {
    const int half = head_dim / 2;
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
        1.0f / powf(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
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

}  // namespace

void gemv(float* y, const float* weight, const float* x, int out_features, int in_features) {
    hip_init();
    check(out_features > 0 && in_features > 0, "gemv: empty shape");
    const int block = gemv_block(in_features);
    hipLaunchKernelGGL(gemv_f32_kernel, dim3(out_features), dim3(block), 0, 0, y, weight, x,
                       out_features, in_features);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
    hip_init();
    check(n > 0, "rmsnorm: empty");
    hipLaunchKernelGGL(rmsnorm_f32_kernel, dim3(1), dim3(kGemvWorkgroup), 0, 0, out, x, weight,
                       n, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim, int pos,
               float theta) {
    hip_init();
    const int total = (n_q_heads + n_kv_heads) * (head_dim / 2);
    if (total <= 0) {
        return;
    }
    const int block = 64;
    hipLaunchKernelGGL(rope_neox_kernel, dim3(ceil_div(total, block)), dim3(block), 0, 0, q, k,
                       n_q_heads, n_kv_heads, head_dim, pos, theta);
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

}  // namespace rdna4
}  // namespace vesper
