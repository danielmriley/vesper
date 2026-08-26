#pragma once

namespace vesper {

// Official last-WG rms on 5120. 256-thread kernels use hidden_float4_*
// (exact 5 trips). Down is 160 and o_proj/ssm_out are 96, so the walk
// is nthreads over N/4 slots. Same sum as the scalar leftover. n is a
// multiple of 4. Do not keep the tiles across the reduce.

inline void hidden_rms_ss4(const float* x, int n, int tid, int nthreads, float& ss) {
    const int n4 = n >> 2;
    for (int j = tid; j < n4; j += nthreads) {
#if defined(__HIP_DEVICE_COMPILE__)
        const float4 xv = reinterpret_cast<const float4*>(x)[j];
        ss += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
#else
        const int i = j << 2;
        ss += x[i] * x[i] + x[i + 1] * x[i + 1] + x[i + 2] * x[i + 2] + x[i + 3] * x[i + 3];
#endif
    }
}

inline void hidden_rms_store4(float* out, const float* src, const float* weight, float inv, int n,
                              int tid, int nthreads) {
    const int n4 = n >> 2;
    for (int j = tid; j < n4; j += nthreads) {
#if defined(__HIP_DEVICE_COMPILE__)
        const float4 sv = reinterpret_cast<const float4*>(src)[j];
        const float4 wv = reinterpret_cast<const float4*>(weight)[j];
        float4 y;
        y.x = sv.x * inv * wv.x;
        y.y = sv.y * inv * wv.y;
        y.z = sv.z * inv * wv.z;
        y.w = sv.w * inv * wv.w;
        reinterpret_cast<float4*>(out)[j] = y;
#else
        const int i = j << 2;
        out[i] = src[i] * inv * weight[i];
        out[i + 1] = src[i + 1] * inv * weight[i + 1];
        out[i + 2] = src[i + 2] * inv * weight[i + 2];
        out[i + 3] = src[i + 3] * inv * weight[i + 3];
#endif
    }
}

}  // namespace vesper
