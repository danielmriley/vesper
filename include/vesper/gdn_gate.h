#pragma once

#include <cmath>

namespace vesper {

// Official conv_tile_gates issues these with the v column, then ALU
// after conv. Same sum as the old load-inside apply.
struct GdnGateIn {
    float alpha;
    float dt;
    float a;
    float beta;
};

inline float gdn_gate_exp(float x) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __expf(x);
#else
    return std::exp(x);
#endif
}

inline GdnGateIn gdn_gate_load(const float* alpha, const float* dt, const float* a,
                               const float* beta, int i) {
    return {alpha[i], dt[i], a[i], beta[i]};
}

inline void gdn_gate_apply(float* decay, float* beta, int i, GdnGateIn in) {
    float t = in.alpha + in.dt;
    if (t > 20.0f) {
        // softplus(t) ~= t
    } else if (t < -20.0f) {
        t = gdn_gate_exp(t);
    } else {
#if defined(__HIP_DEVICE_COMPILE__)
        t = log1pf(gdn_gate_exp(t));
#else
        t = std::log1p(gdn_gate_exp(t));
#endif
    }
    decay[i] = gdn_gate_exp(in.a * t);
    const float b = in.beta;
    if (b >= 0.0f) {
        beta[i] = 1.0f / (1.0f + gdn_gate_exp(-b));
    } else {
        const float z = gdn_gate_exp(b);
        beta[i] = z / (1.0f + z);
    }
}

}  // namespace vesper
