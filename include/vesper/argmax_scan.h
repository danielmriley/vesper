#pragma once

#if defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime.h>
#endif

namespace vesper {

// First max wins. The oi < i clause is for the cross-thread fold:
// two lanes can hold the same value at different indices.
inline void argmax_better(float& v, int& i, float ov, int oi) {
    if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

// Official lm_head last WG is 96 threads over 248320 logits. Same
// first-max-wins as the scalar walk. Tail covers n % 4.
inline void argmax_scan4(const float* y, int n, int tid, int nthreads, float& best, int& bi) {
    const int n4 = n >> 2;
    for (int j = tid; j < n4; j += nthreads) {
#if defined(__HIP_DEVICE_COMPILE__)
        const float4 v = reinterpret_cast<const float4*>(y)[j];
        const int i = j << 2;
        argmax_better(best, bi, v.x, i);
        argmax_better(best, bi, v.y, i + 1);
        argmax_better(best, bi, v.z, i + 2);
        argmax_better(best, bi, v.w, i + 3);
#else
        const int i = j << 2;
        argmax_better(best, bi, y[i], i);
        argmax_better(best, bi, y[i + 1], i + 1);
        argmax_better(best, bi, y[i + 2], i + 2);
        argmax_better(best, bi, y[i + 3], i + 3);
#endif
    }
    for (int i = (n & ~3) + tid; i < n; i += nthreads) {
        argmax_better(best, bi, y[i], i);
    }
}

}  // namespace vesper
