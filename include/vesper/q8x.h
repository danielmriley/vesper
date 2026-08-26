#pragma once

#include <cstdint>

namespace vesper {

inline constexpr int kQ8XBlockElems = 32;

// Q8_1 qs plus per-block scales. llama.cpp MMVQ reads this from L2.
inline int q8x_lds_bytes(int cols) {
    return ((cols + 3) & ~3) + (cols / kQ8XBlockElems) * static_cast<int>(sizeof(float));
}

// Quantize a length-n vector (n % 32 == 0) the way llama.cpp does for MMVQ:
// per 32-wide block, d = amax/127 and qs[i] = round(x[i]/d) in [-127, 127].
// CPU stores sum[b] = d * sum(qs) for the Q4_K/Q5_K min term. HIP MMVQ
// recomputes that term with dp4a(0x01010101) and does not read the sum.
void quantize_q8x(const float* x, std::int8_t* qs, float* d, float* sum, int n);
void dequant_q8x(float* x, const std::int8_t* qs, const float* d, int n);

// HIP decode can write Q8_1 inside the producer of x, then skip the next
// packed GEMV's quantize launch. Skip is one-shot on column count. Do not
// key this on the x pointer: scratch.x is reused across layers. Official
// SwiGLU writes the 17408-wide down x into an alt buffer so the kernel
// can keep reading the 5120-wide input from the primary.
inline bool q8x_can_fuse(int n) {
    return n > 0 && (n % kQ8XBlockElems) == 0;
}

// RDNA wave32. Official hidden 5120 / 256 threads is 20 blocks per warp.
// Official GDN 128 / 128 threads is one block per warp.
inline constexpr int q8x_warp_trips(int n_elems, int nthreads) {
    const int nwarps = nthreads / 32;
    const int nblocks = n_elems / kQ8XBlockElems;
    if (nwarps <= 0 || nblocks <= 0 || (nblocks % nwarps) != 0) {
        return 0;
    }
    return nblocks / nwarps;
}

inline void q8x_clear_ready(int* ready_cols) {
    if (ready_cols != nullptr) {
        *ready_cols = 0;
    }
}

inline void q8x_mark_ready(int* ready_cols, int cols) {
    if (ready_cols == nullptr) {
        return;
    }
    *ready_cols = q8x_can_fuse(cols) ? cols : 0;
}

inline bool q8x_take_ready(int* ready_cols, int cols) {
    if (ready_cols == nullptr || !q8x_can_fuse(cols)) {
        q8x_clear_ready(ready_cols);
        return false;
    }
    if (*ready_cols == cols) {
        *ready_cols = 0;
        return true;
    }
    *ready_cols = 0;
    return false;
}

}  // namespace vesper
