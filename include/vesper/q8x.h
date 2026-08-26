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
// sum[b] = d * sum(qs) so the Q4_K/Q5_K min term stays an integer reduction.
void quantize_q8x(const float* x, std::int8_t* qs, float* d, float* sum, int n);
void dequant_q8x(float* x, const std::int8_t* qs, const float* d, int n);

}  // namespace vesper
