#pragma once

namespace vesper {

// Map a fused gemv3/gemv4 grid row onto one of four output matrices.
// Official attn is 12288+1024+1024. Official GDN is 10240+6144+48+48.
struct MultiRowPick {
    int slot;
    int local;
};

#if defined(__HIPCC__)
__host__ __device__
#endif
inline bool pick_multi_row(int row, int r0, int r1, int r2, int r3, MultiRowPick* out) {
    if (row < r0) {
        out->slot = 0;
        out->local = row;
        return true;
    }
    row -= r0;
    if (row < r1) {
        out->slot = 1;
        out->local = row;
        return true;
    }
    row -= r1;
    if (row < r2) {
        out->slot = 2;
        out->local = row;
        return true;
    }
    row -= r2;
    if (row < r3) {
        out->slot = 3;
        out->local = row;
        return true;
    }
    return false;
}

}  // namespace vesper
