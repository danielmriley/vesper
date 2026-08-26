#pragma once

#include <cstddef>

// First HIP target: AMD Radeon AI Pro R9700, RDNA 4, gfx1201.
// See docs/TARGET.md. RDNA 3 (gfx1100 / gfx1151) is a later peer.

namespace vesper {

inline constexpr const char* kHipArch = "gfx1201";
inline constexpr int kComputeUnits = 64;
inline constexpr int kWavefront = 32;
inline constexpr int kLdsBytesPerCu = 64 * 1024;
inline constexpr int kCachelineBytes = 256;
inline constexpr int kGemvWorkgroup = 256;
// llama.cpp RDNA4 MMVQ (#19478): 8 waves, 1 output row for bs=1 decode.
inline constexpr int kGemvRowsPerWg = 1;
inline constexpr int kGemvWaves = 8;
// RDNA can load 16 B/thread (llama.cpp #22821). Two Q4 iqs slices share
// one scale table and two aligned int2 qs loads, so a super is 8 threads
// and a WG keeps 32 supers in flight. Official FFN down is 3 K-trips
// (68 supers). SwiGLU at 5120 is 1 (20 supers). llama.cpp still uses
// 16 threads / 16 supers (5 and 2 trips on those shapes).
inline constexpr int kQ4MmvqThreadsPerSuper = 8;
inline constexpr int kQ4MmvqSuperStride = kGemvWorkgroup / kQ4MmvqThreadsPerSuper;
// One thread does a full Q8_0 block (32 B qs, two 16 B x loads).
// 256 blocks/iter. Official attn/SSM 5120 and 6144 are 1 K-trip.
// GDN qkv 10240 is 2. llama.cpp still uses 4 threads / VDR=2.
inline constexpr int kQ8MmvqThreadsPerBlock = 1;
inline constexpr int kQ8MmvqPerIter = kGemvWorkgroup / kQ8MmvqThreadsPerBlock;
// Four consecutive Q6 iqs share bq8_offset, scales, and vh shift. 8
// threads per super, 32 supers in flight. Official lm_head / o_proj at
// 5120 / 6144 are 1 K-trip (was 2 with a pair).
inline constexpr int kQ6MmvqThreadsPerSuper = 8;
inline constexpr int kQ6MmvqSuperStride = kGemvWorkgroup / kQ6MmvqThreadsPerSuper;
// Per-row kernels over a head or SSM dim. Official GDN is 128, so a 256-thread
// WG left half the lanes idle. MMVQ stays at kGemvWorkgroup.
inline constexpr int row_workgroup(int dim) {
    return dim <= 128 ? 128 : kGemvWorkgroup;
}
// llama.cpp gated_delta_net.cu: 4 warps, each owns one S column in registers.
inline constexpr int kGdnDeltaWarps = 4;
inline constexpr int kGdnDeltaMaxDim = 256;
inline constexpr int kGdnDeltaRowsPerLane = kGdnDeltaMaxDim / kWavefront;

// Compile-time shard count for one wave. Official GDN is 128 (4), official
// gated attn is 256 (8). Do not keep a 256-wide register file on the 128 path.
inline constexpr int gdn_delta_shard_rows(int dim) {
    const int need = (dim + kWavefront - 1) / kWavefront;
    if (need <= 1) {
        return 1;
    }
    if (need <= 2) {
        return 2;
    }
    if (need <= 4) {
        return 4;
    }
    return kGdnDeltaRowsPerLane;
}
inline constexpr int kQuantizeRowsPerWg = 8;
// 0: never copy Q8_1 x into LDS. llama.cpp MMVQ reads it from L2.
// Staging 5120 B still put a barrier on every ffn_up and lm_head WG.
// lm_head is 248320 WGs. Official FFN down is 19584 B and left 3 WGs/CU.
inline constexpr int kLdsQ8xMaxBytes = 0;
inline constexpr int kDefaultContext = 4096;
inline constexpr int kIdlePowerQueues = 1;
inline constexpr int kDecodeGraphSlot = 0;
// Engine HIP init tries one full-token graph first (n_layers). If
// instantiate fails it halves: 64 → 32 → 16 → 8 → 4 → 2 → 1, then eager.
// kDecodeGraphChunkLayers is the conservative size that should still fit
// RDNA. Official 27B is 4 slots at 16. A failed instantiate cannot
// consume a generated token.
inline constexpr int kDecodeGraphChunkLayers = 16;
// After a multi-chunk capture, init may compose those graphs as child
// nodes of one parent exec. Slot 64 sits above a 1-layer official
// fallback (slots 0..63).
inline constexpr int kDecodeGraphParentSlot = 64;

inline constexpr int decode_graph_chunks(int n_layers,
                                        int chunk_layers = kDecodeGraphChunkLayers) {
    if (n_layers <= 0 || chunk_layers <= 0) {
        return 1;
    }
    return (n_layers + chunk_layers - 1) / chunk_layers;
}

// 16 failed instantiate: try 8, then 4, then 2, then 1. 0 means eager.
inline constexpr int next_decode_graph_chunk_layers(int chunk_layers) {
    return chunk_layers > 1 ? chunk_layers / 2 : 0;
}
inline constexpr std::size_t kHipCopyChunkBytes = 64u * 1024u * 1024u;
inline constexpr double kPeakBandwidthGBs = 640.0;

}  // namespace vesper
