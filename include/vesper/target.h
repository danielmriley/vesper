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
// kGemvWaves is the max. Official shapes often fill 5 or 6. Launch is
// mmvq_launch_threads. reduce_wg sums blockDim waves only.
inline constexpr int kGemvRowsPerWg = 1;
inline constexpr int kGemvWaves = 8;
// RDNA can load 16 B/thread (llama.cpp #22821). Pair (8 threads) is for
// narrow Q4, at most 16 supers. Official SwiGLU is 20 supers: the old
// 256-thread launch left a 4-thread map idle, so it stayed on pair.
// Launch now follows work items, so that grid uses the quad map (80
// work items, 3 waves) and 16 B qs/x loads. Mid-width Q4 (33-64
// supers) stays 4 threads / 1 trip. Official FFN down is 68 supers:
// mid stride is 64, so the 4-thread map needs a leftover trip. Oct
// (2 threads, one 128-wide oct each) keeps OneTrip, 136 work items,
// 5 waves. One-thread super stays for 129-256 supers. llama.cpp still
// uses 16 threads / 16 supers.
inline constexpr int kQ4MmvqThreadsPerSuper = 8;
inline constexpr int kQ4MmvqSuperStride = kGemvWorkgroup / kQ4MmvqThreadsPerSuper;
inline constexpr int kQ4MmvqPairMaxSupers = 16;
inline constexpr int kQ4MmvqMidThreadsPerSuper = 4;
inline constexpr int kQ4MmvqMidSuperStride = kGemvWorkgroup / kQ4MmvqMidThreadsPerSuper;
inline constexpr int kQ4MmvqOctThreadsPerSuper = 2;
inline constexpr int kQ4MmvqOctSuperStride = kGemvWorkgroup / kQ4MmvqOctThreadsPerSuper;
inline constexpr int kQ4MmvqDownThreadsPerSuper = 1;
inline constexpr int kQ4MmvqDownSuperStride = kGemvWorkgroup / kQ4MmvqDownThreadsPerSuper;
// HIP instantiates one Q4 map per launch. A runtime pair/quad/oct/super
// branch would charge official SwiGLU and down for every map's VGPRs.
static_assert(kQ4MmvqThreadsPerSuper != kQ4MmvqMidThreadsPerSuper &&
                  kQ4MmvqThreadsPerSuper != kQ4MmvqOctThreadsPerSuper &&
                  kQ4MmvqThreadsPerSuper != kQ4MmvqDownThreadsPerSuper &&
                  kQ4MmvqMidThreadsPerSuper != kQ4MmvqOctThreadsPerSuper &&
                  kQ4MmvqMidThreadsPerSuper != kQ4MmvqDownThreadsPerSuper &&
                  kQ4MmvqOctThreadsPerSuper != kQ4MmvqDownThreadsPerSuper,
              "Q4 MMVQ maps must have distinct thread counts");

inline constexpr int q4_mmvq_threads(int supers) {
    if (supers <= kQ4MmvqPairMaxSupers) {
        return kQ4MmvqThreadsPerSuper;
    }
    if (supers <= kQ4MmvqMidSuperStride) {
        return kQ4MmvqMidThreadsPerSuper;
    }
    if (supers <= kQ4MmvqOctSuperStride) {
        return kQ4MmvqOctThreadsPerSuper;
    }
    return kQ4MmvqDownThreadsPerSuper;
}

inline constexpr int q4_mmvq_stride(int threads) {
    return kGemvWorkgroup / threads;
}

// Pair/mid/oct/down strides are 32/64/128/256 supers. Official 20 and
// 68 fit in one trip. HIP instantiates a leftover-free kernel when this
// is true. A 257-super row still needs the leftover loop.
// HIP Q4/Q8/Q6 packed is matrix SoA (all headers, then all qs). Same
// bytes as GGUF on official Q4 and Q8 shapes and on Q6 o_proj (24
// supers). Official Q6 lm_head pads d by 8 B/row. Cached headers stay
// off the NT qs stream.
inline constexpr bool q4_mmvq_one_trip(int cols) {
    const int supers = cols / 256;
    return supers > 0 && supers <= q4_mmvq_stride(q4_mmvq_threads(supers));
}
// One thread does two consecutive Q8_0 blocks (each 32 B qs, two 16 B
// x loads). Official attn/SSM/GDN in_proj K is 5120 or 6144: 80 or 96
// work items, 3-wave launch, 1 K-trip. A 10240-wide K walk is also 1
// trip (was 2). llama.cpp still uses 4 threads per block.
inline constexpr int kQ8MmvqThreadsPerBlock = 1;
inline constexpr int kQ8MmvqBlocksPerThread = 2;
inline constexpr int kQ8MmvqPerIter = kGemvWorkgroup / kQ8MmvqThreadsPerBlock;
// Eight consecutive Q6 iqs per thread (two quads). 4 threads per super,
// 64 supers in flight. Official lm_head / o_proj at 5120 / 6144 are 80
// or 96 work items, 3-wave launch, 1 K-trip. llama.cpp still uses 16
// threads / 16 supers on the CUDA table.
inline constexpr int kQ6MmvqThreadsPerSuper = 4;
inline constexpr int kQ6MmvqSuperStride = kGemvWorkgroup / kQ6MmvqThreadsPerSuper;

// Idle waves in a 256-thread WG still occupy VGPR file. Official Q4
// SwiGLU, Q8 K 5120/6144, and Q6 lm_head/o_proj are 80 or 96 work items
// (3 waves). Official Q4 down is 136 work items (5 waves). Q5 still
// launches 256: its walk is s += kGemvWaves, so a short launch would
// miss supers.
inline constexpr int mmvq_launch_threads(int work_items) {
    if (work_items <= 0) {
        return kWavefront;
    }
    const int waves = (work_items + kWavefront - 1) / kWavefront;
    const int threads = waves * kWavefront;
    return threads < kGemvWorkgroup ? threads : kGemvWorkgroup;
}

inline constexpr int q4_mmvq_launch(int cols) {
    const int supers = cols / 256;
    return mmvq_launch_threads(supers * q4_mmvq_threads(supers));
}

inline constexpr int mmvq_waves(int launch_threads) {
    return launch_threads / kWavefront;
}

// HIP host switch instantiates one of these. Bench prints the name so an
// R9700 --bench-q4 log shows whether official down is on oct (5 waves).
inline constexpr const char* q4_mmvq_map_name(int threads) {
    if (threads == kQ4MmvqThreadsPerSuper) {
        return "pair";
    }
    if (threads == kQ4MmvqMidThreadsPerSuper) {
        return "quad";
    }
    if (threads == kQ4MmvqOctThreadsPerSuper) {
        return "oct";
    }
    if (threads == kQ4MmvqDownThreadsPerSuper) {
        return "super";
    }
    return "unknown";
}

inline constexpr int q8_mmvq_launch(int cols) {
    const int nblocks = cols / 32;
    const int work =
        (nblocks + kQ8MmvqBlocksPerThread - 1) / kQ8MmvqBlocksPerThread;
    return mmvq_launch_threads(work);
}

// Official Q8 K 5120/6144/10240 has an even block count. Those grids
// compile two unconditional dots. A leftover 3-block walk keeps the
// tail check.
inline constexpr bool q8_mmvq_tight(int cols) {
    const int nblocks = cols / 32;
    return nblocks > 0 && (nblocks % kQ8MmvqBlocksPerThread) == 0;
}

// One thread slot covers kQ8MmvqPerIter block-pairs. Official 5120/6144
// /10240 are 80/96/160 slots. HIP instantiates a leftover-free kernel
// when this is true. Wider than 256 slots needs another trip.
inline constexpr bool q8_mmvq_one_trip(int cols) {
    const int nblocks = cols / 32;
    const int work = (nblocks + kQ8MmvqBlocksPerThread - 1) / kQ8MmvqBlocksPerThread;
    return work > 0 && work <= kQ8MmvqPerIter;
}

inline constexpr int q6_mmvq_launch(int cols) {
    return mmvq_launch_threads((cols / 256) * kQ6MmvqThreadsPerSuper);
}

// Official lm_head / o_proj are 20 / 24 supers. Stride is 64. HIP
// instantiates a leftover-free kernel when this is true. A 65-super Q6
// row still needs the leftover loop.
inline constexpr bool q6_mmvq_one_trip(int cols) {
    const int supers = cols / 256;
    return supers > 0 && supers <= kQ6MmvqSuperStride;
}

// Per-row kernels over a head or SSM dim. Official GDN is 128, so a 256-thread
// WG left half the lanes idle.
inline constexpr int row_workgroup(int dim) {
    return dim <= 128 ? 128 : kGemvWorkgroup;
}
// Official and tiny-hybrid GDN conv is k=4. HIP instantiates that kernel so
// the hist=3 walk and 16-byte weight load are compile-time.
inline constexpr int kGdnConvKernel = 4;
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

// Official GDN 128 and attn 256 fill the shard file. Those kernels can
// drop i < dim in the load/store and in the attn seq walk. Tiny dim 16
// still has a partial last shard.
inline constexpr bool gdn_delta_tight(int dim) {
    return dim > 0 && dim == gdn_delta_shard_rows(dim) * kWavefront;
}
inline constexpr int kQuantizeRowsPerWg = 8;
// 0: never copy Q8_1 x into LDS. llama.cpp MMVQ reads it from L2.
// Staging 5120 B still put a barrier on every ffn_up and lm_head WG.
// lm_head is 248320 WGs. Official FFN down is 19584 B and left 3 WGs/CU.
inline constexpr int kLdsQ8xMaxBytes = 0;
inline constexpr int kDefaultContext = 4096;
// Official Qwen3.8-27B hidden. copy/add/final rmsnorm launch 256 threads
// and write Q8_1 for the next packed GEMV. 5120 % 256 == 0 and
// (5120/4) % 256 == 0, so that path is five unrolled float4 trips.
inline constexpr int kOfficialHidden = 5120;
// Official GDN head dim. tile_gates and rmsnorm_silu launch 128 threads,
// so each lane owns one element and the dim walk is compile-time.
inline constexpr int kOfficialGdnDim = 128;
// Official gated attn. prepare launches 256 threads at head_dim 256.
// Rope is 64 (32 pairs). Each lane owns one dim.
inline constexpr int kOfficialHeadDim = 256;
inline constexpr int kOfficialRopeDim = 64;
// Official vocab. 248320 / 256 == 970, so argmax_partial is one element
// per thread. Wider than 1024 partial WGs still strides.
inline constexpr int kOfficialVocab = 248320;
inline constexpr int kArgmaxMaxPartialBlocks = 1024;

inline constexpr int argmax_partial_blocks(int n) {
    if (n <= 0) {
        return 1;
    }
    const int nblocks = (n + kGemvWorkgroup - 1) / kGemvWorkgroup;
    return nblocks < kArgmaxMaxPartialBlocks ? nblocks : kArgmaxMaxPartialBlocks;
}

inline constexpr bool argmax_one_trip(int n) {
    return n > 0 && (n % kGemvWorkgroup) == 0 &&
           (n / kGemvWorkgroup) <= kArgmaxMaxPartialBlocks;
}
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
