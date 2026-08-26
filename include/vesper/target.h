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
// llama.cpp gated_delta_net.cu: 4 warps, each owns one S column in registers.
inline constexpr int kGdnDeltaWarps = 4;
inline constexpr int kGdnDeltaMaxDim = 256;
inline constexpr int kGdnDeltaRowsPerLane = kGdnDeltaMaxDim / kWavefront;
inline constexpr int kQuantizeRowsPerWg = 8;
inline constexpr int kLdsXMaxElems = 12288;
inline constexpr int kLdsQ8xMaxBytes = 32768;
inline constexpr int kTileXElems = 4096;
inline constexpr int kDefaultContext = 4096;
inline constexpr int kIdlePowerQueues = 1;
inline constexpr std::size_t kHipCopyChunkBytes = 64u * 1024u * 1024u;
inline constexpr double kPeakBandwidthGBs = 640.0;

}  // namespace vesper
