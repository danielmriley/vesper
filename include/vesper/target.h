#pragma once

// First HIP target: AMD Radeon AI Pro R9700, RDNA 4, gfx1201.
// See docs/TARGET.md. RDNA 3 (gfx1100 / gfx1151) is a later peer.

namespace vesper {

inline constexpr const char* kHipArch = "gfx1201";
inline constexpr int kComputeUnits = 64;
inline constexpr int kWavefront = 32;
inline constexpr int kLdsBytesPerCu = 64 * 1024;
inline constexpr int kCachelineBytes = 256;
inline constexpr int kGemvWorkgroup = 256;
inline constexpr int kGemvRowsPerWg = 8;
inline constexpr int kLdsXMaxElems = 12288;
inline constexpr int kTileXElems = 4096;
inline constexpr int kDefaultContext = 4096;
inline constexpr int kIdlePowerQueues = 1;
inline constexpr double kPeakBandwidthGBs = 640.0;

}  // namespace vesper
