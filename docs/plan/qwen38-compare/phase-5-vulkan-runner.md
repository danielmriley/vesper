# Phase 5. Vulkan runner

Back to [overview](overview.md).

## Goal

The same GGUF and prompt through llama.cpp Vulkan. Same line
format. This is the row Vesper has to beat or match.

## Changes

`scripts/compare-qwen38/run_llamacpp_vulkan.sh` mirrors the HIP
runner with the Vulkan binary and backend `vulkan`.

## Data structures

Same `DecodeReport`. Backend tag is the only required difference.

## Verification

**Static.** Fixture mode, as in phase 4.

**Runtime.** On the R9700, same prompt and `n`. `control-cli` on
the wrapper. Decode tok/s should sit near the 28 to 32 band for
this Q4 file if the run is honest. If it does not, fix the recipe
before you judge Vesper.
