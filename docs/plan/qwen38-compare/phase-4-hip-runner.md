# Phase 4. HIP runner

Back to [overview](overview.md).

## Goal

One command runs llama.cpp HIP on the pinned GGUF and prints one
`DecodeReport` line. This is the first real R9700 number.

## Changes

`scripts/compare-qwen38/run_llamacpp_hip.sh` downloads if missing,
checks sha256, runs the pinned binary, and maps llama.cpp timing
to `DecodeReport`. Peak GB/s is 640.

## Data structures

No new types. The script writes one `DecodeReport` line to stdout.

## Verification

**Static.** CI runs the script with `COMPARE_FIXTURE=1` and a fake
binary that prints known timings. The parser accepts the line.

**Runtime.** On the R9700, run the script for real. Drive the
wrapper with `control-cli`. Confirm engine is `llamacpp`, backend
is `hip`, arch is `gfx1201`, and decode tok/s is filled.
