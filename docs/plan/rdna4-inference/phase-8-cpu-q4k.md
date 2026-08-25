# Phase 8. CPU Q4_K

Back to [overview](overview.md).

## Goal

A CPU `gemv_q4k` matches F32 GEMV on a packed fixture. Q4_K is the
product quant. Do not write the HIP kernel until this oracle exists.

## Changes

`src/quant_q4k.cpp` and a header addition in `include/vesper/quant.h`.
`tests/test_main.cpp` packs a small matrix to Q4_K and compares
against F32.

Start this phase only after phase 7 reports at least 40% of peak
bandwidth on Q8. If Q8 is still broken, Q4 will lie.

## Data structures

`Q4KBlock` follows the GGUF Q4_K super-block. `PackedMatrix` accepts
a Q4_K tag. Same row-stride rule, 256 bytes.

## Verification

**Static.** `ctest --test-dir build --output-on-failure`

**Runtime.** Unit test only. No CLI.
