# Phase 3. CPU Q8 GEMV

Back to [overview](overview.md).

## Goal

A CPU `gemv_q8` on GGUF Q8_0 rows matches `gemv` on the dequantized
F32 matrix, inside a tight tolerance. This is the oracle for the
first GPU kernel.

## Changes

`include/vesper/quant.h` and `src/quant_q8.cpp` define the Q8_0 block
and the row GEMV. `tests/test_main.cpp` builds a small F32 matrix,
packs it to Q8_0, and compares both GEMVs.

Do not load a real Qwen file yet.

## Data structures

`Q8Block` is the GGUF Q8_0 block, 32 int8 values and one F16 scale.
`PackedMatrix` is rows, cols, quant tag, and a 256-byte-aligned row
stride. F32 `gemv` stays the numeric gate.

## Verification

**Static.** `ctest --test-dir build --output-on-failure`

**Runtime.** The unit test is the check. No CLI change. No GPU.
