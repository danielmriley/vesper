# Phase 9. HIP Q4_K GEMV

Back to [overview](overview.md).

## Goal

The gfx1201 Q4_K GEMV matches the CPU Q4_K GEMV. The tiny model or a
row-sized fixture is enough. Then Qwen3-8B Q4_K can stream through
the same kernel.

This is the kernel that can reach the 90 to 100 tok/s roofline on
Qwen3-8B. Dequant must stay in registers. A dequant-to-F16 tile in
LDS that spends the bandwidth savings is a failed phase, not a
tuning TODO.

## Changes

`src/kernels_hip.hip` adds `rdna4::gemv_q4k`. Tests add a CPU-equals-HIP
gate that skips without a gfx1201 device. CLI `--model` accepts a
Q4_K GGUF.

## Data structures

Same `PackedMatrix` with the Q4_K tag. One workgroup per output row.
Wave32. 256-byte aligned packed loads.

## Verification

**Static.** CPU `ctest`.

**Runtime.** On a R9700, `vesper-tests` plus a short
`vesper-infer --model ...q4k.gguf --device hip` run. `DecodeReport`
bytes per token must sit near 4.5 to 5 GB of weights. Achieved GB/s
under 40% of 640 means the inner loop is wrong. Do not fuse. Do not
graph.
