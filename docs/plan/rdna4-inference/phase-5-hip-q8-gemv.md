# Phase 5. HIP Q8 GEMV

Back to [overview](overview.md).

## Goal

One gfx1201 kernel computes Q8_0 GEMV. Host tests copy a packed
matrix to HIP, run the kernel, and match the CPU Q8 GEMV. This is
the first GPU work that can move the score.

Do not extend the F32 HIP full-engine path. Do not add HIP RMSNorm,
RoPE, or attention in this phase.

## Changes

`src/kernels_hip.hip` adds `rdna4::gemv_q8` only. `include/vesper/rdna4.h`
declares it. `tests/test_main.cpp` skips the gate when
`hip_available()` is false.

If F32 `rdna4::gemv` is in the way, delete or stop calling it. Do not
grow it.

## Data structures

Same `PackedMatrix` as phase 3. Device pointer plus the Q8_0 row
stride. Workgroup is one output row, wave32, 256-byte loads.

## Verification

**Static.** Default CPU build and `ctest` stay green without ROCm.

**Runtime.** On a R9700, build with `-DVESPER_USE_HIP=ON` and run
`vesper-tests`. The Q8 HIP gate must pass. There is no control skill
for occupancy. The numeric match is the check in this phase.
