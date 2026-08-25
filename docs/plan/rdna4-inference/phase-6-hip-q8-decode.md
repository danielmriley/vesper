# Phase 6. HIP Q8 decode

Back to [overview](overview.md).

## Goal

`--device hip` runs the tiny Q8 demo. Greedy tokens match CPU Q8 on
the same seed. The CLI prints a `DecodeReport`. The number is not a
claim against llama.cpp.

## Changes

`src/engine.cpp` launches `gemv_q8` on HIP for packed projections.
Activations and KV may stay F32 and cross the bus for this tiny
model. `src/cli.cpp` fills `DecodeReport` from the run.

Do not fuse kernels. Do not capture a HIP graph.

## Data structures

No new types. `Engine` already takes a `Device`. Weights on HIP are
`PackedMatrix` copies, not F32 clones of the whole model.

## Verification

**Static.** `ctest` on the CPU build.

**Runtime.** On a R9700:

```bash
./build/vesper-infer --demo --device hip --prompt hello --tokens 8
```

Drive that command with `control-cli`. Check device, arch, and that
greedy ids match a CPU run. Skip on CI.
