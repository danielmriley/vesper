# Phase 4. CPU Q8 decode

Back to [overview](overview.md).

## Goal

The tiny demo can run from Q8_0 packed projections on CPU. Greedy
tokens match the F32 engine on the same seed, or you document a
logit tolerance and still match argmax.

## Changes

`src/weights.cpp` can pack random F32 projections to Q8_0.
`src/engine.cpp` calls `gemv_q8` for those projections. Keep
activations and KV in F32. Touch at most those files plus the test.

## Data structures

`ModelWeights` gains a quant tag and `PackedMatrix` slots for the
projections. `Engine` still owns one model, one KV arena, and the
scratch set.

## Verification

**Static.** `ctest --test-dir build --output-on-failure`

**Runtime.** `./build/vesper-infer --demo --prompt hello --tokens 8`
still runs. Drive it with `control-cli`. Compare greedy ids to a
known F32 run on seed 1.
