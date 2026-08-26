# Phase 6. Vesper runner

Back to [overview](overview.md).

## Goal

`vesper-infer` can emit the same `DecodeReport` line. If the engine
cannot load the pinned GGUF, the line is `status=unsupported` and
the table still prints. Do not invent tok/s.

## Changes

`scripts/compare-qwen38/run_vesper.sh` invokes
`./build/vesper-infer` with the same prompt, `n`, seed, and
context as the llama.cpp runners. If `--model` is missing, or load
fails on `qwen35`, print `status=unsupported` and exit 0 so the
other two rows still print.

A real row is only legal after the
[destination plan](../rdna4-inference/destination-qwen38-27b.md)
can generate from this file. Do not map `--demo` F32 tok/s onto
this model.

## Data structures

Reuse `DecodeReport`. Engine tag is `vesper`. Backend is `hip` or
`cpu`. Status is `ok` or `unsupported`.

## Verification

**Static.** `ctest` plus `COMPARE_FIXTURE=1` on the wrapper. The
CPU-only build must print `unsupported` for the pinned 27B path.

**Runtime.** Drive `./build/vesper-infer --demo` with `control-cli`
to prove the printer. The 27B row stays `unsupported` until wave 2
of the inference plan.
