# Phase 1. Score contract

Back to [overview](overview.md).

## Goal

A decode run prints a `DecodeReport`. The printer refuses to omit
bytes per token, achieved GB/s, or the model and quant names. Later
phases fill the fields. This phase makes a tok/s-only print illegal.

## Changes

`include/vesper/report.h` and `src/report.cpp` own the type and the
roofline math. `src/cli.cpp` prints the report after `--demo`.
`tests/test_main.cpp` checks the math on a known bytes and tok/s pair.

## Data structures

`DecodeReport` holds model, quant, device, arch, bytes per token,
prefill tok/s, decode tok/s, achieved GB/s, peak GB/s (640), and
context length.

## Verification

**Static.** `ctest --test-dir build --output-on-failure`

**Runtime.** `./build/vesper-infer --demo --prompt hello --tokens 8`
must print every `DecodeReport` field. Drive the CLI with the
`control-cli` skill. Confirm the line set, not only the exit code.
