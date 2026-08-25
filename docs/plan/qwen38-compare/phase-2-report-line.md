# Phase 2. Report line

Back to [overview](overview.md).

## Goal

One line is one engine run. Missing bytes per token or achieved
GB/s is a failed run. This is the lever. Later runners only print
this line.

## Changes

Reuse the `DecodeReport` in
[../rdna4-inference/phase-1-score-contract.md](../rdna4-inference/phase-1-score-contract.md).
If that type is not on disk yet, add `include/vesper/report.h` and
`src/report.cpp` here with that same field list. Do not invent a
second schema.

`scripts/compare-qwen38/parse_report.py` or a C++ helper accepts
one line and rejects a short row. `tests/test_main.cpp` feeds a
fixture line.

## Data structures

Same `DecodeReport` as the inference plan. Engine, backend, status,
prompt tokens, and new tokens stay required so a llama.cpp row and
an `unsupported` Vesper row parse the same way.

## Verification

**Static.** `ctest --test-dir build --output-on-failure`

**Runtime.** Pipe a fixture line through the parser. Drive with
`control-cli` if the helper is a binary. No GGUF.
