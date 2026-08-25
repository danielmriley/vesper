# Phase 2. Report line

Back to [overview](overview.md).

## Goal

One line is one engine run. Missing bytes per token or achieved
GB/s is a failed run. This is the lever. Later runners only print
this line.

## Changes

If `DecodeReport` from the inference plan does not exist yet, add
`include/vesper/report.h` and `src/report.cpp` here. Otherwise
reuse them. `scripts/compare-qwen38/parse_report.py` or a C++
helper accepts one line and rejects a short row.
`tests/test_main.cpp` feeds a fixture line.

## Data structures

`DecodeReport` holds engine, backend, model, quant, arch, prompt
tokens, new tokens, prefill tok/s, decode tok/s, bytes per token,
achieved GB/s, peak GB/s (640), context, status
(`ok` or `unsupported`).

## Verification

**Static.** `ctest --test-dir build --output-on-failure`

**Runtime.** Pipe a fixture line through the parser. Drive with
`control-cli` if the helper is a binary. No GGUF.
