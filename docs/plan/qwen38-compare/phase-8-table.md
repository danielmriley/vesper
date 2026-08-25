# Phase 8. Table command

Back to [overview](overview.md).

## Goal

One command on the R9700 prints three rows. That table is the
compare, not three log files.

## Changes

`scripts/compare-qwen38/compare.sh` sources `artifact.env`, runs
the three runners, parses each `DecodeReport` or `status=` line,
and prints one markdown table. Header lines above the table name
the GGUF sha256, the llama.cpp commit, the Vesper commit, the
prompt, `n`, and context. A table without those lines is not this
compare.

If a runner prints `status=unsupported`, the tok/s cell is
`unsupported`. Do not write `0`.

Do not commit guessed numbers. A `RESULTS.md` filled on the R9700
is a human artifact, not a CI output.

## Data structures

The table is three `DecodeReport` rows plus an `ids` column.
`ids` is `32` after phase 7 passes, `unsupported` before Vesper
generates, and `mismatch` if phase 7 fails.

## Verification

**Static.** `COMPARE_FIXTURE=1 ./scripts/compare-qwen38/compare.sh`
prints a three-row table from checked-in snippets. No network. No
GGUF.

**Runtime.** On the R9700, run `compare.sh` for real. llama.cpp
HIP and Vulkan have real tok/s. Vesper is real tok/s plus matching
ids, or `unsupported`. Drive the wrapper with `control-cli`. There
is no control skill for llama.cpp. Keep the three transcripts.
