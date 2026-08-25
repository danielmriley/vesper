# Phase 7. Quality gate

Back to [overview](overview.md).

## Goal

Speed without matching tokens is a different model. Once Vesper
can generate from the pinned file, the first 32 greedy token ids
must match llama.cpp HIP, or you write the first mismatch.

## Changes

`scripts/compare-qwen38/compare_greedy.sh` runs both engines with
temperature 0, seed 1, and the same prompt, then diffs the new
token ids. Skip when Vesper is `unsupported`.

Compare ids, not detokenized text. A detokenizer can hide a
one-id drift.

## Data structures

`GreedyCmp` is prompt, id list per engine, first mismatch index
or `match`.

## Verification

**Static.** Fixture two id lists, one match and one mismatch.

**Runtime.** On the R9700, only after Vesper loads the GGUF.
`control-cli` is optional. The diff output is the proof.
