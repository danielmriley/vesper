# Phase 1. Pin the GGUF

Back to [overview](overview.md).

## Goal

The compare names one file. Everyone reruns the same bytes.
Official post-trained Qwen3.8-27B, standard Q4_K_M, text only.
No mmproj. No MTP draft file on the first table.

## Changes

`docs/plan/qwen38-compare/ARTIFACT.md` records the Hugging Face
repo, filename, quant, sha256, prompt text, new-token count, and
context. `scripts/compare-qwen38/artifact.env` exports those
values for the runners. Do not commit the GGUF.

Prefer `ggml-org/Qwen3.8-27B-GGUF` Q4_K_M. That pack is converted
from `Qwen/Qwen3.8-27B` and uses architecture `qwen35`. The Q4_K_M
file is about 19 GB. The MTP file is separate. Leave it off.

Reject these pins:

- an uncensored or abliterated fork
- Unsloth `UD-*` files. Those are not stock Q4_K_M.
- a text-only rename that is a different conversion

Write the exact revision and sha256 after the first download on the
R9700. Until then the pin names the repo and the quant tag.

## Data structures

`ArtifactPin` is repo, revision, filename, sha256, quant tag,
prompt, `n_predict`, context.

## Verification

**Static.** A tiny test reads `artifact.env` and checks that
sha256 is 64 hex chars and quant is `Q4_K_M`.

**Runtime.** No GPU. No download on CI. On the R9700, a later
phase verifies the hash after download.
