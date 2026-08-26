# Phase 3. llama.cpp pin

Back to [overview](overview.md).

## Goal

HIP and Vulkan builds come from one llama.cpp commit that supports
`qwen35`. The compare does not drift when llama.cpp main moves.

## Changes

`scripts/compare-qwen38/llamacpp.env` records the git URL, commit,
HIP cmake flags, and Vulkan cmake flags. `ARTIFACT.md` links that
commit. No binary in git.

The floor is tag `b10502` (2026-08-19) or newer. Community GGUF
cards name that tag as the first build that loads these files.
Pin a later commit if `b10502` HIP or Vulkan fails to load the
pinned Q4_K_M on the R9700. Write the SHA you actually built.

MTP stays off. Flash attention is allowed only if both HIP and
Vulkan use the same setting.

## Data structures

`LlamaPin` is commit, hip_build_dir, vulkan_build_dir, extra args
(`-ngl`, and `-fa` if both backends use it).

## Verification

**Static.** The env file names a 40-char commit.

**Runtime.** On the R9700 only. `llama-cli --version` or the
binary help must run from both build dirs. There is no control
skill for llama.cpp. Keep the two transcripts.
