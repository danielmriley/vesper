# Phase 10. R9700 gate

Back to [overview](overview.md).

## Goal

You run the same Qwen3-8B Q4_K prompt on Vesper HIP, llama.cpp HIP,
and llama.cpp Vulkan. You write the three `DecodeReport` lines next
to each other. That measurement decides the next plan. It is not a
kernel phase.

## Changes

`scripts/roofline-compare.sh` or a small C++ helper that prints one
line per engine. The script is the lever. Do not add a model zoo.
Do not add HTTP.

Document the exact llama.cpp commands in [testing](testing.md).

## Data structures

Reuse `DecodeReport`. The compare script needs the same field order
for all three rows.

## Verification

**Static.** The script must fail if a row lacks bytes per token or
achieved GB/s.

**Runtime.** On the R9700 only. Same model file, same prompt, same
token count, same context. If Vesper HIP is more than 15% slower
than llama.cpp Vulkan on decode tok/s, HIP lost. The next plan is a
Vulkan peer or a workgroup rewrite. Fusion and HIP graphs are not
the answer to a bad GEMV.

There is no control skill for llama.cpp. Run both CLIs the same way
you run `vesper-infer`, and keep the transcripts.
