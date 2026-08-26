# Compare Qwen3.8-27B on llama.cpp and Vesper

This plan sets up a rerunnable compare. It does not implement Gated
DeltaNet. The GEMV and hybrid work stays in
[../rdna4-inference/overview.md](../rdna4-inference/overview.md).

## Context

You want to run the official Qwen3.8-27B release on the R9700 and
see it next to llama.cpp and next to Vesper. There is no 3.8B
Qwen3.8. The ".8" is the family. The open dense model is 27B.
"Prem" here means that official post-trained checkpoint
(`Qwen/Qwen3.8-27B`), not a base-only dump and not Qwen3.8-Max.

Vesper cannot load this file yet. llama.cpp can, if the build knows
`qwen35`. The first useful compare is therefore two llama.cpp
backends plus an empty Vesper row that the later engine fills.

A local user feels decode tok/s. The table must also print bytes per
token and achieved GB/s, or the number is advertising.

## Scope

**In.** One pinned GGUF. One prompt. One token count. One context.
Three rows. llama.cpp HIP. llama.cpp Vulkan. Vesper HIP. A script
that prints one `DecodeReport` line per row and fails if a field is
missing. That type is the same one as
[../rdna4-inference/phase-1-score-contract.md](../rdna4-inference/phase-1-score-contract.md).
A quality check once Vesper can emit tokens. Greedy ids or
a stated mismatch.

**Out.** HuggingFace Transformers as a third engine. Ollama.
vLLM and SGLang. Qwen3.8-Max 2.4T. Vision and the mmproj file.
HTTP. A model zoo. Implementing DeltaNet in this plan.

**Done.** On the R9700, one command writes three rows for the pinned
`ggml-org/Qwen3.8-27B-GGUF` Q4_K_M file. llama.cpp HIP and Vulkan
are filled. Vesper is filled once
[the inference plan](../rdna4-inference/destination-qwen38-27b.md)
can generate. Until then the Vesper row is `unsupported` and the
script still exits 0 on the llama.cpp rows.

## Constraints

- R9700, gfx1201, ~640 GB/s, 32 GB. The ggml-org Q4_K_M file is
  about 19 GB. It fits. Q8 is tight.
- Honest one-token decode is about 28 to 32 tok/s at 70% of peak.
  That is the llama.cpp ceiling to match, not a number to invent.
- CI has no GPU and must not download a 19 GB file. Tests use a
  fixture line and a fake runner.
- Same file, prompt, `-n`, and context on every row. MTP off for the
  one-token compare. MTP is a later column, not the first table.
- llama.cpp must be a pinned commit that supports `qwen35`.

## Alternatives

**A. Three runners, one GGUF (this plan).**
llama.cpp HIP, llama.cpp Vulkan, Vesper. Same Q4_K_M file. This is
the only fair tok/s compare on this card.

**B. Official Qwen stack, llama.cpp, Vesper.**
The official recipe is CUDA, vLLM, or SGLang. It does not give an
honest R9700 number. Rejected.

**C. Ollama as the first row.**
Ollama wraps llama.cpp. You would compare Vesper to a wrapper, not
to the kernel. Rejected.

## Applicable skills

- `how` before editing `vesper-infer` or adding a runner adapter
- `interrogate` before you treat a Vesper vs llama.cpp tok/s delta
  as a win
- `/deslop` and `unslop` on each diff
- `control-cli` for the compare script and `vesper-infer`
- `technical-writing` for docs and PR text
- `show-me-your-work` if the pinned GGUF or the done line changes

## Phases

1. [Pin the GGUF](phase-1-pin-gguf.md). URL, quant, sha256, prompt.
2. [Report line](phase-2-report-line.md). One row schema. Fixture
   test. No 19 GB download.
3. [llama.cpp pin](phase-3-llamacpp-pin.md). Commit and build flags
   for HIP and Vulkan.
4. [HIP runner](phase-4-hip-runner.md). Fill the HIP row.
5. [Vulkan runner](phase-5-vulkan-runner.md). Fill the Vulkan row.
6. [Vesper runner](phase-6-vesper-runner.md). Same line, or
   `unsupported`.
7. [Quality gate](phase-7-quality-gate.md). Greedy match once Vesper
   generates.
8. [Table command](phase-8-table.md). One script, three rows.

[Testing](testing.md) lists the checks.

## Verification

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The compare script on CI uses fixture runners. The R9700 run is the
real table. There is no control skill for llama.cpp. Drive it the
same way as `vesper-infer` and keep the transcripts.

## Implementation guidance

Do not start Vesper hybrid work in this directory. Point at
[../rdna4-inference/destination-qwen38-27b.md](../rdna4-inference/destination-qwen38-27b.md)
when the Vesper row must become real.

On every phase:

- the **how** skill on an unread subsystem
- **interrogate** before publishing a speed winner
- `/deslop` before commit. **unslop** on prose
- **show-me-your-work** if the pinned file changes
- Cursor **babysit** only when the user asks
