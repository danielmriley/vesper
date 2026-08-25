# RDNA4 local inference, v1

This is the implementation plan. It is not a method we already shipped.
Do not treat `src/kernels_hip.hip` as a decision.

## Context

You want to run **Qwen3.8-27B** on one card, the
[Radeon AI Pro R9700](../../TARGET.md) (RDNA 4, `gfx1201`).
A local user feels decode tokens per second, and accepted tok/s once
MTP is on. Prefill is time to first token. Those numbers stay
separate.

Qwen3.8-27B is a dense hybrid, not a 3.8B and not a MoE. The current
engine is a Qwen3 GQA loop. It cannot run this checkpoint. The
product path is in
[destination-qwen38-27b.md](destination-qwen38-27b.md).

The GEMV still comes first. Every FFN and every gated-attention
projection is a decode GEMV. A DeltaNet kernel on a 40% bandwidth
Q4 inner loop is a slow 27B.

An F32 HIP path already landed. F32 27B does not fit in 32 GB. That
path cannot be the score.

## Scope

**Product.** Text decode of Qwen3.8-27B Q4_K on the R9700, then MTP.
See [destination-qwen38-27b.md](destination-qwen38-27b.md).

**This wave (phases 1 to 10).** One CLI. One card. GGUF. Q8_0 then
Q4_K. A gfx1201 decode GEMV. CPU F32 oracle. Roofline print.
Compare to llama.cpp on Qwen3-8B Q4 so the kernel is proven on a
small real file before you pay for 27B hybrid work.

**Out of both waves.** HTTP. A model zoo. Continuous batching.
Paged KV. MoE expert cache. Vision / mmproj. A Vulkan backend,
until a gate says HIP lost. Extending the F32 HIP full-engine path.

**Done, this wave.** Qwen3-8B Q4_K on the R9700 prints decode tok/s,
bytes per token, and achieved GB/s, within 15% of llama.cpp Vulkan,
or you write down that HIP lost.

**Done, product.** Qwen3.8-27B Q4_K text decode on the R9700, same
report, then accepted tok/s with MTP. Honest one-token decode is
about 28 to 32 tok/s at 70% of 640 GB/s. MTP is how that feels
faster. It will not become hundreds of tok/s on this card.

## Constraints

- Peak bandwidth is about 640 GB/s. Honest decode is about 70% of that,
  about 450 GB/s. 32 GB of VRAM does not raise tok/s.
- Wavefront size is 32. Cacheline size is 256 B. LDS per CU is 64 KB.
  There is no CDNA-style FP8 MFMA.
- llama.cpp Vulkan already sits near 29 to 33 tok/s on a dense 27B Q4
  stream on this card. That is the ceiling for a dense 27B, not a
  target to invent.
- Qwen3-8B Q4 is about 90 to 100 tok/s at 70% of 640 GB/s if dequant
  stays cheap. Q8 is about 53. F16 is about 27. F32 does not fit.
- CI and this cloud VM have no ROCm and no GPU. CPU tests are the
  merge gate. R9700 numbers are a human run.
- C++17. No Python on the hot path. No per-op Tensor. No alloc after
  load.

## Alternatives

**A. Own the quantized decode GEMV (this plan).**
Load GGUF. Keep CPU F32 as the oracle. Write Q8_0, then Q4_K, GEMV
for gfx1201. Compare to llama.cpp on the R9700. Start a Vulkan peer
only if HIP loses the gate.

**B. Vulkan first.**
Write the same GEMV in SPIR-V. llama.cpp often wins decode this way
on RDNA. You avoid the HIP idle-power queue bug. You also throw away
the in-process HIP dispatch you already have, and you cannot share a
HIP graph later without a second backend. Use this if A fails the
gate. Do not start here on a guess.

**C. Do not write kernels.**
Wrap llama.cpp or hipBLAS and ship a CLI this week. You get a number.
You do not own the kernel that is about 90% of decode. This repo
becomes a front end.

The plan uses A. The score is a number you can move. C cannot teach
the workgroup. B is the pivot after a measured loss, not the opening
bet. Q8 and Q4 math does not care which backend launches it.

## Applicable skills

- `how` before changing `Engine`, weight load, or a kernel file the
  implementer has not read in this session
- `interrogate` before shipping the Q4 GEMV or the llama.cpp gate
- `/deslop` on each diff before commit. `unslop` on every prose file
- `show-me-your-work` if a phase changes the done definition
- `control-cli` for `vesper-infer` checks
- `technical-writing` for any doc or PR text
- `poteto-mode` plan rules. One phase is two or three files and one
  check

## Phases

1. [Score contract](phase-1-score-contract.md). `DecodeReport` and a
   printer that cannot omit the roofline fields.
2. [GGUF map](phase-2-gguf-map.md). Header and tensor table. No
   dequant.
3. [CPU Q8 GEMV](phase-3-cpu-q8-gemv.md). Q8_0 block and a GEMV that
   matches F32.
4. [CPU Q8 decode](phase-4-cpu-q8-decode.md). Tiny demo runs on Q8
   weights.
5. [HIP Q8 GEMV](phase-5-hip-q8-gemv.md). One gfx1201 kernel. CPU
   equals HIP. Do not extend F32 HIP.
6. [HIP Q8 decode](phase-6-hip-q8-decode.md). Tiny demo on HIP Q8.
7. [Qwen3-8B Q8](phase-7-qwen3-8b-q8.md). Real GGUF. Print the
   roofline.
8. [CPU Q4_K](phase-8-cpu-q4k.md). Oracle for the product quant.
9. [HIP Q4_K GEMV](phase-9-hip-q4k.md). The kernel that can hit the
   score.
10. [R9700 gate](phase-10-r9700-gate.md). Same 8B Q4 against llama.cpp
    HIP and Vulkan.

After phase 10, follow
[destination-qwen38-27b.md](destination-qwen38-27b.md). Do not start
Gated DeltaNet or MTP while the GEMV is still under 40% of peak.

The three-row R9700 table for official Qwen3.8-27B lives in
[../qwen38-compare/overview.md](../qwen38-compare/overview.md).

[Testing](testing.md) lists project checks.

## Verification

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/vesper-infer --demo --prompt hello --tokens 8
```

On a R9700, after phase 5:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVESPER_USE_HIP=ON
./build/vesper-infer --hip-info
```

There is no control skill for HIP kernel occupancy. The R9700 run in
phase 10 is the runtime proof. CI cannot do that run.

## Implementation guidance

Read [this file](overview.md) before the phase file. Do not start
phase N+1 while phase N is red.

Apply these poteto-mode rules on every phase:

- the **how** skill on each unfamiliar subsystem before you edit it
- the **interrogate** skill before you call the Q4 GEMV done
- `/deslop` on the diff before commit. **unslop** on prose
- **show-me-your-work** if you change the done line in this overview
- Cursor **babysit** only after the user asks, not at each PR

Do not grow `src/kernels_hip.hip` F32 decode. The first GPU kernel
that counts is Q8 GEMV. F32 HIP may stay as a compile toy until
phase 5 replaces the GEMV.

If a phase needs more than three files or more than one new
operation, split the phase. Do not silently enlarge it.
