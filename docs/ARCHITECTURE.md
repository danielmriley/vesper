# From-scratch local engine, designed for max tok/s

This is the engine I would build if the only score was single-user
tokens per second on one consumer GPU. The first GPU is the
**Radeon AI Pro R9700 (RDNA 4, gfx1201)**. See [TARGET.md](TARGET.md).
RDNA 3 is a later peer. Multi-user serving is a different product.

v0 in this tree is the CPU-correct skeleton. This note is the
destination. Order of work is [TOKS.md](TOKS.md). Other engines are
[ENGINES.md](ENGINES.md).

## The score

A local user feels **decode tok/s**. Prefill is time-to-first-token.
Report both, plus accepted tok/s if you speculate, plus model, quant,
context, and GPU. Never mix them.

Decode is almost always:

```
tok/s ≈ achieved_bandwidth / bytes_touched_per_token
```

There are only three ways to raise that number:

1. **Fewer bytes per token** — quantize weights, quantize KV, skip
   experts you did not route to.
2. **More of the card's bandwidth** — fused quantized GEMV with the
   right workgroup, not a GEMM at M=1, not 1600 launches per token.
3. **More tokens per weight stream** — speculative decode / MTP.
   Stream the model once, accept K tokens.

Everything else (HTTP, paged attention, continuous batching, a model
zoo) is either a serving feature or a tax. None of it raises
single-user decode until 1–3 are done.

On the R9700 (~640 GB/s peak, ~70% honest ≈ 450 GB/s) a dense
Qwen3-8B is roughly 27 tok/s F16, 53 Q8, 90–100 Q4 if dequant
stays cheap. A dense 27B Q4 is ~28–32 before speculation — llama.cpp
already sits on that ceiling. FreeToken's 77–83 tok/s is a
~3B-active MoE on NVIDIA. Design against 640 GB/s and 32 GB, not
against 7900 XTX math or a tweet.

## Process shape

Two programs, one address space of GPU work.

```
┌──────────────────────────┐     token ids      ┌─────────────────────────────┐
│  host process            │ ─────────────────▶ │  token engine (this repo)   │
│  tokenizer, chat, HTTP   │                    │  weights, KV, kernels       │
│  (later; not the score)  │ ◀───────────────── │  sampler                    │
└──────────────────────────┘     token ids      └─────────────────────────────┘
```

The engine accepts token ids and returns token ids. It does not
tokenize, render Markdown, or speak JSON. FlashQwen got this right.
llama2.c got the loop right. vLLM got the zoo wrong for this job.

Rules for the engine process:

- C++. No Python on the hot path. No autograd. No per-op `Tensor`.
- One resident model. One KV arena. A handful of scratch buffers
  allocated at load. `malloc` after `load()` is a bug.
- CPU backend is the correctness oracle and the CI path. HIP is the
  speed path. A Vulkan/RADV backend is a peer if HIP loses decode,
  not a fallback we invent first.
- One architecture family at a time. Qwen3 dense, then that family's
  MoE / hybrid / MTP. A model zoo is how you become llama.cpp.

## Two pipelines, not one forward()

Prefill and decode are different programs that share weights.

| | Prefill | Decode |
| --- | --- | --- |
| Bound | Compute (GEMM, flash-attn) | Memory (weight stream) |
| Batch | Many tokens × one sequence | One token (or a small draft) |
| Kernel | GEMM / WMMA / rocBLAS, flash-attn | Fused quantized GEMV |
| Graph | Optional; shapes change with prompt | Capture the whole step |
| Goal | Low TTFT | Max tok/s |

A training-library `forward()` that does both with the same GEMM is
why the archived Vesper tree cannot win. Decode at M=1 using a tiled
GEMM leaves most of the card idle.

Prefill may chunk a long prompt so working set and kernel shapes stay
in a good range. That is for TTFT, not for the decode score.

## Resident memory

**Weights.** mmap GGUF (Q8_0, then Q4_K). The file is the tensor
store. Convert once into a **decode-friendly layout**: each
projection is a packed row-major matrix the GEMV can stream with
vec8 loads and in-register dequant. Do not keep a training NCHW
tensor and dequant into a workspace every token.

Activations go through a cheap Q8 pass so the huge weight stream
stays narrow (llama.cpp's contract).

**KV.** For one user, one sequence: linear
`[n_layers, max_seq, n_kv_heads, head_dim]`, K and V separate,
written at `pos`. Layout for **decode attention**, not for a
training batched matmul. Quantize KV (Q8, then Q4) when context
is the thing that does not fit, not before.

Paged KV exists only when we have prefix reuse or two sequences.
Until then it is slower and easier to get wrong.

**Scratch.** `x`, residual, Q/K/V, attn, gate/up, logits, scores.
Fixed sizes from `ModelConfig`. Reused every token.

**MoE (later).** Host RAM holds every expert (source of truth).
VRAM holds an LRU of complete `(layer, expert)` slots. GPU
residency is a cache. Evicting never loses correctness.

## The decode step (the product)

One captured HIP graph. Host code between tokens is: feed the next
id (or read it from a device buffer) and replay.

Per token, per layer:

1. RMSNorm of `x` fused into the QKV GEMV.
2. Q, K, V GEMVs from the quantized weights. QK-norm if the model
   wants it. RoPE at `pos`. All in the same launch if occupancy
   allows.
3. Append K, V to the cache at `pos`.
4. GQA decode-attention over `cache[0..pos]`. Split-K / flash-decode
   so the reduction is bandwidth-friendly, not a naive O(seq)
   kernel that thrashes L2.
5. Output projection GEMV + residual.
6. RMSNorm fused into gate/up GEMV, SwiGLU in registers, down GEMV
   + residual.

Then final RMSNorm, lm_head GEMV, sample **on device**, write the
next token id into the graph's input slot. Sync to the host only
when the user needs a streamed token, or every N tokens. One sync
per layer is how you lose.

That graph is static for a given model + max batch (batch=1) +
quant. Context length is a device scalar, not a reason to recapture
every token.

## Kernel inventory

This is the whole GPU product for dense Qwen3. Everything else waits.

| Kernel | Why it exists |
| --- | --- |
| `gemv_q8` / `gemv_q4` | ~90% of decode time. Workgroup = one output row (or a small tile). Wave32 on gfx1201. 256 B aligned packed loads. Several independent FP32 accumulators. Dequant in registers. |
| `rmsnorm_qkv_rope` | Stops three rereads of `x` and three launches. |
| `attn_decode_gqa` | 1×seq attention against the cache. Split-K. |
| `swiglu_down` | Gate/up already produced; this is the fused MLP tail. |
| `gemv_lm_head` | Same GEMV, vocab-wide. |
| `sample_greedy` / `sample_top` | Device-side. Keeps the graph closed. |
| `gemm_prefill` + `attn_prefill` | Separate code. rocBLAS / hipBLASLt / an FP16 WMMA tile sized for 64 KB LDS. Measure flash-attn; do not assume rocWMMA is a win. |

Unfused twins of every fused kernel stay in the tree and are the
CPU/HIP correctness gate. hipEngine is right about that. The first
HIP twins are F32 GEMV / RMSNorm / RoPE / attention / SwiGLU, written
for gfx1201 wave32, in `src/kernels_hip.hip`.

Vulkan/RADV gets the same inventory with different workgroups
(wave64 single-row + subgroup reduce is what beats HIP on llama.cpp
decode). Same math, sibling backend.

## Workgroup is a feature

On RDNA, "more threads" is often slower. llama.cpp HIP has lost
decode to Mesa RADV by launching fat 256-thread blocks with LDS
barriers on small-K matvecs.

The first GEMV is written for **gfx1201**:

- Wave32. Reductions 16…1. `__launch_bounds__` for 32-wide waves.
- One output row per workgroup unless a measured tile wins.
- 256 B alignment on weight rows and KV.
- Grid sized in multiples of 64 CUs as the starting guess.
- `GPU_MAX_HW_QUEUES=1` on the HIP runtime (R9700 idle-power bug).

gfx1100 / gfx1151 get a sibling tune later. Same math, different
occupancy and cacheline.

If our HIP GEMV loses to llama.cpp Vulkan by more than ~15% on the
same Q4-8B on the R9700, we either fix the workgroup or we ship
Vulkan. We do not "add graphs" to a bad GEMV.

## Speculation (dense models)

After the Q4 GEMV is honest, MTP is the only lever that makes a
dense 20B+ look fast. Qwen3.8 already trains MTP heads. The loop:

1. Draft K tokens from the MTP head (cheap).
2. Target verifies in one forward (same weight stream).
3. Accept the longest matching prefix. Exact if we use the
   rejection-sampling correction; also keep a `--strict` greedy
   gate against one-token decode (ds4's lesson).

Draft-model speculation (a second GGUF) is a product feature.
N-gram / prompt lookup is free and wins on copy-heavy work. Do
those after native MTP.

Verification is compute. The win is that one weight stream
produces several accepted tokens. If the GEMV is slow, speculation
multiplies a slow kernel.

## MoE (the FreeToken number)

A different machine, same engine process.

- Host expert banks, one logical `(layer, expert)` id.
- GPU LRU of complete experts. Shared across layers.
- Prefill: double-buffer. While layer `l` runs, stream layer
  `l+1`'s experts over PCIe (or, on Strix Halo, through unified
  memory).
- Decode: hits run on GPU. Of `m` unique misses, fill
  `q* ≈ m * B_P / B_H` over the interconnect and run the rest on
  the CPU from host RAM, overlapped. Profile `B_P` and `B_H` on
  the machine (`ft bench bw`). Merge partial MoE sums exactly.
- Correctness gate: hybrid CPU+GPU output matches fused-GPU
  output.

Do not pin a static "hot expert" set. Routing moves. LRU follows
the router.

This path is how you serve a 35B-A3B or a 284B on a 24 GB card
at interactive speed. It does nothing for a dense 27B that already
fits.

## Hybrid attention (Qwen3.8)

Qwen3.8 is not "Qwen3 with more layers." It interleaves Gated
DeltaNet with gated full attention and ships MTP. The extra
state is a recurrent tensor per GDN layer, not just KV.

Keep a small pool of those states checkpointed at **semantic
anchors** (thinking / tool / turn tokens). Agentic harnesses
edit those blocks. After an edit, resume from the last surviving
anchor and re-prefill only the suffix. FreeToken's paper is the
reference. Do this after dense GQA decode is boring.

## What the host is allowed to do

At load: parse GGUF, mmap, optionally repack rows, allocate KV
and scratch, compile/load kernels, capture the decode graph,
run a correctness probe (tiny prompt, compare to CPU oracle).

At generate: write prompt ids, run prefill, replay the decode
graph, occasionally sync a token for streaming.

It is not allowed to: allocate, walk an autograd graph, launch
a Python callback per layer, copy the full logits to host unless
sampling is on host (it should not be), or recapture the graph
every token.

## Correctness is part of speed

A fast wrong kernel is a wasted week. Gates, in order:

1. Unfused CPU == fused CPU.
2. CPU == HIP on `tiny_demo` and Qwen3-0.6B.
3. Cached decode logits == full-sequence attention over the
   same written KV.
4. Greedy continuation: generate N, restart with N-1, next
   token matches.
5. MTP `--strict` equals one-token decode.
6. Hybrid MoE equals fused MoE.

No tok/s claim without a gate and a roofline printout
(bytes/token, achieved GB/s, llama.cpp HIP and Vulkan on the
same box).

## What I would refuse

- A PyTorch-compatible Tensor API. That was the last repo.
- Starting from vLLM, SGLang, or a hipify of ExLlama / FlashInfer.
- Continuous batching and paged attention as the first feature.
- A 200-model zoo.
- NVFP4, Triton flash-attn, and other NVIDIA-only paths as
  "the" design.
- An HTTP server before the decode GEMV beats or matches
  llama.cpp on one card, one quant, one model.
- Any 2% trick that costs 500 lines (llm.c's rule).

## What I would ship, in order

1. CPU-correct Qwen3 dense loop — **this tree**.
2. GGUF/safetensors load. F32 HIP GEMV + RMSNorm + RoPE, gated
   against CPU.
3. Q8 GEMV with the gfx1201 wave32 workgroup. If it is slower than
   F16, the inner loop is wrong.
4. Fuse norm+QKV+RoPE and SwiGLU. HIP-graph the decode step.
5. Qwen3-8B Q8/Q4 vs llama.cpp HIP and Vulkan. Publish GB/s.
6. Q4_K GEMV. This is the product quant.
7. Native MTP on models that have heads.
8. MoE host pool + LRU + q*.
9. Gated DeltaNet + semantic anchors.
10. Vulkan peer backend if HIP still loses decode.

Steps 2–6 are a local engine that can honestly talk about tok/s.
Steps 7–8 are how you look like FreeToken. Step 9 is Qwen3.8.
Step 10 is admitting the chip's decoder may prefer RADV.

That is the design. The score is achieved GB/s times tokens per
stream, on one GPU, on one model we can reproduce.
