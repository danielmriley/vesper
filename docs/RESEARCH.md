# Research: local LLM inference, NVIDIA-only speed, AMD reality

Date: 2026-08-24.

The old Vesper tree was a PyTorch-like training library. That is the wrong
shape for a local inference engine. This note records what the fast projects
actually do, what works on AMD, and where this rewrite should compete.

The library snapshot lives on `cursor/clearmain-47fb`.

## What the Twitter numbers are measuring

Decode speed on a single GPU is almost always memory-bandwidth bound:

```
tok/s ≈ memory_bandwidth / bytes_touched_per_token
```

For a dense model in decode, each new token streams the full weight set plus
the growing KV cache. A 27B model at 4-bit is ~14–16 GiB of weights. On an
RX 7900 XTX (24 GiB, ~960 GB/s advertised) the roofline for Q4 decode is
roughly 50–70 tok/s before speculation, kernel fusion, or KV compression.
Hundreds of tok/s on a 27B-class dense model therefore require at least one
of:

- a much smaller *active* parameter count (MoE)
- multi-token prediction / speculative decoding that accepts several tokens
  per weight stream
- a much fatter GPU (H100/B200-class bandwidth)
- a smaller model than the headline name implies

Prefill is compute-bound and can look like thousands of tok/s. Marketing
posts mix the two. This project should always report **prefill tok/s**,
**decode tok/s**, **accepted tok/s with MTP**, model, quant, context, and
GPU.

## Projects people are posting

"FastToken" does not resolve to a single public repository. The closest
matches to the posts are:

| Name | What it is | GPU | Notes |
| --- | --- | --- | --- |
| [TokenSpeed](https://github.com/fw-ai/tokenspeed) | Production agentic engine (LightSeek / vLLM partner) | NVIDIA (Blackwell/Hopper first) | 580 tok/s on Qwen3.5-397B-A17B *agentic* workload. MLA kernels, fused prefill/decode. MI350 mentioned as future work, not a consumer Radeon path. |
| [FlashQwen](https://github.com/frankkk96/FlashQwen) | From-scratch C++/CUDA Qwen3-8B engine | NVIDIA sm_89+ | Small, educational, paged KV + continuous batching. No AMD. |
| [Fastgen](https://github.com/facebookresearch/fastgen) | Mini-vLLM (~3k LoC) | NVIDIA | Paged attention, CUDA graphs, chunked prefill. |
| [FastLLM](https://github.com/ztxz16/fastllm) | C++ engine, own operators | NVIDIA + ROCm + others | Real AMD support. Broad model coverage. Not RDNA3-tuned. |
| [hipEngine](https://github.com/shisa-ai/hipengine) | ROCm-native host + hand-tuned HIP | gfx1100 / gfx1151 | Closest analog to what we want. Torch-free hot path, GGUF + ParoQuant, MTP. AGPL. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF runtime | HIP, Vulkan, CUDA, Metal | The practical AMD baseline. On RDNA3, **Vulkan/RADV often beats HIP** on decode. |
| vLLM / SGLang | Production servers | ROCm works; best on MI300 | Continuous batching. Heavy. Consumer RDNA3 is a second-class kernel path (Triton flash-attn issues on gfx1100). |

TokenSpeed and FlashQwen are the "hundreds of tok/s on Qwen" posts. They
are NVIDIA-shaped: CUDA graphs, Tensor Cores / Blackwell MMA, FlashInfer,
and PTX-level kernels. Porting that stack to HIP is a multi-month rewrite,
not a hipify.

## AMD-capable stacks in 2026

ROCm is usable now. It is still not CUDA.

**llama.cpp** is the default local path. Build with `-DGGML_HIP=ON` and
`GPU_TARGETS=gfx1100` (or `gfx1151`). On RX 7900 XTX, community benches
still show Vulkan (Mesa RADV, wave64 cooperative matrix) winning decode
against ROCm HIP on several Q4 models. HIP often wins long prefill. Any
honest AMD engine must treat HIP and a Vulkan/RADV backend as peers, not
assume ROCm is fastest.

**hipEngine** (Shisa, 2026) is the project that matches the goal: HIP-first
kernels written for RDNA3 wave32, vec8 FMA, and the real cache hierarchy;
CPU reference kernels as the correctness oracle; evidence-backed benches
with KL / top-1 gates. Published W7900 numbers for Qwen3.6-35B-A3B
ParoQuant W4: ~2900 prefill tok/s and ~116 decode tok/s (512 in / 128 out).
That is a 3B-*active* MoE, not a 27B dense stream. Dense 27B on the same
card is in the 30–40 tok/s band even with DFlash speculation.

**q38rocm** (community Qwen3.8-27B kits) combine ROCmFP4, MTP speculation,
TurboQuant KV, and Mesa RADV wave64 on Strix Halo. Reported 30–36 tok/s on
a 128 GB Ryzen AI Max+ 395. That is the real dense-27B number on unified
memory, not hundreds.

**vLLM / SGLang on RDNA3** work if you build against a matching TheRock /
ROCm stack and often disable Triton flash-attn
(`VLLM_USE_TRITON_FLASH_ATTN=0`). They win multi-user throughput. They are
the wrong starting point for a from-scratch engine.

## Qwen3.8 is not Qwen3

Qwen3.8-27B (August 2026, `Qwen/Qwen3.8-27B`, Apache-2.0) is a *dense*
hybrid multimodal model, not a small 3.8B and not a MoE:

- 64 layers, hidden 5120, vocab 248320
- 48 Gated DeltaNet (linear attention) + 16 Gated Attention layers
- GQA on the full-attention layers (24Q / 4KV)
- native 262K context, YaRN to 1M
- trained with multi-token prediction
- vision encoder on top of the language stack
- BF16 ~56 GiB; Q4 ~14–16 GiB plus KV

The `model_type` is `qwen3_5`. Engines that already run Qwen3.5/3.6 get
day-zero text support. A from-scratch engine that only implements Llama-style
GQA+SwiGLU will not run Qwen3.8 until Gated DeltaNet, gated attention, and
MTP are real.

Qwen3 dense (0.6B / 1.7B / 4B / 8B / 14B / 32B) is the right first target:
RMSNorm, GQA, RoPE, SwiGLU, QK-norm. That family still matters, fits a
24 GiB card at 4-bit, and is the correctness scaffold for 3.8 later.

## Why the old Vesper library should stay archived

The archived tree already had HIP kernels, GQA, RoPE, KV cache, and a
generator. It also had autograd, Adam, conv2d, dropout, training loops, and
a generic Tensor graph. Decode on that path pays for:

- allocator / Tensor metadata on every op
- separate kernel launches for norm, linear, residual
- no weight quantization
- no paged KV
- no HIP graphs
- no speculative decoding

Those are training-library costs. Fast decode is a different program:
one resident model, fused GEMV+norm+RoPE, a KV arena, and a sampling loop
that does not allocate.

## What we should steal (ideas, not code)

From **llama.cpp**: GGUF as the interchange format; mmap weights; a CPU
oracle; graph capture; the empirical fact that workgroup shape beats
"more threads" on RDNA3.

From **hipEngine**: HIP-first kernel trees keyed by
`(backend, layer, quant)`; fused and unfused twins; CPU reference as CI;
never claim a tok/s number without a correctness gate.

From **TokenSpeed / Fastgen / FlashQwen**: keep the host small. Scheduler,
KV ownership, and kernel dispatch should stay readable. Do not grow a
training framework.

From **Qwen3.8 community kits**: MTP is the only realistic way to look
"hundreds of tok/s" on a consumer AMD card with a dense 20B+ model. The
draft head is part of the model card, not an optional trick.

## Non-goals for v0

- Matching TokenSpeed on Blackwell
- Training, autograd, or a PyTorch-compatible Tensor API
- Multi-GPU
- Vision / multimodal
- Shipping an OpenAI HTTP server before the decode loop is honest

## Competitive position

```
                    NVIDIA only          AMD first
small host          FlashQwen            this project
                    Fastgen
production          TokenSpeed           hipEngine (AGPL)
                    TRT-LLM              vLLM-ROCm (MI300)
format zoo          llama.cpp CUDA       llama.cpp HIP/Vulkan
```

Vesper's opening is: MIT, C++-only, HIP-first, CPU-correct, aimed at
RDNA3/RDNA3.5 local boxes, starting with Qwen3 dense and aimed at Qwen3.8
hybrid + MTP. We will lose to llama.cpp on model coverage for a long time.
We should try to beat it on one architecture, one quant, one GPU class,
with numbers we can reproduce.
