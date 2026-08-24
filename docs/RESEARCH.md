# Research: local LLM inference, NVIDIA-only speed, AMD reality

Date: 2026-08-24.

The old Vesper tree was a PyTorch-like training library. That is the wrong
shape for a local inference engine. This note records what
[FreeToken](https://github.com/FlashML-org/FreeToken) actually does, why
its numbers do not transfer to a dense model on AMD, and where this
rewrite should compete.

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

## The project: FreeToken

[FreeToken](https://github.com/FlashML-org/FreeToken) (FlashML, UC Berkeley /
MIT, [arXiv:2608.16157](https://arxiv.org/abs/2608.16157)) is the engine
behind the Twitter numbers. It is **not** a dense-model kernel race. It is
an edge-native **MoE offload** server: the full expert pool lives in host
RAM, a slice of VRAM is an LRU cache of `(layer, expert)` slots, and each
decode miss is split between PCIe cache-fill and in-place CPU expert
execution.

Install docs are explicit: Linux x86_64, **NVIDIA GPU, driver r580+, CUDA
13**. Hardware list is RTX 30 / 40 / 50. There is no ROCm, HIP, or Radeon
path. Kernels go through FlashInfer, CUDA Graphs, and NVIDIA quants
(NVFP4, MXFP4, FP8). The host is Python on a mini-sglang / vLLM substrate.

Paper numbers (real agent traces, not synthetic `llama-bench`):

| Model | Active / total | Machine | Decode |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | ~3B / 35B | RTX 5090 32 GB | 77–83 tok/s |
| Qwen3.6-35B-A3B NVFP4 | ~3B / 35B | RTX 4060 laptop 8 GB | 39 tok/s |
| DeepSeek-V4-Flash | 13B / 284B | RTX 5090 | 22–25 tok/s |
| GLM-5.2 NVFP4 | 40B / 753B | RTX PRO 6000 96 GB | ~15 tok/s, 2× llama.cpp |

Those are **sparse** numbers. Each token streams a few experts, not 35B
or 284B of weights. A dense Qwen3.8-27B on the same 5090 is a different
roofline. FreeToken *can* load Qwen3.6-27B dense (`fused` backend, all
weights in VRAM); that is not what the viral posts are measuring.

The actual inventions, in order:

1. **Host expert pool is the source of truth.** GPU residency is a cache.
   Evicting an expert never loses correctness.
2. **Prefill double-buffering.** Prefill activates almost every expert.
   While layer `l` runs, layer `l+1`'s full expert set streams over PCIe.
3. **Shared LRU expert cache for decode.** Routing is temporally local.
   Most hits stay in VRAM. Residual misses go to the next policy.
4. **\(q^\star\) miss split.** Profile two bandwidths on the machine:
   PCIe expert-copy \(B_P\) and CPU expert-kernel \(B_H\). Of \(m\)
   unique misses, fill \(q^\star \approx m\,B_P/B_H\) over PCIe and
   compute the rest on the CPU from host RAM, overlapped. Merge the
   partial MoE sums. `ft bench bw` is this calibration.
5. **Semantic anchors.** Hybrid-attention models (Gated DeltaNet in
   Qwen3.5/3.6/3.8) compress history into a recurrent state. Agentic
   harnesses delete thinking/tool blocks. Checkpoints at those special
   tokens let the engine re-prefill only the new suffix.
6. **Elastic VRAM.** Resize the expert-cache vs KV split at a scheduler
   safe point without restarting or reloading host weights.

Backends: `fused` (all experts on GPU), `offload` (LRU + PCIe), `cpu`
(misses stay on CPU), `hybrid` (the \(q^\star\) split), `auto`.

Why we cannot run it on AMD as-is: CUDA Graph capture of a device-side
LRU (no host sync per layer), FlashInfer, NVFP4 tensor-core kernels,
and `cudaHostRegister` DMA. hipify does not produce that stack.

What we should copy on AMD: the *machine model* (GPU + CPU + host RAM
as one platform) and the \(q^\star\) policy. Radeon boxes often have
strong host bandwidth (7900 XTX desktops, Strix Halo unified memory)
and weaker or quirkier GPU decode than NVIDIA. That is exactly the
regime FreeToken's hybrid path is for — once the kernels are HIP.

## Other projects

| Name | What it is | GPU | Notes |
| --- | --- | --- | --- |
| [TokenSpeed](https://github.com/fw-ai/tokenspeed) | Production agentic engine | NVIDIA (Blackwell/Hopper) | 580 tok/s on Qwen3.5-397B-A17B *agentic MoE*. Not consumer Radeon. |
| [FlashQwen](https://github.com/frankkk96/FlashQwen) | From-scratch C++/CUDA Qwen3-8B | NVIDIA sm_89+ | Small, educational. No AMD. |
| [Fastgen](https://github.com/facebookresearch/fastgen) | Mini-vLLM (~3k LoC) | NVIDIA | Paged attention, CUDA graphs. |
| [FastLLM](https://github.com/ztxz16/fastllm) | C++ engine, own operators | NVIDIA + ROCm | Real AMD support. Not RDNA3-tuned. |
| [hipEngine](https://github.com/shisa-ai/hipengine) | ROCm-native host + HIP kernels | gfx1100 / gfx1151 | Closest AMD analog. GGUF + ParoQuant, MTP. AGPL. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF runtime | HIP, Vulkan, CUDA, Metal | Practical AMD baseline. On RDNA3, **Vulkan/RADV often beats HIP** on decode. |
| vLLM / SGLang | Production servers | ROCm; best on MI300 | FreeToken's substrate. Heavy. RDNA3 is a second-class kernel path. |

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

From **FreeToken**: treat GPU + CPU + host RAM as one platform. For MoE,
keep the expert pool on the host and put an LRU of complete experts in
VRAM. Split decode misses with a measured \(q^\star = m B_P / B_H\).
Double-buffer prefill at layer granularity. Checkpoint hybrid-attention
state at thinking/tool boundaries. Do not pin a static "hot expert" set.

From **TokenSpeed / Fastgen / FlashQwen**: keep the host small. Scheduler,
KV ownership, and kernel dispatch should stay readable. Do not grow a
training framework.

The landscape (vLLM, llama.cpp, SGLang, ExLlama, hipEngine, and the
small engines) is [ENGINES.md](ENGINES.md). The small / from-scratch
notes and the three Vesper templates are
[SMALL_ENGINES.md](SMALL_ENGINES.md).

From **Qwen3.8 community kits**: MTP is the only realistic way to look
"hundreds of tok/s" on a *dense* 20B+ model on consumer AMD. FreeToken's
77–83 tok/s is the other route: a 3B-active MoE plus hybrid offload.

## Non-goals for v0

- Porting FreeToken or matching it on RTX 5090
- Training, autograd, or a PyTorch-compatible Tensor API
- Multi-GPU
- Vision / multimodal
- Shipping an OpenAI HTTP server before the decode loop is honest

## Competitive position

```
                    NVIDIA only          AMD first
edge MoE            FreeToken            (this, later)
small host          FlashQwen            this project (now)
                    Fastgen
production          TokenSpeed           hipEngine (AGPL)
                    TRT-LLM              vLLM-ROCm (MI300)
format zoo          llama.cpp CUDA       llama.cpp HIP/Vulkan
```

Vesper's opening is: MIT, C++-only, HIP-first, CPU-correct, aimed at
RDNA3/RDNA3.5 local boxes. First a dense Qwen3 loop we can prove, then
HIP GEMV, then a FreeToken-style expert cache on AMD. We will lose to
llama.cpp on model coverage for a long time. We should try to beat it
on one architecture, one quant, one GPU class, with numbers we can
reproduce.

What to implement, in what order, and which AMD traps are real:
[TOKS.md](TOKS.md).
