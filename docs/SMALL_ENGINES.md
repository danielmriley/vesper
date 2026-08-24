# Small and clever LLM inference engines

Survey date: 2026-08-24.

Vesper is a MIT C++ HIP-first engine that currently has a CPU-correct Qwen3
dense loop (RMSNorm, GQA, RoPE, SwiGLU, QK-norm, linear KV). This note
covers the small / from-scratch / educational engines that stay fast
despite low LoC, plus a few products that hide one useful trick.

LoC numbers are author claims or repo-scale estimates, not `cloc` of a
frozen checkout. "Product" means people run it as a server. "Technique
demo" means you read it for one idea.

## Why low LoC can still be fast

Decode is memory-bandwidth bound:

```
tok/s ≈ bandwidth / bytes_touched_per_token
```

A small engine is fast when it does not waste those bytes or those
launches. The recurring bag of tricks:

- **Paged KV.** Block tables instead of one contiguous `[seq, heads, dim]`
  tensor per request. Lets you pack many sequences without padding, and
  recycle blocks. [vLLM paper](https://arxiv.org/abs/2309.06180).
- **CUDA / HIP graphs.** Capture the decode step once, replay it. Kills
  launch overhead on the 1-token path.
- **Fused GEMV + norm + RoPE.** One launch streams weights once. Separate
  RMSNorm / QKV / RoPE kernels re-read `x` three times.
- **Chunked prefill / token-budget scheduler.** Mix a slice of a long
  prompt with decode tokens so the GPU stays at a fixed batch size.
- **Prefix / radix cache.** Content-hash KV blocks and reuse shared
  system prompts.
- **Expert cache.** For MoE, VRAM holds an LRU of complete experts. Host
  RAM or SSD is the source of truth.
- **Speculation (Medusa / EAGLE / MTP).** Several accepted tokens per
  weight stream. The only way a *dense* 20B+ model looks "hundreds of
  tok/s" on a consumer card.

A 1.2k-line Python engine that calls FlashAttention, cuBLAS, and
`torch.compile` is not "1.2k lines of GPU work." It is 1.2k lines of
*ownership*: who owns KV, who schedules, who captures the graph.

---

## facebookresearch/fastgen

https://github.com/facebookresearch/fastgen

- **LoC:** ~3k Python (author claim). Repo is tiny: `cache.py`,
  `forward.py`, `generate.py`, `model.py`, a `kernels/` dir.
- **GPU:** NVIDIA. PyTorch CUDA. No ROCm story.
- **Kind:** library you import into an RL loop or data pipeline, plus
  `fgchat` / `fgserve`. Technique-complete mini-vLLM, not a product
  surface.

Architecture:

- Batched decode with **paged attention** and **chunked prefills**.
- **CUDA graphs** on the decode step.
- **Host-side KV cache** as well as device pages. Useful when VRAM is
  the constraint and host RAM is cheap.
- Tensor parallelism across H100s.
- CPU/GPU profiling hooks baked in.

**The one trick.** It is the inventory of a modern serving engine
without the serving engine. They match vLLM 0.8.4 throughput on Mistral
7B / Qwen2.5 14B / Llama 70B (H100) because they kept the *same*
techniques and deleted everything else.

Do not confuse this with Microsoft DeepSpeed-FastGen (Dynamic
SplitFuse). Different project.

---

## frankkk96/FlashQwen

https://github.com/frankkk96/FlashQwen

- **LoC:** Stage 1 is **under 2000 lines of C++/CUDA**, zero deps (no
  PyTorch, no cuBLAS, no HF tokenizers). Later stages grow a Go gateway
  and pull in CUTLASS / cuBLAS / FlashAttention-2.
- **GPU:** NVIDIA, default `sm_89` (RTX 4090). No AMD.
- **Kind:** Stage 1 is a teaching engine. Stage 3 is a single-model
  serving runtime that claims 95–98% of vLLM throughput on Qwen3-8B
  bf16, one 4090. Product-shaped, one architecture only.

Architecture (Stage 2+):

- **Inner C++/CUDA token engine.** Token ids in, token ids out. Owns
  weight load, prefill, decode, sampling, paged KV, continuous
  batching. Knows nothing about text.
- **Outer Go process.** Tokenizer, ChatML, tool calls, OpenAI HTTP.
  Embeds the engine binary and talks gRPC.
- Evolution they actually measured: scalar matmul → WMMA → FlashDecoding
  split-K → INT8 → bf16+cuBLAS → fused QKV/gate-up GEMM → GQA-shared
  FlashDecoding → `mma.sync` prefill attention → CuTe/CUTLASS →
  **decode CUDA-graph + async scheduling**.
- Token-budget scheduler (chunked prefill by another name). Automatic
  prefix caching via content-hashed KV.

**The one trick.** They treated one model (Qwen3-8B) as a closed world
and closed the gap to vLLM *step by step*, with a bench after every
commit. The host stays small because the engine never grew a model zoo.

This is the closest C++ twin of Vesper's current tree.

---

## nano-vllm / "nano-vllm-amd"

https://github.com/GeeeekExplorer/nano-vllm

- **LoC:** ~1,200 Python (author claim). Layout: `engine/`, `layers/`,
  `models/`.
- **GPU:** NVIDIA via PyTorch. CUDA graphs, `torch.compile`, optional
  FlashAttention / FlashInfer. There is **no** well-known
  `nano-vllm-amd` repo as of this survey. On AMD it only runs if your
  PyTorch is a ROCm build, and then the CUDA-graph / FlashInfer path is
  the usual second-class HIP story. Closest *from-scratch C++/HIP*
  sibling is [jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm)
  (`-DUSE_HIP=ON`, `cuda_to_hip.h`).
- **Kind:** educational engine that happens to match vLLM throughput on
  Qwen3-0.6B / RTX 4070 Laptop in their bench. Not a product.

Architecture:

- Offline `LLM.generate` API that mirrors vLLM.
- Paged KV + continuous batching + prefix caching.
- Tensor parallelism (basic).
- CUDA graph + `torch.compile` for decode.

**The one trick.** The scheduler and block table, written so you can
read them in an afternoon. Speed comes from calling the same vendor
kernels vLLM uses, not from 1,200 lines of GEMM.

---

## mini-sglang

https://github.com/sgl-project/mini-sglang
https://github.com/sgl-project/mini-sglang/blob/main/docs/structures.md

- **LoC:** ~5,000 Python (author claim).
- **GPU:** NVIDIA only. Depends on `sgl-kernel` and FlashInfer. Linux
  x86_64 / aarch64.
- **Kind:** readable SGLang. Capable enough to bench against full
  SGLang. Still a teaching / research engine.

Architecture:

- Process split: API server, tokenizer, detokenizer, one scheduler
  per TP rank. ZMQ for control, NCCL for tensors.
- **Radix cache** (prefix tree over KV blocks) plus a naive cache.
- Chunked prefill.
- **Overlap scheduling:** CPU prepares the next batch while the GPU
  runs the current one.
- FlashAttention / FlashInfer backends. CUDA graph replay inside
  `Engine`.

**The one trick.** Overlap scheduling plus radix cache, in 5k lines.
That is the SGLang thesis, stripped of the product.

---

## antirez ds4 (DwarfStar)

https://github.com/antirez/ds4

- **LoC:** started as a narrow C engine. The tree is now a full
  product (Metal / CUDA / ROCm, HTTP server, agent, GGUF tools,
  QA fixtures). Tens of thousands of lines, still one-model-family
  shaped. MIT, with GGML copyright retained for quant tables.
- **GPU:** Metal first (96 GB+ Macs). CUDA including multi-GPU / DGX
  Spark. **ROCm on Strix Halo** (`make strix-halo`). Real AMD path.
- **Kind:** product for DeepSeek V4 Flash / PRO and GLM 5.2. Not a
  general GGUF runner.

Architecture:

- Custom GGUF layout. Routed MoE experts quantized hard (IQ2_XXS /
  Q2_K); shared experts, projections, routing left at higher precision.
- **SSD streaming expert cache.** Non-routed weights stay resident.
  Routed experts live in an in-memory cache, filled from the GGUF on
  miss. Prefill overlaps two full routed layers. Decode is miss-
  sensitive.
- Graph backend on Metal / CUDA / ROCm.
- **DSpark** speculative decoding: a 5.6 GiB support GGUF reads hidden
  states and proposes up to five tokens. Target verifies; accepted
  prefix keeps the verifier state. Confidence threshold prunes dead
  suffixes. Replaces the older one-stage MTP file for Flash.
- GLM has a native MTP block (`--glm-mtp`). Experimental.
- Disk KV for long context. Tensor / pipeline parallelism, including
  two-Mac RDMA.

**The one trick.** Treat host RAM / SSD as the expert pool and VRAM as
a cache, then add a model-native speculator (DSpark / MTP). That is
the FreeToken machine model, written in C, with a HIP backend already
on Strix Halo.

Published q2 numbers (2048-token context, 128 greedy tokens): M5 Max
128 GB Metal 39 tok/s decode; DGX Spark GB10 CUDA 18 tok/s. These are
*sparse* 13B-active-class numbers, not dense 27B.

---

## karpathy llama2.c / llm.c

https://github.com/karpathy/llama2.c
https://github.com/karpathy/llm.c

- **LoC:** `run.c` is ~700 lines of dependency-free C (llama2.c).
  `train_gpt2.c` is ~1,000 lines of CPU fp32 (llm.c). The CUDA
  training loop is larger and uses cuBLAS / cuDNN.
- **GPU:** llama2.c is CPU (OpenMP). llm.c training is NVIDIA CUDA.
  Community HIP ports exist; they are not the mainline.
- **Kind:** teaching artifacts. llama2.c inferences. llm.c pretrains.

Architecture (llama2.c):

- Hard-coded Llama 2: RMSNorm, RoPE, MHA/GQA, SiLU SwiGLU, linear KV.
- One `Transformer` struct, one `forward()`, malloc'd scratch.
- fp32 `.bin` weights. No paged KV, no graphs, no quant in the
  original file.

**The one trick.** There is no framework. The decode loop *is* the
program. Compiler flags (`-Ofast`, OpenMP) are the only "runtime."
That is why Vesper's CPU oracle looks like this and should stay
looking like this.

llm.c's useful lesson for inference is negative: they refuse a 2%
speedup that costs 500 lines. Keep the hot path short.

---

## candle / mistral.rs / burn

Three Rust stacks, not one engine.

### huggingface/candle

https://github.com/huggingface/candle

- **LoC:** framework (candle-core + kernels + transformers). Tens of
  thousands.
- **GPU:** CUDA (plus cuDNN, flash-attn crate, NCCL). CPU with MKL /
  Accelerate. WASM. **No first-class ROCm.**
- **Kind:** tensor library whose goal is "small binary, no Python."
  Examples are inference demos, not a serving engine.

**The one trick.** GGUF-compatible quant types and a tiny runtime, so
you can ship a Llama binary without libtorch.

### EricLBuehler/mistral.rs

https://github.com/EricLBuehler/mistral.rs

- **LoC:** full product on top of Candle.
- **GPU:** CUDA (FlashAttention v2/v3, paged attention) and Metal.
  Prebuilt install script. **Not a HIP-first AMD engine.**
- **Kind:** product. OpenAI + Anthropic APIs, web UI, agents, ISQ /
  GGUF / UQFF, LoRA, continuous batching, prefix cache.

**The one trick.** In-situ quantization plus paged attention in a
single Rust binary. Fast because Candle kernels + FA3, not because
the repo is small.

### tracel-ai/burn (+ burn-lm)

https://github.com/tracel-ai/burn
https://github.com/tracel-ai/burn-lm

- **LoC:** framework. `burn-lm` is a small Llama-family inference
  shell (Llama 3 / 3.1 / 3.2, TinyLlama).
- **GPU:** CUDA, **ROCm**, Metal, Vulkan, WebGPU. Backend is a trait.
  Fusion decorator is on by default.
- **Kind:** training + inference framework. burn-lm is an early
  product demo.

**The one trick.** One model IR, many backends, including ROCm and
Vulkan. That is portable. It is also the opposite of Vesper's
"one architecture, one quant, one GPU class."

---

## tinygrad

https://github.com/tinygrad/tinygrad

- **LoC:** the whole stack (tensor, IR, compiler, JIT, nn) is
  intentionally small. Not an LLM serving engine.
- **GPU:** first-class **AMD** (`ops_amd.py`), plus CUDA, NV, Metal,
  OpenCL, QCOM, WebGPU. A new backend needs ~25 low-level ops.
- **Kind:** compiler / training framework. tiny corp's own stack.

Architecture:

- Lazy tensors. Fusion is the default (a matmul+reduce becomes one
  kernel).
- `TinyJit` captures and replays.
- BEAM search over kernel schedules.

**The one trick.** The compiler *is* the optimization. You write
PyTorch-shaped Python and get a fused kernel on RDNA. Useful if
Vesper ever wants a kernel generator. Wrong shape for the decode
loop we actually ship.

---

## airllm

https://github.com/lyogavin/airllm

- **LoC:** small Python wrapper around HuggingFace transformers.
- **GPU:** NVIDIA CUDA, plus a Mac MLX path. CPU path exists. No
  HIP-first kernels.
- **Kind:** productized layer-streaming hack. "70B on 4 GB" is the
  pitch.

Architecture:

- Split the HF checkpoint layer-wise on disk.
- Keep **one layer** (or one routed expert, for sparse MoE) on GPU.
- Prefetch the next layer while computing the current one.
- Optional 4/8-bit *weight-only* block quantization so the PCIe /
  disk load is smaller. Activations stay higher precision.

**The one trick.** VRAM is a sliding window over layers, not a
working set of the whole model. Correct, and slow. Decode still
streams every weight of every layer from host or disk, so tok/s is
PCIe/SSD bound. Fine for "does it run." Useless as a speed
template. FreeToken's expert LRU is the same *idea* at a better
granularity (experts, not whole layers).

---

## exo

https://github.com/exo-explore/exo

- **LoC:** Python product (dashboard, libp2p, MLX distributed).
- **GPU:** Apple Silicon via **MLX**. Linux is CPU-only in the
  README; GPU Linux is "under development."
- **Kind:** product. Cluster your Macs.

Architecture:

- Auto device discovery. Topology-aware pipeline or tensor parallel.
- Day-0 RDMA over Thunderbolt 5.
- OpenAI / Claude / Ollama APIs.

**The one trick.** Treat a Thunderbolt mesh as one GPU. Wrong
problem for a single Radeon box.

---

## TabbyAPI

https://github.com/theroyallab/tabbyAPI

- **LoC:** FastAPI server, not the kernel stack.
- **GPU:** NVIDIA Ampere+ via [ExllamaV3](https://github.com/turboderp-org/exllamav3).
  Exl2 is frozen on a side branch. No AMD.
- **Kind:** product. Official API for Exllama. AGPL. Authors say it
  is not for production-scale serving.

Architecture:

- Exl3 (and fp16/bf16) weights.
- Continuous batching + paged attention (Ampere+).
- Draft-model speculative decoding.
- OpenAI API, tool calling, embeddings, Horde.

**The one trick.** They did not write an engine. They wrote a thin
server on top of the fastest single-GPU NVIDIA kernel stack in the
hobbyist world (Exllama). Speed is Exllama's fused decode, not
Tabby's Python.

---

## aphrodite-engine (now Sonar)

https://github.com/PygmalionAI/aphrodite-engine
https://sonar.dphn.ai/
https://github.com/dphnAI/sonar

- **LoC:** vLLM-scale. The project rebranded to **Sonar**;
  `aphrodite serve` is still the CLI.
- **GPU:** NVIDIA CUDA, **AMD ROCm**, CPU, Apple Metal, Intel XPU,
  TPU. Installer covers Linux x86_64 ROCm.
- **Kind:** product. Serves Dolphin / Pygmalion traffic.

Architecture:

- vLLM fork: continuous batching, paged KV, prefix cache on by
  default, TP/PP/DP/EP, PD disaggregation.
- Extra quants, sampling, and APIs the roleplay community wanted.
- Speculative decoding: **MTP, EAGLE, DSpark, DFlash, n-gram**,
  draft models.

**The one trick.** A vLLM that kept older GPUs and weird quants
alive, then grew a speculator zoo (including DSpark, same family as
ds4). Too big to copy. The speculator menu is the useful catalog.

---

## LoRAX

https://github.com/predibase/lorax
https://loraexchange.ai/guides/speculative_decoding/

- **LoC:** HuggingFace TGI 0.9.4 fork plus Punica SGMV kernels.
  Product-sized.
- **GPU:** NVIDIA Ampere+. No AMD.
- **Kind:** product. Multi-LoRA serving.

Architecture:

- One base model resident. Adapters load just-in-time from HF /
  disk, prefetch/offload GPU↔CPU.
- Heterogeneous continuous batching: different adapters in one
  batch via **SGMV** (Punica).
- Paged attention, flash-attention, TP, GPTQ/AWQ/bnb.
- Speculation: Medusa adapters per request, or prompt-lookup
  n-grams.

**The one trick.** SGMV so 1,000 LoRAs share one GEMM. Irrelevant
until Vesper has a base model people fine-tune.

---

## lmdeploy / TurboMind

https://github.com/InternLM/lmdeploy
https://lmdeploy.readthedocs.io/en/latest/inference/turbomind.html
https://arxiv.org/abs/2508.15601

- **LoC:** production C++ engine (TurboMind, from FasterTransformer)
  plus a Python engine for experiments.
- **GPU:** NVIDIA first (wheels are CUDA 12.8). PyTorch engine also
  on Huawei Ascend. Not a consumer-Radeon project.
- **Kind:** product. InternLM's serving stack.

Architecture (TurboMind):

- **Persistent batch** (continuous batching with N slots).
- **KV cache manager as an LRU of KV caches.** Evicted sequences
  collapse to token ids and re-prefill on miss. "Infinite" device
  memory from the API's point of view.
- Blocked KV, dynamic split-and-fuse, INT8 KV, W4A16 AWQ, prefix
  cache.
- Cutlass FMHA with mismatched Q/K lengths (multi-turn cache hit
  skips context decode). Indirect pointers for discontinuous pages.
- PyTorch engine: CUDA graphs, DeepSeek MTP, EAGLE-3.

**The one trick.** The KV *cache of caches*. Multi-turn chat does
not re-prefill on a hit. Combined with split-and-fuse, that is why
they publish 1.8× vLLM request throughput on some InternLM sizes.

---

## PowerInfer / PowerInfer-2

https://github.com/SJTU-IPADS/PowerInfer
https://arxiv.org/abs/2312.12456
https://arxiv.org/abs/2406.06282
https://powerinfer.ai/v2/

- **LoC:** llama.cpp / ggml fork, plus predictors. PowerInfer-2 adds
  ~12k lines on top (paper).
- **GPU:** NVIDIA CUDA, **AMD ROCm/HIP** (`-DLLAMA_HIPBLAS=ON`).
  PowerInfer-2 is smartphone XPUs (Qualcomm).
- **Kind:** research engine + sparse models. Needs ReLU / ReGLU /
  squared-ReLU (or TurboSparse) activations. Will not run stock
  Qwen3 SwiGLU.

Architecture:

- Profile neuron activations. A small "hot" set lives on GPU. Cold
  neurons stay on CPU.
- Online predictor decides which neurons fire this token. Skip the
  rest (neuron-aware sparse GEMV).
- PowerInfer-2 adds a neuron cluster I/O pipeline so a 47B sparse
  Mixtral-class model runs from phone flash at 11.68 tok/s.

**The one trick.** Activation locality: most FFN neurons are dead
on any given token. GPU holds the hot set, CPU computes leftovers,
**no PCIe of cold weights.** This is the ancestor of FreeToken's
expert cache, at neuron granularity instead of expert granularity.
It requires a model you sparsified. Qwen3 is not that model.

---

## Speculative decoding: Medusa, EAGLE, MTP

These are techniques, not engines. Every serious stack now has at
least one. Papers:

- Speculative decoding: https://arxiv.org/abs/2211.17192
- Medusa: https://arxiv.org/abs/2401.10774
- EAGLE: https://arxiv.org/abs/2401.15077
- EAGLE-2: https://arxiv.org/abs/2406.16858
- EAGLE-3: https://arxiv.org/abs/2503.01840
- DeepSeek-V3 MTP: https://arxiv.org/abs/2412.19437
- NVIDIA overview: https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/

Shared loop:

1. A cheap drafter proposes K tokens.
2. The target verifies them in **one** forward (tree or chain
   attention over the draft).
3. Accept the longest prefix that matches the target. Sample the
   first rejection from the target. Output distribution stays
   exact if you do the rejection-sampling correction.

| Method | Drafter | Extra weights | Typical gain | Notes |
| --- | --- | --- | --- | --- |
| Draft model | Smaller sibling LM | Yes, a whole model | 1.5–2.5× | Easy. Memory heavy. |
| Prompt lookup / n-gram | CPU string match | No | 1.0–2× on copy-heavy tasks | Code, RAG, editing. |
| **Medusa** | K linear heads on the last hidden | Small | ~1.8–2.5× | Parallel heads, tree attention. Lower accept than EAGLE. |
| **EAGLE / EAGLE-2 / EAGLE-3** | Tiny autoregressive head that predicts *hidden states*, then tokens | One layer-ish | ~2–4× | EAGLE-3 fuses low/mid/high features and a dynamic draft tree. |
| **MTP** | Heads trained *with* the target (DeepSeek V3, Qwen3.8, GLM 5.2) | Already in the ckpt | ~1.6–1.8× | Flip a flag. No extra training. |
| DSpark / DFlash | DeepSeek V4 auxiliary draft (hidden-state or diffusion block) | ~5.6 GiB support GGUF | workload-dependent | What ds4 ships. Can lose if accept is low. |

**The one trick.** Verification is compute. Decode is bandwidth.
On a 7900 XTX, streaming a 27B Q4 once and accepting 2 tokens is
almost 2× tok/s at the same bandwidth. That is why DESIGN.md says
MTP is the only realistic way a *dense* 20B+ looks fast on
consumer AMD.

Vesper should not build Medusa first. Qwen3.8 and DeepSeek-class
checkpoints already ship MTP heads. Implement draft/verify on
those, with a greedy exactness gate against one-token decode.

---

## Comparison

| Engine | LoC-ish | NVIDIA | AMD | Product or demo |
| --- | --- | --- | --- | --- |
| Fastgen | ~3k Py | yes | no | demo / embeddable lib |
| FlashQwen | <2k → growing C++ | yes (`sm_89`) | no | demo → single-model product |
| nano-vllm | ~1.2k Py | yes | only via ROCm torch | demo |
| mini-sglang | ~5k Py | yes | no | demo |
| ds4 | C, grown large | yes | **ROCm Strix Halo** | product, narrow models |
| llama2.c | ~0.7k C | no | CPU | demo |
| llm.c | ~1k C + CUDA | train | no | demo |
| candle | framework | CUDA | no | library |
| mistral.rs | product | CUDA | Metal, not HIP | product |
| burn / burn-lm | framework | CUDA | **ROCm + Vulkan** | framework + small demo |
| tinygrad | compiler | yes | **yes** | compiler |
| airllm | small Py | yes | no (MLX on Mac) | product, slow |
| exo | product | no | no (MLX / CPU) | product |
| TabbyAPI | small Py + Exllama | yes | no | product |
| Aphrodite / Sonar | vLLM-scale | yes | **ROCm** | product |
| LoRAX | TGI-scale | yes | no | product |
| lmdeploy / TurboMind | product C++ | yes | no | product |
| PowerInfer | llama.cpp fork | yes | **HIPBLAS** | research + sparse models |
| Medusa / EAGLE / MTP | papers | any engine | any engine | technique |

---

## Three templates for Vesper

Vesper is MIT, C++17, HIP-first, starting from a CPU-correct Qwen3
dense loop. It is not a Python server and not a model zoo. The
three engines that match that shape:

### 1. FlashQwen, for the C++ Qwen3 loop

Copy the *layering*, not the CUDA.

- Token engine vs text process. Vesper's `Engine` already owns
  prefill/decode and must never own HTTP or ChatML.
- Stage 1 is the next work: safetensors load, one-sequence decode,
  then fused GEMV+norm+RoPE, then a HIP graph of that step. Their
  measured ladder (unfused → fused QKV → FlashDecoding → graph)
  is the order that actually moved tok/s.
- Stay on one model (Qwen3 dense) until the loop is honest. Their
  95% of vLLM number is what happens when you refuse a zoo.

Skip their Go/gRPC serving layer. Skip CUTLASS until a HIP GEMV
beats llama.cpp on one card.

### 2. Fastgen, for memory and graphs

After single-sequence HIP decode works, Fastgen is the smallest
map of the next objects:

- Paged KV with a block table.
- Chunked prefill / token budget so a long prompt does not stall
  decode.
- **HIP graph** of the decode step (their CUDA graph, ported).
- Host-side KV pages. Radeon boxes often have fat host RAM and
  awkward VRAM. Fastgen already treats host as a KV tier.

nano-vllm teaches the same scheduler in fewer lines. Fastgen is
the better *template* because it is meant to be imported, which is
how DESIGN.md wants the engine used, and because host-side KV is
already in the file list (`cache.py`).

### 3. ds4, for HIP, expert cache, and MTP

When the dense loop is correct and we want FreeToken-class
numbers on a Radeon:

- They already ship `make strix-halo`. The graph backend is the
  existence proof that a from-scratch C engine can be HIP without
  going through vLLM.
- SSD / host expert cache + overlapped prefill of the next routed
  layer is the FreeToken machine model in C. Pair it with
  DESIGN.md's \(q^\star\) split (PCIe fill vs CPU expert GEMV).
- DSpark / GLM MTP is the draft/verify loop Qwen3.8 will need.
  Steal the policy (confidence threshold, keep verifier state,
  `--dspark-strict` for byte-identical decode), not the Metal
  kernels.
- Continuation fixtures against official tokens. Same religion as
  hipEngine and as `tests/test_engine.cpp`.

---

## What not to clone

- **llama2.c** is already the v0 style. Keep that file small. It
  has no GPU lesson.
- **PowerInfer** is the right *CPU+GPU split* only if we train a
  ReLU-sparse Qwen. We will not. Expert cache (ds4 / FreeToken)
  is the version that works on stock MoE.
- **AirLLM** is layer streaming. Too coarse, too slow.
- **tinygrad / burn / candle** are frameworks. Using them would
  recreate the old Vesper training-library mistake.
- **mini-sglang / Aphrodite / lmdeploy / LoRAX / TabbyAPI / exo**
  are products or Python servers. Read them for one trick
  (overlap schedule, SGMV, KV LRU, Exllama fused decode, MLX TP).
  Do not start from their trees.

---

## Practical steal-list for the current roadmap

Matches `docs/DESIGN.md` steps 1–7:

1. CPU oracle: already llama2.c-shaped. Keep it.
2. Weight load: FlashQwen's safetensors path, then GGUF like ds4 /
   llama.cpp.
3. HIP GEMV + RMSNorm + RoPE: FlashQwen Stage 1 kernels, written
   for wave32 / gfx1100, compared to the CPU oracle.
4. Fuse + HIP graph: FlashQwen S19 / Fastgen CUDA graphs.
5. Paged KV + token budget: Fastgen `cache.py` + FlashQwen
   scheduler. Still one architecture.
6. MoE offload: ds4 expert cache + FreeToken \(q^\star\).
7. MTP draft/verify: ds4 DSpark policy on Qwen3.8 / DeepSeek MTP
   heads. Exactness gate like `--dspark-strict`.
