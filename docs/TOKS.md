# How you squeeze tok/s

Date: 2026-08-24.

This is the practical order of work for a from-scratch local inference
engine on one consumer GPU. The first target is AMD Radeon RDNA3 /
RDNA3.5 (gfx1100 / gfx1151) over HIP/ROCm, with Vulkan/RADV as a peer
backend, not a fallback. NVIDIA blog posts, NVFP4, and multi-user
vLLM tricks are cited only when they transfer. Other engines and the
tricks they actually ship are in [SMALL_ENGINES.md](SMALL_ENGINES.md).

Vesper already has the cheap part. `Engine` owns one model, one linear
KV arena, and a fixed scratch set. `forward_token` in
`src/engine.cpp` is the whole decode step. There is no autograd graph
and no per-op Tensor. That is the correct shape. Everything below is
about what to add next, and what to refuse.

Always report **prefill tok/s**, **decode tok/s**, accepted tok/s if
you speculate, plus model, quant, context, and GPU. Marketing posts
mix the two. Prefill is compute-bound and can look like thousands of
tok/s. Decode is the number a local user feels.

## Leverage, not hype

| Rank | Technique | Typical decode win on one user | When to do it |
| --- | --- | --- | --- |
| 1 | No alloc / no Tensor graph | 2–10× vs a training-library loop | Already done. Guard it. |
| 2 | Decode roofline accounting | Does not raise tok/s. Stops you wasting weeks. | Before every kernel project |
| 3 | Weight quant (Q8, then Q4) | ~2× Q8 vs F16, ~2× Q4 vs Q8 if dequant stays cheap | Before Qwen3-8B will fit |
| 4 | GEMV layout for decode | 2–5× vs a batched GEMM used at M=1 | First HIP kernel |
| 5 | Fuse RMSNorm+QKV+RoPE, SwiGLU | 10–40% once you have many small launches | After unfused HIP matches CPU |
| 6 | Workgroup / wavefront | 10–25% on RDNA3. This is why Vulkan often wins. | While writing the GEMV |
| 7 | HIP graphs | 5–20% if you still launch hundreds of kernels/token | After fusion, static decode |
| 8 | KV layout, then KV quant | Small at 512. Large at 32k+. | After HIP decode works |
| 9 | Speculative / MTP | 1.3–3× accepted tok/s on dense 20B+ | After Q4 decode is honest |
| 10 | MoE expert LRU + q* | The FreeToken numbers. Zero on dense. | After dense HIP GEMV |
| 11 | Prefix / radix / GDN anchors | Huge on multi-turn agents. Zero on a fresh prompt. | After hybrid layers exist |
| 12 | Continuous batching / paged attn | Near zero for one sequence | Defer. Do not start here. |

The viral 70–80 tok/s posts on a 35B-class name are MoE with ~3B
active, plus host-RAM expert offload. A dense Qwen3.8-27B Q4 on a
7900 XTX lives around 40–60 decode tok/s before speculation. If you
do not believe that, do the roofline first.

## The decode roofline

```
tok/s ≈ achieved_bandwidth / bytes_touched_per_token
```

For dense decode, each new token streams the **whole** weight set
plus the KV written so far. Prefill streams each weight once for
many tokens, so arithmetic intensity rises with prompt length and
you become compute-bound.

Worked numbers for Vesper's Qwen3-8B shape (`config.cpp`: 36 layers,
hidden 4096, 32Q/8KV, head_dim 128, intermediate 12288, untied
lm_head, ~8.2B params):

| Quant | Weight bytes/token | KV F16 at 512 tok | 7900 XTX peak (960 GB/s) | Honest 70% peak |
| --- | --- | --- | --- | --- |
| F32 | ~33 GB | +75 MB | does not fit 24 GB | n/a |
| F16 | ~16.4 GB | +75 MB | ~58 tok/s | ~41 |
| Q8 | ~8.5 GB | +75 MB | ~112 | ~78 |
| Q4 | ~4.5–5.0 GB | +75 MB | ~190–210 | ~130–150 |

KV for this shape is `36 * 2 * 8 * 128 * sizeof * seq`. F16 is
144 KiB/token. At 4k that is 576 MiB. At 32k it is 4.5 GiB. At
128k it is 18 GiB, which evicts Q4-8B off a 24 GB card unless you
quantize KV.

A 27B dense Q4 is ~14–16 GiB of weights. Same card, same formula:
50–70 tok/s before speculation. Community Qwen3.8-27B kits on a
128 GB Strix Halo report 30–36 tok/s. That is the real dense-27B
number.

A Qwen3.6-35B-A3B Q4 streams ~1.5 GB of **active** weights. Peak
math says hundreds of tok/s. hipEngine's published W7900 row is
~116 decode tok/s (512 in / 128 out, ParoQuant W4). The gap is
dequant ALU, dispatch, and routing, not a missing CUDA Graph
flag. See [hipEngine ROOFLINE](https://github.com/shisa-ai/hipEngine/blob/main/docs/ROOFLINE.md):
a well-written sequential stream on Navi 31 hits 75–85% of peak.
Scattered dequant drops you to 40–60%, or worse. Their PARO W4
path back-calcs ~150 GB/s effective. That is 16% of a W7900's
864 GB/s peak. W4 is not automatically 2× W8.

Instrument this before you fuse anything:

```
bytes/token = resident_weight_bytes
            + kv_bytes_per_token * seq
            + scratch_readwrite_bytes
achieved_GB/s = bytes/token * decode_tok/s / 1e9
```

If achieved GB/s is under ~40% of `rocm-smi` memory clock peak,
the kernel is not bandwidth-bound yet. Fix layout, occupancy, or
dequant first. If you are already at 70%+, more fusion will not
double tok/s. You need fewer bytes (quant, KV quant, MoE) or more
tokens per stream (MTP / speculative).

---

## 1. Don't allocate on the hot path

**What it is.** Decode is a loop over resident buffers. Allocate
weights, KV, and scratch at `Engine` construction. Reuse them.
Do not build an autograd Tensor graph, do not `hipMalloc` per
layer, do not return new `std::vector<float>` from kernels.

**When it helps.** Every token, prefill and decode, dense and MoE.
Worst if you came from PyTorch: allocator + metadata + backward
hooks on a 36-layer Qwen3 block will dominate the math. Single
user or multi user does not matter. Short context makes the tax
worse because the kernels themselves are shorter.

**Complexity.** Hours if you are writing C++ from scratch and you
already think in arenas. Weeks if you start from a training
library and try to "just turn grads off." Vesper already does
this. `Scratch` in `engine.h` is the contract. Do not grow a
generic Tensor type to make HIP feel like PyTorch.

**AMD caveats.** ROCm's hipMalloc / hipFree path is slower and
more fragmentation-prone than people coming from CUDA expect.
`hipMallocAsync` plus a pool is the minimum if anything must
allocate after startup. The default stream syncs more than you
want. Open a dedicated non-blocking stream and stay on it.

**Dependencies.** None. This is the prerequisite for HIP graphs,
fusion, and honest tok/s. If you allocate, graphs cannot capture,
and every other technique reports noise.

---

## 2. Weight quantization (Q8, then Q4; FP8 later)

**What it is.** Store weights in fewer bits and dequantize in the
GEMV. Q8_0 is a block-scaled int8 (GGUF: 32 values + one F16
scale). Q4_K is a super-block scheme with mixed scales. FP8 is a
native float8 with a scale, useful when the GPU has FP8 matrix
hardware.

**When it helps.** Decode on dense models, immediately and a lot,
because you cut the bytes that dominate the roofline. Prefill
gains too, but prefill is often compute-bound, so Q4 can be
slower than F16 if dequant ALU exceeds the saved bandwidth. MoE
gains per expert the same way, and also fits more experts in the
VRAM LRU. Short context: almost all win is weights. Long context:
KV starts to compete, so weight quant alone plateaus. Single user
gets the full win. Multi user still wants it, then wants KV quant.

**Complexity.** Q8 GEMV: a few days (load path + one dequant inner
loop + CPU oracle). Q4_K: one to two weeks. The format is
annoying, and the inner loop is where RDNA3 tok/s lives or dies.
A naive "dequant to F16 tile in LDS, then FMA" often loses to Q8
because you spent the bandwidth savings on extra instructions.
FP8 pack/load: days if you only store it. Weeks if you want a
real FP8 pipeline. Skip FP8 on gfx1100 as a compute path.

**AMD caveats.**

- There is **no NVFP4** on Radeon. NVFP4 is Blackwell tensor-core
  FP4 with a 16-element E4M3 scale
  ([NVIDIA](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)).
  Chasing FreeToken's NVFP4 checkpoints is wasted work.
- RDNA3 / RDNA3.5 have **no native FP8** WMMA or `v_cvt_f32_fp8`.
  FP8 on gfx1100/gfx1151 is emulated: upcast to F16, then F16
  WMMA. Memory can shrink. Compute does not get cheaper. hipEngine
  documents the invalid opcodes in
  [VLLM_RDNA3.md](https://github.com/shisa-ai/hipEngine/blob/main/docs/VLLM_RDNA3.md).
  FP8 arrives as a real datapath on CDNA3 and RDNA4.
- RDNA3 **does** have integer dots: `v_dot4_i32_iu8` and
  `v_dot8_i32_iu4`, plus WMMA IU8/IU4 for tiles with M≥16
  ([GPUOpen WMMA](https://gpuopen.com/learn/wmma_on_rdna3/)).
  Decode (M=1) cannot use WMMA. Prefill can. Do not call rocBLAS
  WMMA for a 1×K×N GEMV.
- GGUF Q4_K / Q8_0 is the interchange to beat. llama.cpp already
  has the formats and the AMD baseline. Inventing a new 4-bit
  layout is a research project, not next week.

**Dependencies.** A working F32 or F16 GEMV and a CPU dequant
oracle. Load GGUF or safetensors first so you are not generating
random Q4 for tests. HIP graphs are independent. Fusion wants the
quant GEMV as the inner loop, not the other way around.

---

## 3. Fused kernels (RMSNorm+QKV+RoPE, SwiGLU)

**What it is.** One launch that does RMSNorm, the Q/K/V projections,
optional QK-norm, and RoPE, writing K/V straight into the cache.
A second fused kernel does silu(gate)*up so you never materialize
both full `intermediate_size` vectors plus a third SwiGLU output
if you can overwrite one of them.

**When it helps.** Decode, always, once you have more than ~20
launches per layer. Prefill less so: the GEMMs dominate, and a
fused "norm + QKV" at sequence length 4k is a different kernel.
Dense and MoE both want the attention-side fuse. SwiGLU fuse
matters more on dense (the MLP is most of the weights) and on
each expert GEMV for MoE. Short context and small models (0.6B)
see the biggest relative win because launch overhead is a larger
fraction of a 1 ms token. Multi user can hide launches behind
other sequences. Single user cannot.

**Complexity.** Unfused twins first: a couple of days. A correct
fused decode kernel: several days to a week, including the
Qwen3 QK-norm branch. Do not fuse residual-add into the next
norm until the unfused path matches the CPU oracle to ~1e-4
relative on logits.

**AMD caveats.** hipEngine's rocprof note for llama.cpp-like
decode is on the order of **~1600 HIP dispatches per token**. At
100 tok/s that is 10 µs per dispatch budget. A 1–3 µs launch
eats 17–50% of the token. Fusion is how you cut that count in
half. Triton flash-attn is the wrong fuse to start with:
`VLLM_USE_TRITON_FLASH_ATTN=0` is still the gfx1100 survival
switch ([vLLM #4514](https://github.com/vllm-project/vllm/issues/4514),
stack frame 277288 > 262136). Write the attention mix yourself
for seq-in-cache, one query. rocWMMA flash-attn for **prefill**
has been a regression on gfx1100/gfx1151 when people leave
`ROCWMMA_FATTN=ON` ([llama.cpp discussion](https://github.com/nabe2030/hip-vs-vulkan-evo-x2)).
Default it off and measure.

**Dependencies.** Resident scratch (technique 1) and a GEMV you
trust (technique 4). Quant can land inside the same fused kernel
later. Graphs (technique 6) become worthwhile after the launch
count drops and the remaining graph is static.

---

## 4. GEMV vs GEMM layout for decode

**What it is.** Prefill is GEMM: many tokens × weight. Decode is
GEMV: one token × weight. Those want different weight layouts,
different workgroup shapes, and different instructions. A 16×16
WMMA tile used at M=1 throws away 15/16 of the tile.

**When it helps.** Decode, always. This is the first HIP kernel
that should exist. Prefill should keep a real GEMM (rocBLAS /
WMMA / your own tiled kernel) once sequence length is tens to
thousands. Dense and MoE both split this way: MoE decode is a
handful of GEMVs, MoE prefill is a fat GEMM over many experts.
Single user decode is pure GEMV. Multi-user decode with batch 8
is a skinny GEMM and a different kernel.

**Complexity.** A correct F32 GEMV: a day. A Q8 GEMV that actually
beats F16: several days. A Q4 GEMV that beats Q8: a week of
layout and dequant work. Prefill GEMM can stay "call rocBLAS
F16" for a long time on large N. Do not spend a month on a
prefill kernel before decode GEMV is honest.

**AMD caveats.**

- HIP/ROCm on RDNA3 defaults to **wave32**. Vendor BLAS is tuned
  for CDNA (MI300), not Navi 31. llama.cpp's HIP backend shares
  CUDA kernels and loses decode to Vulkan for that reason
  ([llama.cpp #20934](https://github.com/ggml-org/llama.cpp/issues/20934)).
- Store decode weights so a wave issues wide consecutive loads
  (128-byte cache lines). hipEngine's working decode profile is
  wave32, vec8 FMA, `-mcumode`, no `-mwavefrontsize64`.
- Do not LDS-stage the whole weight row "because NVIDIA does."
  On RDNA3, LDS plus `__syncthreads` on a bandwidth-bound GEMV
  often costs more than it saves (hipEngine §9 / §11).
- Occupancy is the bandwidth tool. 96 VGPRs/wave → 16 waves/SIMD
  (max). 192 VGPRs → 8 waves. VRAM latency on Navi 31 is ~1000
  cycles. Low occupancy means you cannot hide it.

**Dependencies.** Technique 1. Quant (technique 2) is the same
kernel with a different inner load. You can ship F16 GEMV first
if you only need Qwen3-0.6B. Qwen3-8B F16 is 16.4 GB, which
fits a 24 GB card and leaves little room for KV.

---

## 5. KV cache layout, paged KV, KV quant

**What it is.** Three separate ideas that get sold as one.

1. **Layout.** Vesper v0 is `[n_layers][max_seq][n_kv_heads * head_dim]`
   with K and V separate (`kv_cache.cpp`). Attention-friendly
   layouts pack `head_dim` in the inner stride and keep K and V
   in different orders (K sequential in `seq` for dots, V
   sequential in `head_dim` for the mix).
2. **Paged KV.** Split the cache into fixed blocks (16–256 tokens)
   and address them through a block table, like vLLM
   PagedAttention ([SOSP 2023](https://arxiv.org/abs/2309.06180)).
3. **KV quant.** Store K/V in Q8 or Q4 with per-block scales so
   long context fits.

**When it helps.**

- Layout: decode attention, more as `seq` grows. Prefill
  attention too. Dense and MoE both pay it on full-attention
  layers. Hybrid Gated DeltaNet layers have a recurrent state
  instead of a growing KV, so this shrinks.
- Paged KV: **multi user**, prefix reuse, and "I do not know
  max_seq at start." For **one sequence with a known max_seq**,
  a linear arena is faster and simpler. The page table is an
  extra gather on every score/mix.
- KV quant: long context on a 24 GB card. At 512 tokens, Qwen3-8B
  F16 KV is 75 MiB. Do not quantize that. At 32k–128k it is the
  difference between fitting and not.

**Complexity.** Better linear layout: a day or two, plus retuning
the attention kernel. KV Q8: several days (pack on write, dequant
on read, KL gate). Paged KV: one to two weeks, and it infects
every attention kernel and any HIP graph. Do not page "because
vLLM does."

**AMD caveats.** Gather from random pages is harsher on chiplet
Navi 31 (GCD + 6 MCDs, ~1000-cycle VRAM) than on a monolithic
GPU. Keep the block size large enough that a wave streams a
full cache line. hipEngine's INT8 KV on Qwen3.6-35B-A3B at 128k
is nearly speed-neutral vs BF16 KV (62.2 → 60.0 tok/s) because
decode is still weight-bound. That is the result you want. If
KV quant tanks tok/s, the dequant is in the wrong place.

**Dependencies.** Working decode attention over a linear cache.
Paged KV wants a block allocator and makes graphs harder
(technique 6). KV quant wants the write path to pack in-place so
you do not store F16 and Q8. Prefix cache (technique 10) is
much easier if pages exist, which is the one good reason to
page later.

---

## 6. CUDA / HIP graphs

**What it is.** Capture a static sequence of launches into a
`hipGraph_t`, instantiate, and `hipGraphLaunch` each decode
step. One doorbell instead of hundreds of `hipLaunchKernel`
calls. Docs:
[HIP graphs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html).

**When it helps.** Decode, after you still have many launches
and the launch topology does not change token-to-token. Short
context and small models gain more. Long context is attention-
bound and the graph win shrinks. Dense static graphs are easy.
MoE with host-side routing is hostile: a host sync to read
expert ids breaks capture. FreeToken's trick is a **device-side**
LRU so the CUDA graph never comes back to the CPU. Multi-user
continuous batching also breaks naive graphs, because the batch
shape changes every step. Single user, one sequence, greedy or
fixed top-k: this is the friendly case.

**Complexity.** A week if the decode step is already allocation-
free and sampling can stay off-graph (read logits once per
token). Two weeks if you need device-side sampling or graph
update of the KV write cursor. Do not capture until unfused
HIP matches the CPU oracle. Debugging a wrong graph is miserable.

**AMD caveats.**

- Capture forbids sync `hipMalloc` / `hipMemcpy` / `hipFree`.
  Use the async variants, or better, never allocate.
- HIP graphs on RDNA3 have been less reliable than CUDA graphs
  at long context. llama.cpp ROCm has crashed in graph execution
  around 32k on gfx1100 ([#20934](https://github.com/ggml-org/llama.cpp/issues/20934)).
  Keep a non-graph path.
- Graph replay is a **dispatch** optimization. hipEngine measured
  +2.6–3.9% on an already-fused 4k decode. If you still have
  1600 launches/token, graphs help more. If you fused down to
  ~20, measure before you build a graph compiler.
- A host `argmax` on logits is a sync. Either sample on device
  or accept one sync per token. One sync is fine. One sync per
  layer is not.

**Dependencies.** Technique 1 is mandatory. Fusion first is
smarter than graphing 40 tiny kernels. Quant does not block
this. Paged KV and host-routed MoE do, unless the page table
and the LRU live in device memory.

---

## 7. Continuous batching / paged attention

**What it is.** Iteration-level scheduling: after each decode
step, finished sequences leave and new prefills join, so the
GPU does not wait for the longest request. Paged attention is
the memory system that makes that insertion cheap
([vLLM blog](https://vllm.ai/blog/2023-06-20-vllm),
Orca/iteration batching).

**When it helps.** **Multi user.** Throughput at concurrency 4–32
is the whole point. Prefill/decode mixing on a server. It does
**not** raise single-user decode tok/s. For one local sequence
it adds a block table, a scheduler, ragged batching, and
usually a Python runtime. That is why llama.cpp beats vLLM on
a desktop with one chat window, and why vLLM wins the two-
client TTFT test on Strix Halo
([soothill.io, 2026-08-03](https://www.soothill.io/blog/2026/08/03/llamacpp-vulkan-vs-rocm-strix-halo/)).

**Complexity.** Weeks. You will rewrite KV ownership, attention,
and the CLI. Skip it until someone has two concurrent users
and a measured queue.

**AMD caveats.** vLLM-ROCm on RDNA3 is a second-class kernel
path. Triton flash-attn is still a footgun on gfx1100. AIter
is MI300. If we ever want this, we implement a tiny C++
scheduler ourselves. We do not import vLLM.

**Dependencies.** Paged KV. A batch>1 GEMM path. Graphs that can
recapture or that are parameterized. None of that is on the
v0 critical path.

**When it is not worth it.** One user, one model, known
`max_seq_len`. That is Vesper. Linear KV, one forward, done.

---

## 8. Speculative decoding / MTP / draft models

**What it is.** A cheap drafter proposes γ tokens. The main model
verifies them in one parallel forward (a GEMM with M=γ+1).
Accepted tokens are committed; the first rejection is resampled.
Leviathan et al., ICML 2023:
[paper](https://proceedings.mlr.press/v202/leviathan23a.html).
MTP is the same idea with extra heads on the main model
(Qwen3.8 trains this). No second model required if the heads
exist.

**When it helps.** **Dense decode**, especially 20B+, where one
weight stream is expensive and a 2–3× accepted-token rate is
the only honest way to look fast on a 7900-class card. Prefill
does not use it. Short, easy completions (code, JSON, repeated
system style) accept more. Long, high-entropy chat accepts
less; if accept length stays ~1.1 you lost VRAM and gained
nothing. MoE with 3B active already streams fewer bytes, so
speculation is extra, not required. Single user is the sweet
spot. Multi user can speculate per request but the scheduler
gets ugly.

**Complexity.** Draft+verify with a small external model: one to
two weeks, plus VRAM for the draft. MTP heads on a model that
already has them: a week if the main decode works. Matching
llama.cpp's MTP speed may mean matching their numerical
shortcuts. hipEngine kept an "exact" mode and a "llama-compat"
mode for that reason.

**AMD caveats.** Verify is a small-M GEMM. M=4–8 is still a bad
shape for WMMA 16×16. Write a skinny GEMM, do not call the
prefill kernel. Two models means two graphs. Device-side
accept/reject is what keeps this from syncing every token.
There is no NVFP4 draft path. A Q4 draft + Q4 target is the
RDNA3 version.

**Dependencies.** Honest Q4/Q8 decode first. If the target is 30
tok/s because the GEMV is bad, speculation multiplies a bad
number. Graphs help the verify step. Paged KV is optional.

---

## 9. MoE expert cache + CPU offload (FreeToken q*)

**What it is.** Host RAM holds every expert (source of truth).
VRAM holds an LRU of complete `(layer, expert)` slots. Prefill
double-buffers the next layer's experts over PCIe. Decode
misses split between PCIe fill and in-place CPU expert GEMV
using two measured bandwidths:

```
q* ≈ m * B_P / B_H
```

`m` is unique misses this step, `B_P` is host→device expert-copy
bandwidth, `B_H` is CPU expert-kernel bandwidth. Fill `q*`
experts, run the rest on the CPU, merge the partial sums.
Paper: [arXiv:2608.16157](https://arxiv.org/abs/2608.16157).
Code: [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken).
Calibrate with the equivalent of `ft bench bw`.

**When it helps.** **MoE only**, and only when the full expert
pool does not fit in VRAM. Prefill: double-buffer or you will
serialize PCIe behind every layer (prefill touches almost every
expert). Decode: LRU hit rate is the game. FreeToken reports
16% miss on Qwen3.6 vs 62% for a static llama.cpp split, same
cache capacity. Dense models: this section does not apply.
Single user is enough. Multi user shares the LRU, which can
help or thrash.

**Complexity.** Weeks. Host packing, pinned memory, LRU, two
execution backends, a merge kernel, and a bandwidth bench.
Do it after a fused-GPU MoE path exists so the correctness
gate is "hybrid == fused."

**AMD caveats.**

- FreeToken is NVIDIA-only (CUDA 13, RTX 30/40/50, FlashInfer,
  NVFP4). hipify will not produce the stack. Copy the **machine
  model**, not the repo.
- `hipHostRegister` / pinned DMA is the ROCm analog of
  `cudaHostRegister`. Measure `B_P`. Do not assume 32 GB/s.
- A 7900 XTX desktop often has fat host RAM and a strong CPU.
  A Strix Halo box is unified memory: `B_P` is not PCIe, it is
  a copy inside the same fabric, and `B_H` vs GPU is a
  different ratio. Recalibrate on the machine. That is the
  point of q*.
- Device-side LRU is what keeps HIP graphs legal. A host
  `cudaMemcpy` of routing indices every layer kills the graph
  and most of the speedup.

**Dependencies.** Quantized experts (Q4/Q8). A CPU expert GEMV
that is actually fast (the CPU side of q* is worthless if
`B_H` is a naive triple loop). Dense HIP GEMV first, so the
expert kernel is a known quantity. Do not start Vesper here.

---

## 10. Prefix / radix cache, semantic anchors (Gated DeltaNet)

**What it is.** Two caches for two kinds of history.

- **Prefix / radix cache.** Keep KV for token prefixes in a
  trie and reuse it across turns. SGLang RadixAttention:
  [arXiv:2312.07104](https://arxiv.org/abs/2312.07104).
  Chat system prompts, RAG prefixes, and agent loops hit this
  every request.
- **Semantic anchors.** Hybrid models (Qwen3.5/3.6/3.8 Gated
  DeltaNet) compress history into a recurrent state. Agentic
  harnesses then delete thinking/tool blocks. If you
  checkpoint the recurrent state at those special tokens, an
  edit re-prefills only the new suffix. FreeToken §3.3.

**When it helps.** Multi-turn and agent workloads. Prefill of
the second request, not the first. Decode is unchanged except
you skip rebuilding KV. Dense full-attention models get prefix
KV reuse. Hybrid GDN models get the anchor trick, and need it,
because you cannot cheaply reconstruct a recurrent state from
a leftover KV page. Single user with a 200-token prompt and no
session: zero win. Long context without reuse: zero win.

**Complexity.** Exact-prefix KV reuse on a linear cache: a day
(hash the prompt, keep the arena if it matches). A real radix
tree over paged KV: one to two weeks. Semantic anchors: after
GDN layers exist, a few days for the checkpoint table, plus
harness integration. Vesper has no GDN yet. Do not build
anchors for a model you cannot run.

**AMD caveats.** None that change the algorithm. The cost is
VRAM: a radix cache competes with weights and with a MoE LRU.
On 24 GB, keep the prefix cache small and evict LRU. Unified
memory (Strix Halo) makes a large prefix cache more
tempting. Still measure. A 20 GB prefix cache that evicts
Q4 weights is a self-own.

**Dependencies.** KV ownership that can persist past
`generate()`. Paged KV if you want partial prefix hits.
GDN kernels before anchors. This is Qwen3.8 work, not Qwen3-8B
work.

---

## 11. Workgroup shape / wavefront (why Vulkan beats HIP)

**What it is.** On RDNA3 a workgroup processor has two CUs, each
with two SIMD32s. HIP/ROCm runs **wave32** (32 lanes). Mesa RADV
Vulkan runs **wave64** and ACO emits packed FP32 / dual-issue
more often. Workgroup size, VGPR count, and how you reduce
inside the wave decide whether a GEMV hits 70% of peak or 30%.

**When it helps.** Decode GEMV and dequant, every token. Prefill
GEMM cares about tile shape and WMMA, not the same knobs.
Small models and short context show it clearly because you are
not yet KV-bound. Dense Q4/Q8 inner loops are the whole fight.
This is not a "feature" you add. It is how you write the
kernels in techniques 3–5.

**Complexity.** Days of measurement per kernel, forever. Budget
a day to get `rocprofv3` / `rocm-smi --showclocks` in the
workflow. Budget another day the first time you chase a 15%
regression that was occupancy.

**AMD caveats.** This is the section that actually explains the
20–30% llama.cpp gap.

1. **HIP is wave32 and AMD does not support HIP wave64.**
   `-mwavefrontsize64` is explicitly unsupported. LLVM removed
   the compile-time macros that told you it was on
   ([ROCm #4121](https://github.com/ROCm/ROCm/issues/4121),
   [llama.cpp #20934](https://github.com/ggml-org/llama.cpp/issues/20934)
   comment by IMbackK). A local fork on W7800 saw +0.2–16%
   decode from wave64, then they dropped the patch because it
   cannot ship. Write wave32 kernels. Do not wait for AMD to
   reverse this.
2. **RADV is wave64 and packed-FP32 friendly.** IMbackK's
   summary: RDNA3/4 can double FP32 via packed math, LLVM
   rarely emits it in wave32, ACO in wave64 does, and the HIP
   backend is a CUDA port nobody tunes for Navi. Vulkan/RADV
   decode wins of ~20–22% on 7900 XTX (Llama-7B Q4_0, tg128
   175 vs 144) and 25–32% on Strix Halo MoE are the normal
   result, not a misconfig.
3. **Hardware wave64 on RDNA3 is two wave32 passes.** Shuffles
   that assume 64-lane XOR are silent no-ops. `__shfl_xor(val,
   32, 64)` on gfx1100 does nothing. Reductions stay 32-wide
   even if a Vulkan subgroup says 64. hipEngine has seen
   "wave64" reductions that still behave as two 32-lane halves.
4. **VOPD (dual-issue VALU) is wave32-only.** HIP kernels can
   win that back if the compiler pairs instructions. It often
   does not. Hand-written `v_pk_*` packed FP32 in wave32 is
   the supported way to chase Vulkan, not `-mwavefrontsize64`.
5. **Workgroup size.** 128 or 256 threads is a starting point,
   not a religion. Decode GEMV often wants one wave (32) or
   two (64) per output row plus a cheap reduction. Fat
   workgroups that made sense on CUDA waste LDS and VGPRs
   here.
6. **Submission, not just ISA.** [ROCm #6409](https://github.com/ROCm/ROCm/issues/6409)
   shows retained-submission / Redline beating both HIP and
   Vulkan on independent small kernels. Some of the HIP loss
   is HSA queue overhead, which is another reason to fuse and
   graph.

**Dependencies.** You need a kernel to tune. Do this while
writing the GEMV, not as a later "optimization pass" after
the layout is wrong.

**Practical rule for Vesper.** HIP-first, wave32, vec8 loads,
CPU oracle on every change. If HIP decode still loses to
llama.cpp Vulkan by >15% on the same Q4-8B after fusion and
graphs, add a Vulkan/RADV backend. Do not add it first.

---

## 12. Memory bandwidth accounting

Covered as the opening tool. Putting it last in the user's
list is how people skip it.

Add a `vesper-bench` path that prints, every run:

- prefill tok/s and decode tok/s
- model, quant, seq in / tokens out, GPU, backend
- bytes/token (weights + KV + scratch)
- achieved GB/s and percent of a configured peak
- launch count per token, if `rocprof` is on

Refuse to quote a tok/s number without a correctness gate.
The existing gate in `tests/test_main.cpp` (cached decode
logits match full-sequence attention) stays mandatory. HIP
kernels get a second gate: max abs / KL vs the CPU oracle
on the same weights.

---

## What not to start with

- NVFP4, MXFP4 tensor cores, or "FP8 everywhere." gfx1100
  does not have that machine.
- Triton flash-attn. It has broken on gfx1100 in the way that
  makes vLLM unusable until you export
  `VLLM_USE_TRITON_FLASH_ATTN=0`.
- Continuous batching, paged attention, or an OpenAI HTTP
  server.
- hipify of the archived Vesper training GEMM. Decode is a
  different program (`docs/DESIGN.md`).
- A Vulkan backend before a HIP GEMV exists. We need one
  honest number on one backend first.
- Matching FreeToken on an RTX 5090.

---

## Implementation order for Vesper

Vesper today: C++17, CPU-only, Qwen3-style dense GQA +
QK-norm + RoPE + SwiGLU, linear F32 KV, scratch buffers,
tiny random demo. Roadmap in `docs/DESIGN.md` still holds.
This is that roadmap with timeboxes.

### Already done. Keep it.

- No Tensor graph, no alloc in `forward_token`.
- CPU kernels as the oracle.
- Cached-decode == full-attention test.
- CLI that prints prefill and decode tok/s separately.
- Config factories for Qwen3-0.6B and Qwen3-8B.

If a HIP patch introduces a smart pointer per layer or a
`std::vector` resize in the decode loop, reject it.

### Next. Dense, HIP, honest.

Do these in order. Each step has a gate.

1. **Weight load (days).** Safetensors F16/BF16 → F32, then
   GGUF Q8_0 into the same `ModelWeights` slots. Gate: loaded
   Qwen3-0.6B CPU decode is not garbage (smoke a known
   prompt, or compare a short logit vector against a
   transformers dump). Still CPU. Still F32 compute.

2. **HIP device allocator + memcpy (hours–days).** `Buffer`
   grows a device pointer. `copy_from` / `copy_to` are the
   only host syncs at load time. Gate: round-trip a vector.

3. **HIP F32 GEMV, RMSNorm, RoPE, SwiGLU, attention (days).**
   Sibling `kernels_hip`, same signatures as `kernels.h`.
   Prefill may still be a token loop. Gate: layer logits
   match CPU within 1e-4 relative on `tiny_demo` and on a
   0.6B slice.

4. **Decode GEMV layout, not a GEMM (days, overlaps 3).**
   One output row (or a small row tile) per workgroup.
   Wave32. vec4/vec8 loads. No WMMA. Gate: `rocprof` shows
   the GEMV is the time, and achieved GB/s is not
   embarrassing. Compare tok/s to llama.cpp HIP on the same
   0.6B F16.

5. **Q8 GEMV (days).** Dequant in-register, `v_dot4` if it
   wins, otherwise F32 FMA on dequantized values. Gate: KL
   vs F32 CPU, and decode tok/s vs F16 HIP. If Q8 is slower
   than F16, the inner loop is wrong. Do not proceed to Q4.

6. **Fuse RMSNorm+QKV+RoPE and SwiGLU (days).** Unfused twins
   stay in the tree. Gate: same logits, fewer launches,
   higher decode tok/s on 0.6B. Then try Qwen3-8B Q8 on a
   24 GB card.

7. **Roofline printout (hours).** Bytes/token, achieved GB/s,
   llama.cpp HIP and Vulkan numbers on the same box. This is
   the first time we are allowed to say "tok/s" in public.

### As soon as 8B Q8 is honest

8. **Q4_K GEMV (1–2 weeks).** This is the product quant.
   8B Q4 leaves room for KV. Gate: quality + tok/s vs
   llama.cpp Q4_K_M on 7900 XTX / W7900 / Strix Halo.

9. **HIP graph the decode step (days).** After fusion. Keep
   the non-graph path. Gate: equal logits, equal or better
   tok/s, no crash at 4k and 16k.

10. **KV write/read layout + optional KV Q8 (days).** Only
    if 8B Q4 at the context you care about is KV-heavy.
    Do not page yet.

11. **Prefill GEMM (days).** Separate kernel. rocBLAS F16 or
    a WMMA tile for long prompts. Measure `ROCWMMA_FATTN`
    off first.

### Defer until the dense loop is boring

| Item | Why wait |
| --- | --- |
| Vulkan/RADV backend | Need a HIP number to lose to. DESIGN step 8. |
| Continuous batching / paged attention | One user. Linear KV is faster. |
| Paged KV | Add when prefix cache or two sequences exist. |
| Speculative decoding / MTP | Multiplies a good GEMV. Useless before that. Required later for dense 27B. |
| MoE host pool + LRU + q* | FreeToken path. Needs a fast expert GEMV and a MoE model. After 8B. |
| Prefix / radix cache | After sessions exist. |
| Semantic anchors / Gated DeltaNet | Qwen3.8. After dense GQA is done. |
| FP8 weights | No native FP8 on gfx1100/1151. |
| NVFP4 | NVIDIA-only. |
| Triton / vLLM / FlashInfer | Wrong stack. |
| HTTP server | After decode is honest. |

### What "next week" means in this tree

A focused week, one person, no MoE:

- Safetensors or GGUF load into `ModelWeights`
- `kernels_hip` F32 GEMV + RMSNorm + RoPE + SwiGLU
- CPU==HIP gate on `tiny_demo` and Qwen3-0.6B
- Decode tok/s printed next to llama.cpp on the same card
- A short note in this file with the measured GB/s

If that week slips, cut fusion and Q8. A correct F32 HIP GEMV
plus a loader is worth more than a clever graph around CPU
weights.

A second week is Q8 + fused norm/QKV/RoPE + the 8B run.

Vulkan, MTP, and q* are how we chase hipEngine and FreeToken
later. They are not how we debug the first kernel.

---

## Sources

- FreeToken paper: https://arxiv.org/abs/2608.16157
- FreeToken repo: https://github.com/FlashML-org/FreeToken
- hipEngine: https://github.com/shisa-ai/hipEngine
- hipEngine roofline: https://github.com/shisa-ai/hipEngine/blob/main/docs/ROOFLINE.md
- llama.cpp HIP vs Vulkan on 7900 XTX: https://github.com/ggml-org/llama.cpp/issues/20934
- ROCm wave64 unsupported in HIP: https://github.com/ROCm/ROCm/issues/4121
- HIP vs RADV gaps: https://github.com/ROCm/ROCm/issues/6409
- Strix Halo Vulkan vs ROCm: https://www.soothill.io/blog/2026/08/03/llamacpp-vulkan-vs-rocm-strix-halo/
- Triton flash-attn on gfx1100: https://github.com/vllm-project/vllm/issues/4514
- HIP graphs: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html
- RDNA3 WMMA: https://gpuopen.com/learn/wmma_on_rdna3/
- NVFP4 (NVIDIA, not us): https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- PagedAttention: https://arxiv.org/abs/2309.06180
- Speculative decoding: https://proceedings.mlr.press/v202/leviathan23a.html
- SGLang / RadixAttention: https://arxiv.org/abs/2312.07104
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Qwen3.8 is dense hybrid, not a 3.8B: see `docs/RESEARCH.md`
