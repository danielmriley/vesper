# Vesper — Implementation Report

*How the library is actually built.* This document describes Vesper's architecture and implementation as it stands on branch `refactor/audit-fixes`, grounded in the current source and in measured behaviour on the target GPU (AMD Radeon R9700, gfx1201/RDNA4). It supersedes the older, scattered review/plan documents (now under `docs/dev/` and `docs/reports/`).

---

## 1. What Vesper is

Vesper is a **pure C++17, zero-BLAS deep-learning library** for training language models on a single GPU. It has no Python and no external math dependencies — every kernel, from GEMM to attention, is hand-written. The design mirrors PyTorch's *eager, define-by-run* model: you build tensors and modules, ops execute immediately, and a reverse-mode autograd graph is recorded on the fly.

- **Primary backend:** HIP/ROCm, tuned for **gfx1201 (RDNA4)**. A CUDA mirror exists but is off by default; a CPU reference path exists mainly as a correctness harness.
- **Status:** trains a transformer end-to-end on the R9700 today — a 19.8 M-param model at **~54K tok/s (FP32)**, loss descending, **106/107 tests passing** on GPU (the one failure is the intentionally-guarded fp16 path).
- **Scope:** a complete LLM stack — tensor/autograd core, transformer modules (RoPE, GQA, SwiGLU, flash-attention, KV-cache), optimizers, a training loop, text generation, safetensors/checkpoint IO, and a small web server.

---

## 2. Architecture at a glance

```
  examples/ train_tinystories        web/ server        <- applications
        |                                |
  models/  TransformerLM, blocks, config                <- model assembly
        |
  nn/      Linear, RMSNorm/LayerNorm, RoPE, SwiGLU,      <- neural-net modules
           GQA attention, flash-attn, Module system, AMP
        |
  ops/     ~20 dispatch .cpp  (device routing + autograd node)   <- op layer
        |        |            \
  ops/hip 19 .hip   ops/cuda 18 .cu   ops/cpu 6 .cpp    <- kernels per backend
        |
  core/    Tensor, Storage, caching Allocator, Stream,  <- foundation
           autograd Engine (Node/Edge/guards)
```

The layering is strict: modules call ops, ops dispatch to a backend kernel and (if grad is on) record an autograd node. Nothing above the op layer contains device code.

---

## 3. Core: tensors, storage, autograd, allocator

**Tensor / Storage (`core/tensor.*`, `core/storage.*`).** A `Tensor` is a lightweight handle: a `shared_ptr<Storage>` (the device buffer), a `DType`, and `shape/strides/offset`. Copying a Tensor is a shallow copy — another view of the same storage — so views, transposes, and reshapes are metadata-only (`view_ops.cpp`), and `contiguous()` materialises via a strided-copy kernel only when needed. `data_ptr<T>()` carries a dtype-width guard that throws on a genuinely wrong-width reinterpret (e.g. reading an fp16 tensor as `float`). Gradients are shared across handle copies through a `shared_ptr<shared_ptr<Tensor>>` indirection, so `x.grad()` is consistent no matter which copy you hold.

**Autograd (`autograd/engine.*`, `autograd/node.h`).** Reverse-mode, eager, cycle-free *by design*. Each differentiable op builds a `Node` whose `backward_fn` **captures its inputs by value** and its output by `weak()`; `Node::next_edges` hold `weak_ptr`s. The graph is therefore kept alive entirely by the closure chain from the loss down to the leaves, and the weak output-capture deliberately breaks the node→output→node self-cycle. `backward()` runs Kahn's algorithm: a node executes only after all its consumers have decremented it to zero, i.e. exactly when its output-gradient is fully accumulated. `NoGradGuard`/`grad_mode_enabled` gate node creation. This is a considered, correct design — but it means peak backward memory holds the *whole* activation graph until the loss tensor dies (activations aren't freed incrementally as each node completes).

**Allocator (`core/allocator.cpp`).** A per-device caching allocator implemented as an intentional **leaky singleton** (it outlives every `Storage`, avoiding a static-destruction crash). It bins allocations to **powers of two** and reuses freed blocks per bin. It's simple and stable for fixed-shape training, but has no block splitting/coalescing and rounds every request up to the next power of two — so it can over-reserve up to ~2× on large tensors, which is the main VRAM-efficiency limit on the 32 GB card.

**Precision types (`core/half.h`).** FP16 and BF16 scalar types with conversion. **BF16 conversion is correct** (round-to-nearest-even, NaN preserved) and carries FP32's exponent range. FP16's *normal-path* rounding is correct, but its overflow/NaN classification is inverted (large finite values → NaN, real NaN → Inf) — a reason to prefer **BF16** for mixed precision.

---

## 4. The op layer & backend dispatch

Every operator is a host function in `src/ops/*.cpp` (~20 files) that: (1) computes the output shape/dtype, (2) allocates the output, (3) **routes by device** —

```cpp
if (device == HIP)  { #if USE_HIP_BACKEND  op_hip_dispatch(...);  #else throw; #endif }
else if (CPU) { ... } else if (CUDA) { ... }
```

— and (4) if `requires_grad && grad_mode_enabled`, records the autograd node. Broadcasting is handled on-device by passing broadcasted strides into the same kernel (no host-side expansion). This is a clean, PyTorch-like dispatch model. Its structural consequence: **execution is op-by-op, one kernel launch per op**, and there is no cross-op fusion except where a *fused op* has been written by hand — which matters a lot for performance (§9).

Backends: **19 `.hip` kernels** (the tuning surface), 18 `.cu` mirrors (off by default, not CI-validated), 6 `.cpp` CPU references. GPU coverage of the training path is essentially complete — matmul, all elementwise (incl. broadcast), RMSNorm/LayerNorm fwd+bwd, softmax/log_softmax, SiLU/GELU/SwiGLU fwd+bwd, embedding fwd+bwd, RoPE, flash-attention fwd+bwd, GQA `repeat_kv`, cross-entropy, gather/scatter, cast, reductions, fused Adam, dropout — all run on-GPU. The only non-GPU spots on the training path are a small `targets→CPU` sync in cross-entropy and (only under the unused mixed-precision path) `GradScaler`'s per-grad CPU scan. Whole-tensor `max`/`min` reductions have no HIP kernel and throw, but aren't on the transformer path.

---

## 5. GPU kernels (RDNA4 / gfx1201)

The `.hip` files are compiled **as C++ via hipcc** with `--offload-arch=gfx1201` (so `CMAKE_HIP_ARCHITECTURES` is effectively inert; correctness relies on the CXX compiler being hipcc). Key kernel-level facts:

- **Wave32-correct reductions.** RDNA4 wavefronts are 32-wide (not 64). The block reductions in `normalization.hip` use `WARP_SIZE=32`; an earlier wave64 assumption that halved norm/softmax denominators was fixed and validated to ~1e-8 vs the CPU reference.
- **rocWMMA FP16 GEMM (`gemm.hip`).** A register-blocked matrix-core kernel (fp16 in, **fp32 accumulate**) using the rocWMMA library — the raw `__builtin_amdgcn_wmma_*` intrinsics don't select on gfx1201, so rocWMMA is required. It handles ragged edges via LDS zero-padding and routes fp16 output through an fp32 LDS scratch (rocWMMA can't store an fp32 accumulator straight to fp16). Measured **~25–59 TFLOPS** integrated (a faster 97-TFLOPS warp-tiled variant is benchmarked but not yet wired in).
- **Batch-fold for linears (`gemm.cpp`).** `[B,T,C] @ W` is reshaped to `[B·T,C] @ W` so QKV/MLP/LM-head hit the fast 2D kernel instead of a single-buffered batched one — **~5–7×** on transformer linears.
- **Fused ops.** `silu_mul` (SwiGLU), a fused RMSNorm+Linear, and a fully-fused Adam step (`adam_update_`/`lerp_`/`addcmul_`, float4) exist and are used on the hot path. Elementwise kernels are float4-vectorised on contiguous data.
- **Flash-attention (`flash_attention.hip`).** A genuine online-softmax forward and a correct backward (verified term-by-term against the CPU reference) — but both are **scalar, one-thread-per-query/KV**, with heavy register use; they don't use the matrix cores. *Landmine:* the forward silently no-ops for `head_dim != 64` (hardcoded `HEAD_DIM=64`), so any model with a different head dim gets OOB garbage — must be fixed before scaling.

**Precision routing in GEMM:** dispatch is by dtype — FP32 → scalar register-tiled kernel; FP16 → the WMMA kernel *only* for `!transA && !transB`, else a scalar fp16 kernel. There is **no BF16 compute kernel**. Two consequences: FP32 training never touches the matrix cores at all, and even in fp16 the *backward* GEMMs (always transposed) fall back to scalar — so only forward fp16 matmuls are accelerated.

---

## 6. The transformer stack

- **Config (`models/config.*`).** `TransformerConfig` — vocab, dim, layers, heads, KV-heads (GQA), FFN hidden, seq len, RMSNorm-vs-LayerNorm, RoPE base, `tie_word_embeddings` (defaults **false** — worth enabling for real models, where embed+lm_head is otherwise double-counted), and `gradient_checkpointing`.
- **Block (`models/transformer_block.cpp`).** Pre-norm residuals: `h = x + attn(norm(x))`, `out = h + ffn(norm(h))`. Attention is GQA with a **fused QKV projection** (one GEMM), RoPE applied to Q/K, `repeat_kv` for grouped heads, causal masking, and a KV-cache for decode. FFN is SwiGLU (fused `silu_mul`). Attention uses flash-attention for `seq ≥ 512` and a materialised `[B,H,S,S]` softmax path below that (so the training default seq=256 currently takes the materialised path).
- **Model (`models/transformer.cpp`).** `TransformerLM` = token embedding → N blocks → final norm → LM head, with optional weight tying and opt-in per-block gradient checkpointing (`autograd::checkpoint`). *Caveat:* checkpoint recompute doesn't save/restore RNG state, so dropout inside a checkpointed block would draw a different mask on recompute (wrong grads) — fine as long as checkpointed blocks are dropout-free.
- **RoPE — current slow path.** The model applies rotary embeddings via the *functional* `apply_rotary_emb` — ~15 elementwise ops plus `stack`/`permute`/`contiguous`, **and it rebuilds cos/sin on the CPU and copies them H2D on every forward, every layer**. A fused `apply_rope_hip` kernel and a precompute-once `RoPEFrequencies` class already exist but are unused; wiring them in is a large, cheap launch-count reduction (§9).

---

## 7. Training, optimization, data

**Loop (`examples/train_tinystories.cpp`).** Standard: gradient accumulation (micro-batch 8 × 8 = effective 64) → `compute_loss` (cross-entropy) → scaled `backward()` → `clip_grad_norm_(1.0)` (a single, well-batched device→host sync) → `Adam.step()` → inline warmup+cosine LR. Everything runs in **FP32**; the AMP scaffold is present but unused. The loop reads `loss.item()` once per micro-batch (8 host syncs/step, for the NaN check + logging).

**Cross-entropy (`nn/functional.cpp`).** Runs on GPU with an analytic `(softmax − one_hot)/N` backward (not autograd-through-softmax) and supports `ignore_index`. It copies the 1-D `targets` to CPU to build the ignore/OOB-safe mask (a small per-step sync), and materialises a couple of `[B·S, V]` tensors — the main memory cost at large vocab, alongside the LM-head GEMM.

**Optimizers (`optim/`).** Adam/AdamW (fully fused GPU step; step counter stored as Int32 for checkpoint fidelity), SGD, Lion, and warmup/cosine schedulers. Adam holds 2× the parameters in fp32 state. `GradScaler` exists but scans every gradient on the CPU for inf/nan — a throughput killer that only fires under the (currently unused) mixed-precision path.

**Data (`data/`).** A `DataLoader` (synchronous, main-thread) and a `PrefetchDataLoader` (worker-thread, queue-based). The example uses the **synchronous** one: tokens are read from a pre-tokenised `uint16` `.bin`, widened to `int32` per sample on the CPU, collated to `[B,S]`, and copied to the GPU with a **plain pageable, synchronous** `hipMemcpy` (`non_blocking` is ignored; no pinned memory). So batch preparation doesn't overlap compute.

---

## 8. Generation, serving, IO

- **Generation (`generation/`, `models/transformer.cpp::generate`).** `TransformerLM::generate` currently samples on the **CPU** (`probs.to(CPU)` + `std::discrete_distribution`) — the main decode bottleneck — even though a full GPU sampling suite (`ops::multinomial`, `top_k_filter`, `top_p`) exists in `sampling.hip`. `Generator` and beam search (with a correct `KVCache::reorder` for beam crossing) are the richer paths.
- **IO (`io/`, `serialization.cpp`).** safetensors reader/writer (mmap, bounds-checked, JSON-depth-guarded — genuinely hardened) and a custom binary `state_dict` format (magic + per-field EOF checks + dtype/shape validation). **Checkpoints currently save model weights only** — not optimizer state, step, or scheduler position — and there is no resume path, so a crashed long run can't restart correctly (the Adam `state_dict()` machinery exists; the loop just doesn't use it).
- **Web (`web/server.cpp`).** An httplib server with a static UI; inference is serialised under a mutex and calls `generate()`.

---

## 9. Measured performance characteristics (how it behaves today)

This is the most important section for anyone optimizing the library, because the intuition and the measurement disagree.

On the R9700, training the debug model shows **100% GPU utilization** and ~3.6 GB of 32 GB VRAM — the card is saturated, not data-starved. But a HIP dispatch trace over 3 optimizer steps reveals **~13,700 kernel launches per step (~425 per transformer layer)**, and the composition is:

| Kernel category | Share of launches |
|---|---|
| Elementwise (broadcast/binary/scalar/unary) | **~50%** |
| `copy_strided` (transpose→contiguous materialisation) | **~19%** |
| GEMM (all matmul variants) | ~18% |
| norm / softmax / reduce | ~7% |
| silu / adam / misc | ~6% |

So the GPU is 100% busy **launching a swarm of tiny, memory-bound kernels**, not crunching matmuls. The two biggest contributors both trace to the **RoPE + attention data-movement path** (RoPE's elementwise/stack/permute/contiguous chain, and the transpose-into-heads `contiguous()` copies). GEMM — the part you'd expect to dominate — is only ~18% of launches and runs on the **scalar ALU** (FP32 ≈ 13 TFLOPS, ~27% of peak); the **matrix cores are idle** during training. Bandwidth-bound kernels are already efficient (wide softmax ~542 GB/s ≈ 85% of peak).

**Implication:** the eager op-by-op design has no cross-op fusion, so the hot path fragments into thousands of launches. The library's own answer is the *fused-op* pattern (already used for SwiGLU, RMSNorm+Linear, Adam) — the win is to extend it to the remaining hot chains, not to rewrite the framework.

---

## 10. Roadmap — prioritized actions

Merged from the measured profile (§9) and the R9700 specialization analysis, ordered by leverage for *this* GPU. Kernel-level detail (51 items) is in `docs/dev/RDNA4_TUNING_ROADMAP.md`.

**A. Cut launch overhead — the biggest win on the current model (§9: ~70% of launches are tiny elementwise/`copy_strided`).**

| Action | Where | Gain |
|---|---|---|
| Wire the existing fused RoPE kernel + precomputed frequencies (drop per-layer CPU cos/sin + H2D) | `rope.hip`, `functional.cpp`, `gqa_attention.cpp` | Collapses much of the elementwise/`copy_strided` swarm |
| Drop the per-micro-step `loss.item()`; use `PrefetchDataLoader` + pinned/async H2D | `train_tinystories.cpp`, `prefetch_dataloader.h`, `tensor.cpp` | Removes host syncs, overlaps the input pipeline |
| Fuse residual+norm and norm+linear on the hot path (pattern already in `fused_ops`) | `fused_ops.hip`, `transformer_block.cpp` | Fewer launches, less activation traffic |

**B. Put training on the matrix cores — scales with model size.**

| Action | Where | Gain |
|---|---|---|
| BF16-WMMA GEMM covering **all four** trans/no-trans combos + `fp32→bf16` dispatch (fp32 accumulate, no loss-scaling) | `gemm.hip`, `gemm.cpp` | ~2× matmul-bound layers; covers transposed backward (else only ~1/3 benefits) |
| Redesign AMP as a **matmul-boundary autocast** (fp32 masters, fp32 norms/softmax) — retire the whole-model-cast dead-end | `amp.cpp`, `train_tinystories.cpp` | Makes mixed-precision training actually work |
| WMMA flash-attention (fwd+bwd); route seq<512 to it | `flash_attention.hip`, `functional.cpp` | ~4–8× attention at length |
| Wire in the 97-TFLOPS warp-tiled WMMA kernel (transpose-safe) | `gemm.hip` | Higher GEMM ceiling |

**C. Inference & memory.**

| Action | Where | Gain |
|---|---|---|
| Device sampling in `generate()` (wire the existing `ops::multinomial`/`top_k`) | `models/transformer.cpp`, `sampling.hip` | ~1.5–3× decode; kills per-token full-vocab D2H |
| GQA: index KV inside the attention kernel instead of `repeat_kv` | `gqa_attention.cpp`, `attention_ops.hip` | Less decode KV bandwidth |
| Activate streams (`StreamGuard`) + stream-ordered free; allocator block-splitting / finer bins | `stream.cpp`, `allocator.cpp` | Overlap + less VRAM waste |
| `tie_word_embeddings` for real configs; incremental activation free in backward | `config.*`, `engine.cpp` | Fit bigger models |

**D. Correctness & operational — do before scaling / long runs.**

| Action | Where | Why |
|---|---|---|
| **Fix the `head_dim != 64` flash no-op** (OOB garbage for real head dims) | `flash_attention.hip` | Blocks any non-64 head_dim model |
| **Add checkpoint-resume** (save/load Adam state + step + scheduler) | `train_tinystories.cpp`, `serialization.cpp` | Long runs currently can't resume |
| Fix (or formally deprecate) fp16 overflow/NaN classification; prefer BF16 | `half.h` | fp16 loss-scaling detection is broken |
| Save/restore RNG in gradient checkpoint (dropout correctness) | `checkpoint.cpp` | Wrong grads if dropout is checkpointed |
| Add missing tests: GPU flash-bwd numerics, mixed precision, end-to-end convergence, CPU↔HIP parity | `tests/` | Where a training bug would hide |

**Bottom line:** Vesper is a correct, GPU-resident, feature-complete LLM trainer. Its remaining work is *specialization* — cutting launch overhead, putting the already-saturated GPU on the matrix cores, and closing a few scaling landmines — **not a rewrite**.

---

*The refactor/tuning worklogs live under `docs/dev/` (`REFACTOR_CHANGELOG.md`, `RDNA4_TUNING_ROADMAP.md`, `GROK_VESPER_REPORT.md`).*
