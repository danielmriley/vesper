# Vesper × AMD Radeon R9700 — Specialization Report

**Purpose:** Assess what Vesper does well and what must improve to fully utilize the AMD Radeon AI PRO R9700 (gfx1201 / RDNA4) on this system.

| | |
|---|---|
| **Date** | 2026-07-01 |
| **Hardware** | R9700, gfx1201, wave32, 64 CU, 32 GB GDDR6 (~640 GB/s) |
| **Software** | ROCm 7.2.4, hipcc 7.2.53211 |
| **Tree** | `/home/daniel/Projects/vesper` · branch `refactor/audit-fixes` |
| **Scope** | Full library: core, autograd, nn, models, generation, optim, data, io, ops (HIP/CUDA/CPU), tests, examples, web |

---

## Executive summary

Vesper is a pure C++17, PyTorch-like DL library (**~179 source files**, **77 public headers**, **19 HIP TUs / ~7.4k LOC**) with a complete LLM stack and **107 CTest targets**. It is HIP-first and already carries meaningful gfx1201 work: `GPU_TARGETS=gfx1201` in CMake, wave32 correctness fixes, batch-folded transformer GEMMs, and rocWMMA FP16 matmul at **25–59 TFLOPS** (vs **~13 TFLOPS** FP32 scalar baseline, validated in `TUNING_PROGRESS.md`).

**The remaining gap is vertical, not horizontal.** Lower layers (tensor → ops dispatch → HIP kernels) are R9700-aware; upper layers (models → generation → examples → web) still default to FP32 training, CPU token sampling, pageable I/O, and the default HIP stream. WMMA is integrated in `gemm.hip` only — attention, fused norm+linear, and the FP32 training path leave matrix cores idle.

### Top 10 actions (ordered by ROI on this GPU)

| # | Action | Where | Expected gain |
|---|--------|-------|---------------|
| 1 | BF16 autocast → WMMA GEMM for training | `amp.cpp`, `gemm.cpp`, `train_tinystories.cpp` | ~2× matmul-bound layers |
| 2 | WMMA flash attention (fwd + bwd) | `flash_attention.hip` | ~4–8× long-seq attention |
| 3 | Batched WMMA for seq \< 512 SDPA | `gemm.hip`, `functional.cpp` | ~2–4×; TinyStories uses seq=256 |
| 4 | Device sampling in `generate()` | `models/transformer.cpp` | ~1.5–3× decode tokens/s |
| 5 | Fix FP16/BF16 elementwise dtype routing | `elementwise.hip` | Correctness gate for mixed precision |
| 6 | Activate streams + stream-ordered allocator | `stream.cpp`, `allocator.cpp` | Enables overlap, graphs, pinned H2D |
| 7 | GQA: index KV in attention, drop `repeat_kv` | `gqa_attention.cpp`, `attention_ops.hip` | ~1.5–3× decode attention BW |
| 8 | Persistent sampling buffers (no per-call malloc/sync) | `sampling.hip` | Per-token latency cut |
| 9 | `PrefetchDataLoader` + pinned host pool | `prefetch_dataloader.h`, `allocator.cpp` | Input pipeline overlap |
| 10 | Complete fused RMSNorm+Linear backward | `fused_ops.hip` | Correctness + atomic→GEMM |

Granular kernel tasks (51 items) live in `RDNA4_TUNING_ROADMAP.md`. Phase status in `TUNING_PROGRESS.md`.

---

## 1. R9700 hardware vs. Vesper today

| Resource | Peak (gfx1201) | Vesper today |
|----------|----------------|--------------|
| FP32 vector ALU | ~48 TFLOPS | FP32 GEMM ~13 TFLOPS (~27%); default training uses this |
| FP16/BF16 WMMA | ~96 TFLOPS | **Only** `gemm_fp16_hip_dispatch` (`!transA && !transB`) |
| FP8 WMMA | ~191 TFLOPS | Unused |
| Memory BW | ~640 GB/s | Wide softmax ~542 GB/s (~85%); scalar norm/elemwise ~40–55% |
| Wavefront | 32 (native) | Fixed in `normalization.hip`, `flash_attention.hip` |
| VRAM | 32 GB | Power-of-2 allocator can over-reserve ~2× on large tensors |

Transformer projections are compute-bound and sit right of the FP16 WMMA ridge — **matrix cores are the main training lever**. Autoregressive decode on this machine is often **latency-bound** by full-vocab D2H and host sampling until the application layer is fixed.

---

## 2. Library map

### 2.1 What ships

| Component | Key paths | R9700 notes |
|-----------|-----------|-------------|
| **Core** | `src/core/` (tensor, allocator, stream) | Streams exist; `set_current()` never called from app code |
| **Autograd** | `engine`, `checkpoint`, `guard` | Checkpoint recomputes slow kernels — kernel speedups compound |
| **NN** | Linear, norms, RoPE, SwiGLU, GQA, AMP, Conv2d | Hot paths → `ops/` → HIP |
| **Models** | `TransformerConfig`, `ModelTransformerBlock`, `TransformerLM` | `generate()` is the main inference bottleneck |
| **Generation** | `Generator`, `BeamSearcher`, `sampling.cpp` | GPU sampling ops exist; beam search is CPU-structured |
| **Optim** | Adam (fused GPU), SGD, Lion, schedulers, `GradScaler` | Adam is solid; GradScaler D2H-scans every grad |
| **Data** | `DataLoader`, `PrefetchDataLoader` (header-only) | Example uses sync loader; pinned memory not implemented |
| **IO** | safetensors, model_loader, serialization | Not hot-path critical |
| **Ops / HIP** | 20 host dispatch `.cpp` → 19 `.hip` | Primary tuning surface |
| **Ops / CUDA** | 18 `.cu` mirrors | `USE_CUDA=OFF`; not CI-validated |
| **Ops / CPU** | 6 reference `.cpp` | Dispatch fallbacks + `build-cpu/` harness |
| **Examples** | `train_tinystories.cpp` | FP32, sync `DataLoader`, AMP unused |
| **Web** | `vesper_server` + static UI | Mutex-serialized; calls `generate()` |
| **Tests** | 107 `add_test` in `tests/CMakeLists.txt` | HIP gate: 106/107 (`MemoryEfficientTraining` known fail) |

**Build targets:** `vesper` (static), tests, `train_tinystories`, `vesper_server`. Defaults: `USE_HIP=ON`, `GPU_TARGETS=gfx1201`, per-`.hip` `--offload-arch=gfx1201`. Global `-ffast-math`/FTZ dropped — breaks `where`/`masked_fill` on gfx1201.

### 2.2 Dispatch matrix (host → device)

| Dispatch | HIP | R9700 status |
|----------|-----|--------------|
| `gemm.cpp` | `gemm.hip` | rocWMMA FP16 ✓; FP32 scalar; batch-fold 3D@2D ✓ |
| `flash_attention.cpp` | `flash_attention.hip` | Scalar QK/PV; dropout ignored on GPU |
| `normalization.cpp` | `normalization.hip` | Wave32 ✓; multi-pass LayerNorm/RMSNorm |
| `elementwise.cpp` | `elementwise.hip` | float4 contiguous ✓; broadcast + FP16 broken |
| `sampling_ops.cpp` | `sampling.hip` | Full GPU suite; **unused by `generate()`** |
| `fused_ops.cpp` | `fused_ops.hip` | Fwd scalar GEMM; **bwd TODO / wrong grads** |
| `attention_ops.cpp` | `attention_ops.hip` | `repeat_kv` wastes KV bandwidth |
| `cat.cpp` | `cat.hip` | By-value args ✓; rare fallback uses `hipStreamSynchronize` only |
| `index_ops.cpp` | `index_ops.hip` | By-value args ✓ (malloc/sync removed) |
| `embedding.cpp` | `embedding.hip` | Scalar gather + per-element int64 div |
| `rope` | `rope.hip` | float4 ✓; 16-thread blocks at head_dim=64 |
| `random.cpp` | `random.hip` | Hash PRNG on stream 0 |
| `im2col` / `pooling` | `.hip` | CV path; inherits GEMM tuning via im2col→GEMM |

View/shape ops (`view_ops.cpp`, `stack.cpp`) are host-only by design.

### 2.3 Application hot path

**Training (`train_tinystories.cpp`):** `seq_len=256`, `batch_size=8`, FP32 weights, sync `DataLoader`, comments note FP16/GradScaler as future work. Attention uses materialized scores (seq \< 512 path), not flash.

**Inference (`TransformerLM::generate`):** Per token — GPU softmax → **`probs.to(CPU)`** → `std::discrete_distribution` → H2D token → `forward_with_cache`. Ignores `ops::multinomial` / `top_k_filter` in `sampling_ops.cpp`.

**Serving (`web/server.cpp`):** 4-thread httplib pool, **mutex** on model, SentencePiece tokenize, calls `generate()` above.

**Beam search (`beam_search.cpp`):** CPU `std::priority_queue`, `log_probs.to(CPU)`; in-file comment documents GPU upgrade (preallocated tensors + `topk` + `KVCache::reorder`).

---

## 3. What is done well

1. **gfx1201 targeting** — CMake defaults; rocWMMA emits matrix ops on RDNA4.
2. **Wave32 correctness** — Prior wave64 bug halved norm/softmax denominators; fixed and validated to ~1e-8 vs CPU.
3. **Batch-fold GEMM** — `[B,T,C]@W` → `[B×T,C]@W`; ~5–7× on transformer linears (7.5 TFLOPS on MLP-shaped case).
4. **rocWMMA FP16 GEMM** — 25–59 TFLOPS integrated; 97 TFLOPS warp-tiled variant benchmarked for later. Must use rocWMMA, not raw `__builtin_amdgcn_wmma_*`.
5. **Fused Adam on GPU** — `adam_update_`, `lerp_`, `addcmul_` with float4; optimizer is not the training bottleneck.
6. **Dispatch hygiene** — cat/index by-value kernel args; no per-call malloc on common paths.
7. **Bandwidth kernels** — Wide softmax ~85% peak BW; residual add, SwiGLU, RoPE body ~70–85%.
8. **LLM completeness** — RoPE, GQA, KV-cache, SwiGLU, flash attention (long seq), CE `ignore_index`, checkpointing, safetensors, web server — rare for zero-BLAS C++.
9. **Test + docs culture** — 107 tests; `RDNA4_TUNING_ROADMAP.md`, `TUNING_PROGRESS.md`, `REFACTOR_PROGRESS.md`.

---

## 4. What should improve (full priority list)

### P0 — Throughput & correctness blockers

| ID | Issue | Files | Fix |
|----|-------|-------|-----|
| P0-1 | Attention 100% scalar VALU | `flash_attention.hip`, `functional.cpp`, `gemm.cpp` | rocWMMA FA2; batched WMMA for seq\<512 |
| P0-2 | Training defaults to FP32 | `train_tinystories.cpp`, `amp.cpp`, `linear.cpp` | BF16 autocast → WMMA; FP32 master weights |
| P0-3 | FP16 elementwise reads bytes as float | `elementwise.hip` | Dtype-templated `half2` kernels |
| P0-4 | Fused RMSNorm+Linear bwd wrong | `fused_ops.hip:387` | WMMA weight-grad; norm Jacobian |

### P1 — Inference, pipeline, memory

| ID | Issue | Files | Fix |
|----|-------|-------|-----|
| P1-1 | CPU sampling in decode | `models/transformer.cpp` | Wire `ops::multinomial` / `top_k_filter` |
| P1-2 | CPU beam search | `beam_search.cpp` | GPU `topk` + preallocated beam tensors |
| P1-3 | Sampling malloc+sync per call | `sampling.hip` | Persistent device buffers |
| P1-4 | Streams never set | `stream.cpp`, allocators | `StreamGuard`; stream-ordered free |
| P1-5 | GQA `repeat_kv` expansion | `gqa_attention.cpp` | KV head index inside attention kernel |
| P1-6 | Pageable I/O, sync loader | `allocator.cpp`, `prefetch_dataloader.h`, example | `hipHostMalloc` pool; use prefetch loader |
| P1-7 | GradScaler full grad D2H | `grad_scaler.cpp:41` | GPU `isfinite` reduction |
| P1-8 | Norm multi-pass, no fusion | `normalization.hip` | Single-pass LN; fused `add_rmsnorm` |
| P1-9 | RoPE half-wave blocks | `rope.hip` | 128–256 thread blocks |
| P1-10 | Generator stop-mask D2H | `generator.cpp` | Device-side flags |

### P2 — Peak tuning & polish

| ID | Issue | Files |
|----|-------|-------|
| P2-1 | Warp-tiled 97-TFLOPS WMMA (transpose-safe) | `gemm.hip` |
| P2-2 | gfx1201 tile/occupancy autotune | `gemm.hip`, flash |
| P2-3 | FP8 WMMA inference (LM-head) | `gemm.hip` |
| P2-4 | HIP graph capture for decode step | `transformer.cpp` |
| P2-5 | Finer allocator size classes | `allocator.cpp` |
| P2-6 | DMA / float4 for contiguous copy/cast | `copy.hip`, `cast.hip` |
| P2-7 | Flash dropout silently ignored + O(N²) mask alloc | `flash_attention.cpp` |
| P2-8 | Web mutex → request queue / batching | `web/server.cpp` |
| P2-9 | ~16 assert-on test failures (pre-existing) | see `REFACTOR_PROGRESS.md` |

---

## 5. Stack gap (where time goes)

```
  APPLICATION     FP32 train · CPU sample · sync loader · mutex infer   ← largest gap
       ↓
  DISPATCH        batch-fold ✓ · AMP unused · SDPA split by seq length
       ↓
  HIP KERNELS     WMMA GEMM ✓ · attention scalar · sampling GPU (unused) · stream 0
       ↓
  R9700           96 TFLOPS WMMA · 640 GB/s · wave32
```

**TinyStories example (seq=256, FP32, current code):**

| Component | ~Step share | R9700 efficiency |
|-----------|-------------|----------------|
| MLP/QKV GEMM | ~70% | ~27% FP32 peak |
| Attention (materialized, scalar batched) | ~15% | ~10% |
| Norm + elementwise | ~10% | ~50% BW |
| Adam + H2D | ~5% | Adam good; H2D pageable |

Fully optimized (P0 + P1 + mixed precision): estimated **~3–5×** training step vs original FP32 baseline.

---

## 6. Implementation phases

| Phase | Status | Contents |
|-------|--------|----------|
| **A** | ✓ Done | gfx1201 CMake, wave32, batch-fold, f32 transcendentals, dispatch hygiene, rocWMMA FP16 |
| **B** | Next | P0-2, P0-1, P0-3, P1-1 (kernel + app coupling) |
| **C** | Then | P1-4, P1-3, P1-6, P1-5, P1-8 (pipeline + memory) |
| **D** | Then | P0-4, P1-7, P2-7, P2-9 (correctness debt) |
| **E** | Later | P2-1–P2-4, P2-8 (peak tuning + serving) |

---

## 7. Build & validation

```bash
export PATH=/opt/rocm-7.2.4/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib:$LD_LIBRARY_PATH
mkdir -p build-hip && cd build-hip
cmake .. -DUSE_HIP=ON -DCMAKE_CXX_COMPILER=/opt/rocm-7.2.4/bin/hipcc
make -j$(nproc) && ctest --output-on-failure
```

Rebuild if `libamdhip64.so.6` errors appear — ROCm 7.2.4 provides `.so.7`.

**Gate tests after each change:**

| Change type | Tests |
|-------------|-------|
| WMMA GEMM | `FP16GemmTests`, `GemmComprehensiveTests` |
| Attention | `FlashAttentionTests`, `AttentionComprehensiveTests` |
| Norm / wave32 | `SoftmaxComprehensiveTests`, `NormalizationIntegrationTests` |
| End-to-end | `TransformerModelChapter33Tests`, `TrainingStabilityTests` |
| Generation | `GenerationChapter33Tests` |
| Mixed precision | `HalfPrecisionTests`, `MemoryEfficientTrainingTests` |

**Benchmarks** (gfx1201, `TUNING_PROGRESS.md`): FP32 GEMM 1024³ ~13 TFLOPS · FP16 WMMA 1024³ ~25–28 · MLP-shaped 2048×1024×4096 ~45–58 · wide softmax ~542 GB/s · fold-batch linear ~7.5 TFLOPS (was ~1).

---

## 8. Risks

| Risk | Mitigation |
|------|------------|
| WMMA numeric drift | FP32 accumulate; `test_fp16_gemm` |
| rocWMMA required on RDNA4 | No raw `__builtin_amdgcn_wmma_*` |
| Global fast-math | Per-kernel only; breaks masked ops on gfx1201 |
| Attention bwd complexity | CPU reference in `FlashAttentionTests` |
| Async without stream-ordered alloc | Sequence P1-4 before removing syncs |
| Stale build / wrong `.so` | Use `build-hip` + ROCm 7.2.4 `LD_LIBRARY_PATH` |

---

## 9. Conclusion

Vesper is a strong, feature-complete C++ LLM library with real gfx1201 progress in GEMM and wave32 correctness. It is **not yet fully specialized for the R9700** because the code that *calls* the kernels still trains in FP32, samples on CPU, and never activates async streams or pinned I/O.

The highest-leverage work is now **vertical**: wire existing GPU sampling into `generate()`, enable BF16→WMMA training, and WMMA-ize attention — not merely add more scalar kernels. The Top 10 table (§Executive summary) is the actionable entry point; `RDNA4_TUNING_ROADMAP.md` has line-level detail for each kernel change.

---

## References

| Doc | Role |
|-----|------|
| `RDNA4_TUNING_ROADMAP.md` | 51 kernel optimizations with citations |
| `TUNING_PROGRESS.md` | Phase A–B status, measured TFLOPS |
| `REFACTOR_PROGRESS.md` | Correctness refactor, assert-on findings |
| `src/CMakeLists.txt` | HIP sources, `GPU_TARGETS` |
| `tests/CMakeLists.txt` | All 107 test targets |
| `src/generation/beam_search.cpp` | Documented GPU beam-search path |