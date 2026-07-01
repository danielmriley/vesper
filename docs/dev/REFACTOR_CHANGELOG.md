# Refactor Changelog (`refactor/audit-fixes`)

An audit-driven correctness + hardening pass over Vesper. **65 files, +1712/−741, 4 new test files.** Validated: full HIP suite **106/107** (the one failure is the intentionally-guarded fp16 path), and an *asserts-enabled* rebuild confirmed **0 regressions** — the ~16 assert-on failures that surfaced are all pre-existing (verified identical on a clean `main` baseline).

*(This replaces the former 146 KB `REFACTOR_PLAN.md` grounded spec and the wave-by-wave `REFACTOR_PROGRESS.md`.)*

## Training loop & optimizer
- **T1** — removed the training-freezing skip heuristics + per-param `.item()` sync storms from the loop; kept `clip_grad_norm_(1.0)` + one NaN check.
- **T2** — removed Adam's per-param `sum(grad).item()` NaN-skip (sync storm + bias-correction desync).

## Core & autograd
- **M1** — `data_ptr<T>()` now checks dtype width before reinterpret-cast (was a silent fp16/fp32 aliasing hole).
- **G3** — added the autograd safety guard to in-place `zero_()/copy_()/to_()` (across 6 files).
- **allocator** — fixed static-destruction segfault via a leaky per-device singleton.
- **D5** — fp16 normal-path conversion now round-to-nearest-even (was truncating).
- **L** — null-`grad_handle_`/`storage_` guard on default-constructed tensors.

## Op correctness / gradients
- **G1** — `cat()` now propagates `requires_grad` and has a backward (was silently non-differentiable).
- **G2** — `max/min` backward splits ties correctly; dim-max no longer throws for M>1.
- **G4** — fused RMSNorm+Linear no longer overwrites the correct sub-op autograd graph on CPU.

## GPU kernels (HIP; CUDA twins mirrored)
- **M3** — broadcast online-softmax block-max to all threads (fixes wide-row NaN/zero output).
- **M4** — index bounds guard on gather/scatter (prevents OOB global read/write).
- **M5a/b** — sampling top-k barrier fix (deadlock when `vocab % blockDim != 0`) and CAS-based float `atomicMax` (fixes NaN softmax on all-negative rows).
- **dead-flash** — deleted never-called incomplete flash-attention backward kernels.
- **wave32** — corrected wave64 reduction assumption in `normalization.hip`/`flash_attention.hip`.

## Untrusted-input hardening (safetensors / serialization / model loader)
- **B1a-d** — validate per-tensor offsets/shape vs file size; terminate JSON `parse_string` at EOF; recursion-depth cap; underflow-safe header-size check.
- **B4** — validate dtype enum and cap `name_len`/`ndim`/`data_bytes` before allocation.

## Features & correctness
- **M2** — `generate()` reads Int32 stop/done masks as Int32 (was int8 → wrong lanes for batch>1).
- **B2** — beam search reorders per-layer KV-cache rows on beam crossing (`KVCache::reorder`).
- **B3** — PrefetchDataLoader: fixed `num_workers=0` deadlock, worker-exception `terminate`, `batch_size==0` div0, out-of-order delivery.
- **D2** — `Module` made non-copyable/non-movable; ModuleList copy-insertion removed.
- **D3** — deterministic generation/shuffle (seedable global RNG, honours `manual_seed`).
- **D4** — `cross_entropy_loss` `ignore_index` (masks loss + backward; fixes OOB gather on negative targets).
- **M2nn** — `named_parameters[_ptrs]()` now traverse registered ModuleLists.

## Dead code / hygiene / build
- Deleted dead `models::initialize_weights`, `reduction_cpu.cpp`, `get_rng_locked()`, unused silu_mul allocations.
- Gated unconditional CUDA `.cu` sources behind `USE_CUDA`; `<stdexcept>` added to public headers; IWYU includes on CPU sources.
- `.gitignore` build/CTest artifacts; tempered the overstated CUDA README claim.
- Shrank the fused perf benchmark (~1000×); fixed broken-assert test APIs surfaced by the asserts-on capstone.

## Deferred (flagged for follow-up)
- **D1** — bind optimizers to *live* parameters via `Tensor*` (`parameters_ptrs`) so `to(device)`/member reassignment never desyncs updates. Larger migration; not yet applied.
- Opt-in perf items: contiguous fast-path for binary elementwise CPU ops (**PERF-ELTWISE-CONTIG**); opt-in OpenMP on the serial batched-GEMM loop (**PERF-OPENMP-OPTIN**).
