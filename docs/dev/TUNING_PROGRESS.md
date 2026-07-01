# Vesper RDNA4 Tuning — Implementation Progress

GPU: AMD R9700, gfx1201 (RDNA4), 32GB, ROCm 7.2.4. Full roadmap: `RDNA4_TUNING_ROADMAP.md` (51 opts).
Build: `build-hip/` (gitignored). Env for GPU runs:
`export PATH=/opt/rocm-7.2.4/bin:$PATH; export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib:/opt/rocm-6.2.0/lib`
Baseline (pre-tuning): GEMM ~13.2 TFLOPS FP32 (~27% peak, 0% matrix cores); wide softmax 542 GB/s (~85% BW).

## Phase 1 — P0 quick wins
- [x] **A: fold-batch→M** (gemm.cpp) — 3D@2D linears now route through the fast 2D kernel. **VALIDATED: 7.5 TFLOPS on [32,128,1024]@[1024,4096] (was ~1 TFLOPS batched) = ~5-7×.** Autograd-correct. Needed a companion fix:
- [x] **sum robustness** (reduction.cpp) — `ops::sum` now handles non-contiguous via a contiguous copy (fold-batch backward produced non-contiguous grads → was throwing). Fixed SwiGLU/GQA gradient-flow.
- [x] **B: FP64→f32 transcendentals** (activation.hip, elementwise.hip) — `exp`/`tanh` doubles → `__expf`/`tanhf`; SiLU family `expf`→`__expf`. Tests pass.
- [x] **E: wave32 fix** (normalization.hip, flash_attention.hip) — WARP_SIZE 64→32 (native wave32). **VALIDATED: softmax GPU==CPU to ~1e-8 across W=64…32000.**
- [x] **C: dispatch overhead** (cat/index_ops/comparison/copy/reduction.hip) — by-value IdxDims (no per-call hipMalloc) for gather/scatter; stream launches (note: Stream::current≡stream 0 by default, so the stream change is currently a no-op).
- [x] **D: build flags** (src/CMakeLists.txt) — baked `--offload-arch=gfx1201` + GPU_TARGETS. **DROPPED** `-fno-math-errno`/`-fgpu-flush-denormals-to-zero`/`-munsafe-fp-atomics`: CONFIRMED they miscompile the where/masked_fill kernels on gfx1201 (index_ops failed with them, passes without). Codegen hazard — do not re-add without per-kernel testing.

### Phase 1 COMPLETE — correctness gate GREEN: full HIP suite 106/107 (only pre-existing M1-fp16 MemoryEfficient fails).
Wins: fold-batch ~5-7× on transformer linears (7.5 TFLOPS); wave32 softmax correct+efficient; f32 transcendentals; reduced dispatch overhead.

## Phase 2 — WMMA GEMM (headliner) — DONE ✓ (integrated + validated, full suite 106/107)
The fp16 matmul path (`gemm_fp16_hip_dispatch`, `!transA && !transB`) now uses RDNA4 matrix cores via the SIMPLE rocWMMA kernel (scalar fallback for transpose). `test_fp16_gemm` PASSES (max rel 8.2e-4), `gemm_comprehensive` 10/10, full HIP suite still 106/107. Integrated fp16 throughput: 1024³ ~25-28, 2048×1024×4096 ~45-58, 4096³ ~59 TFLOPS — 4-7× over the fp32 baseline. (fp16-out uses an LDS-staged fp32→fp16 store since rocWMMA has no direct fp32-acc→fp16 store; fp32-out uses the direct fast store.)

Ran a 3-strategy rocWMMA tournament (ultracode), all GPU-validated on gfx1201:
- **Warp-tiled: 97 TFLOPS** (rel_err 8.2e-4) — LDS double-buffered, size-adaptive tiling; assumes contiguous (no explicit strides).
- **Simple register-blocked: 61 TFLOPS** (rel_err 2.7e-5) — threads lda/ldb/ldc, ragged-safe, clean transA/transB via fragment-layout swap. **Chosen for the first integration** (robustness: strides + fallback).
- Cooperative: 50 TFLOPS.
Baseline was ~13 TFLOPS FP32 scalar → **~5–7.5× on the matrix-core path**. rocWMMA (not raw intrinsics — RDNA4 layout differs).
Integrating the SIMPLE kernel into `gemm_fp16_hip_dispatch`: WMMA fast path for `!transA && !transB` (both output dtypes), scalar fallback otherwise; validate `test_fp16_gemm` + benchmark. Warp-tiled 97-TFLOPS kernel saved (`/tmp/winner_kernel.txt`) as a follow-up upgrade once transB/stride handling is added.

## Phase 2 (original de-risk notes)
- rocWMMA 1.7 compiles for gfx1201 ✓ (raw `__builtin_amdgcn_wmma_*_w32` intrinsic does NOT — RDNA4 layout differs from RDNA3; use rocWMMA).
- Integration point: `gemm_fp16_hip_dispatch` (gemm.hip:955) — swap the scalar `gemm_fp16*_kernel` launch for a rocWMMA 16×16×16 (fp16 in, fp32 accumulate) kernel; same M/K/N/stride plumbing. Batch variant at :986.
- Correctness target: `test_fp16_gemm` (≤1% elements large-error, ~0.016 rel for K=256).
- Fold-batch (A) means transformer linears now hit this 2D fp16 path → WMMA covers QKV/MLP/LM-head.

## Next
1. Confirm index_ops fix (D revert) + full-suite gate (must hold 106/107).
2. Implement rocWMMA GEMM kernel → validate test_fp16_gemm → benchmark (target 40–60+ TFLOPS FP16).
3. Then P1 items (BF16-WMMA FP32 path + AMP, attention WMMA, GQA fusion, norm single-pass).
