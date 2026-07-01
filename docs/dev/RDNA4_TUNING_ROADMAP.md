# Vesper RDNA4 (gfx1201 / Radeon R9700) Performance Tuning Roadmap


Hardware: 64 CU @ ~2.9GHz; ~48 TFLOPS FP32, ~96 TFLOPS FP16/BF16 WMMA, ~191 TFLOPS FP8; 32GB GDDR6 ~640 GB/s; native wave32; rocWMMA 1.7 installed.

Measured baseline: GEMM ~13.2 TFLOPS FP32 (~27% peak, 0% matrix cores); wide softmax 542 GB/s (~85% BW peak).



## RDNA4 (gfx1201 / Radeon R9700) GEMM + attention-matmul optimization roadmap for Vesper (src/ops/hip/gemm.hip, src/ops/gemm.cpp, attention path)

**Bottleneck:** Hardware (verified gfx1201, ROCm 7.2.4, Navi 48): 64 CU @ ~2.9 GHz; ~48 TFLOPS FP32 (dual-issue VOPD); ~96 TFLOPS FP16/BF16 dense via WMMA 16x16x16 (~191 with 2:4 sparsity); ~191 TFLOPS FP8 WMMA; 32 GB GDDR6, 256-bit @ ~640 GB/s. The compute-bound ridge is ~150 FLOP/byte for FP16-WMMA and ~75 for FP32, and transformer linears (K,N ~512-8192) sit far right of it => COMPUTE-bound => matrix cores are THE lever. Reality of the code: zero WMMA/MFMA/rocBLAS anywhere (grep-confirmed); all 9 GEMM kernels do scalar `rC[i][j] += rA[i]*rB[j]`, and the FP16 kernels even convert half->float first (gemm.hip:728/734/737, 821/826/928) so 'FP16' buys only LDS/HBM storage halving, not throughput. Worse, the data flow funnels EVERYTHING through the weakest kernel: Linear/FusedQKV/MLP/LM-head are [B,T,C]@[C,N] (3D@2D, linear.cpp:43,78) and SDPA scores/output are 4D@4D (functional.cpp:438,459); both hit the batched dispatch -> gemm_batch_tiled_kernel (gemm.hip:1346), which is single-buffered, scalar-load, fixed 64x64, no float4. The good double-buffered+float4 kernel (gemm_large_tile_vectorized_kernel, gemm.hip:162) is only reachable on the non-batched 2D path (gemm.hip:1257) and is therefore essentially never used during training. Estimated current efficiency: FP32 path ~10-18 TFLOPS (~20-35% of FP32 peak, fragmented further by per-batch tiling); 'FP16' path ~8-15 TFLOPS-equivalent (~10-15% of matrix peak). Realistic WMMA target 50-67 TFLOPS (55-70% of 96) => 3-6x on the dominant FLOP sink. rocWMMA 1.7 headers are present (/opt/rocm-7.2.4/include/rocwmma/rocwmma.hpp) plus CK amd_wmma.hpp, so the fragment API is available and lowers WMMA effort. Secondary issues: flash kernel hardcodes WARP_SIZE=64 (wrong for wave32; flash_attention.hip:17) and is fully scalar; vectorization.h has float4 but no half-vector loads.


### [P0] Replace scalar half->float FMA with WMMA 16x16x16 FP16/BF16 GEMM, FP32 accumulate (the P0 win)  (L effort, compute, lever=WMMA)
- Kernel: gemm_fp16_kernel / gemm_fp16_fp32out_kernel / gemm_fp16_batch_kernel (gemm.hip:661, 765, 857)
- Current: All three FP16 kernels stage __half tiles in LDS then do `rA[i]=__half2float(...); rB=__half2float(...); rC[i][j]+=rA[i]*rB` (gemm.hip:728-737, 821-829, 920-928). Pure FP32 VALU; zero matrix-core use. Dispatched from gemm_fp16_hip_dispatch (gemm.hip:955) and the batch variant (gemm.hip:986).
- Change: Write a gemm_wmma kernel using rocWMMA fragments (rocwmma.hpp, already installed) or __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 / _bf16_. Tiling: workgroup tile BM=BN=64-128, BK=16-32; cooperatively load FP16 A[BM][BK] and B[BK][BN] into LDS (reuse uint4/half8 vectorized loads, pad rows to kill 16-bit bank conflicts), double-buffer. Each wave32 wavefront owns a 32x32 warp-tile = 2x2 grid of 16x16 fragments (A frag = 8 FP16/lane, B frag = 8 FP16/lane, C acc = 8 FP32/lane), looping K in steps of 16 with rocwmma::mma_sync, accumulating in FP32 the whole K-loop; convert to FP16/FP32 only on epilogue store. Route both fp16 dispatch and fp16-batch dispatch (blockIdx.z) into it.
- Expected: 3-6x on FP16 GEMM. Grounding: test_fp16_gemm 1024^3 (2.1 GFLOP) currently ~10-15 TFLOPS-equiv; WMMA at 55-70% of 96 TFLOPS = ~53-67 TFLOPS. The MLP-shaped 2048x1024x4096 test benefits most (large, compute-bound, far right of the FP16 ridge).
- Risk: FP16 dynamic-range overflow on activations (mitigate: FP32 accumulate already in place + expose BF16 variant); fragment<->LDS layout bugs; must pass test_fp16_gemm tolerance. rocWMMA lowers risk vs raw intrinsics.

### [P0] Fold batch into M for activation x weight linears: B small GEMMs -> one big 2D GEMM  (S effort, occupancy, lever=fusion)
- Kernel: gemm() batched dispatch for 3D@2D linears (gemm.cpp:130-138) -> gemm_batch_tiled_kernel (gemm.hip:1346)
- Current: Linear/FusedQKV/MLP/LM-head call matmul([B,T,C], weight^T) (linear.cpp:43,78). a_rank=3,b_rank=2 takes the broadcast-B branch (gemm.cpp:130) with batch_count=B, M=T, dispatching gemm_batch_tiled_kernel: scalar, single-buffered, 64x64, re-reads the weight tile for every batch, fragments work, and can never reach gemm_large_tile_vectorized_kernel (gemm.hip:162).
- Change: When B is 2D and broadcast across the batch (the activation x weight case), reshape A from [B,T,C] to [B*T,C], call the 2D path once, reshape C from [B*T,N] back to [B,T,N]. Pure host-side view (contiguous activations already satisfy this). This collapses B fragmented GEMMs into a single [B*T,C]@[C,N], maximizes weight reuse via L2/LDS, fills the machine, AND makes both the existing vectorized 2D kernel and the new WMMA 2D kernel reachable for the bulk of transformer FLOPs.
- Expected: 1.5-2.5x standalone in FP32 (reaching the double-buffered/vectorized kernel + weight reuse + occupancy); and it is the prerequisite multiplier that lets WMMA (P0) cover QKV/MLP/out-proj/LM-head. Cheapest item with the widest blast radius.
- Risk: Low - shape-only. Must preserve transA/transB semantics, verify contiguity (insert .contiguous() if a view is non-trivial), and keep the autograd reshape symmetric on the backward GEMMs.

### [P1] Route FP32 transformer matmuls through BF16-WMMA with FP32 accumulate (opt-in mixed precision)  (M effort, compute, lever=dtype)
- Kernel: FP32 transformer GEMM path (gemm_large_tile_vectorized_kernel gemm.hip:162 / gemm_batch_tiled_kernel gemm.hip:1346); AMP wiring (src/nn/amp.cpp)
- Current: Default training is FP32 (Linear weights are Float32, linear.cpp:19,63; train_tinystories runs no autocast), so GEMMs hit the scalar FP32 kernels. AMP infrastructure exists (amp.cpp casts params to FP16/BF16, keeps FP32 master weights) but when enabled it currently lands on the scalar gemm_fp16 kernels, not matrix cores.
- Change: Add a GEMM precision policy: for matmul-bound layers, cast A/B to BF16 and run the P0 WMMA kernel with FP32 accumulate. Prefer BF16 over FP16 for activations (same exponent range as FP32, no loss scaling). Wire AutocastContext (amp.cpp:106) so autocast dispatches to the WMMA kernel; keep FP32 master weights (already done) for the optimizer.
- Expected: ~3x on the matmul-bound layers vs the FP32 scalar kernel (96 vs 48 TFLOPS peak, plus the scalar kernel only realizes ~25-35% of FP32 peak while WMMA realizes ~55-70%). End-to-end training step ~1.8-2.5x depending on non-GEMM (norm/softmax/elementwise) share.
- Risk: Numerical/convergence - mitigate with BF16 inputs + FP32 accumulate + FP32 master weights; validate the loss curve on TinyStories before defaulting it on.

### [P1] Batched-WMMA for attention B*H matmuls (Q@K^T and P@V, seq < 512)  (M effort, compute, lever=WMMA)
- Kernel: SDPA score/output matmuls -> gemm_batch_tiled_kernel (functional.cpp:438,459; gemm.hip:1346)
- Current: For seq_len < 512 (e.g. TinyStories), SDPA does scores=matmul(Q,K^T) (functional.cpp:438) and out=matmul(P,V) (functional.cpp:459) as 4D@4D -> batched scalar gemm_batch_tiled_kernel with batch=B*H, M=N=S, K=head_dim=64. Genuinely batched (distinct K per b,h) so it cannot be folded into M like the linears; also materializes the S x S score matrix in HBM.
- Change: Add a batched (blockIdx.z = b*h) variant of the P0 WMMA kernel for the two attention GEMMs. head_dim=64 => K=64 = 4 WMMA K-steps; pad head_dim=80/96 cases to 16. Run in BF16/FP16 with FP32 accumulate. Longer term, fuse Q@K^T -> softmax -> @V (flash-attention-2 style) to keep scores in LDS/registers and kill the S x S HBM round-trip.
- Expected: 2-4x on the attention GEMMs (matrix cores vs scalar). Attention is a major FLOP sink at longer T; the fused variant adds memory-bandwidth savings on top.
- Risk: Small-K efficiency at head_dim=64 (acceptable; 128 is better); correctness vs softmax masking; reuse of the P0 core keeps risk bounded.

### [P2] Interim: give the batched FP32 kernel double-buffering + float4 (until WMMA/fold supersede it)  (S effort, latency, lever=double-buffering)
- Kernel: gemm_batch_tiled_kernel (gemm.hip:1346, compute loop 1406-1419, launch 1477)
- Current: The batched kernel is single-buffered with scalar global loads and fixed 64x64 tiles - strictly weaker than gemm_large_tile_vectorized_kernel (gemm.hip:162), which has double-buffered LDS prefetch + float4 loads but is only wired to the 2D path (gemm.hip:1272). The batched path (the actual training workhorse) never gets that treatment.
- Change: Either dispatch per-batch into the existing gemm_large_tile_vectorized_kernel with a batch pointer offset (blockIdx.z), or port its double-buffer + float4 loads into gemm_batch_tiled_kernel. Removes __syncthreads stalls (compute overlaps next-tile LDS load) and halves load instructions via 128-bit transactions matching the 128-byte cache line.
- Expected: 1.3-1.8x on FP32 batched GEMM. Pure stop-gap: superseded once P0 (WMMA) + P1 (batch-fold) land, so do it only if FP32 training must stay fast in the interim.
- Risk: Low - same algorithm, established pattern already in the codebase; watch the float4 alignment guard for non-multiple-of-4 N/K.

### [P2] Tile/LDS/occupancy autotune for gfx1201 (wave32, 64KB LDS, BK=32)  (M effort, occupancy, lever=LDS)
- Kernel: New WMMA kernel + gemm_large_tile_vectorized_kernel launch bounds (gemm.hip:161 __launch_bounds__(256,2))
- Current: Tiles and launch bounds were chosen generically: BK=16 across kernels, LDS padded +1/+4, __launch_bounds__(256,2) for large / (64,4) for small. 256 threads = 8 wave32 wavefronts on RDNA4; nothing is tuned to the 64KB LDS / 2-waves-per-SIMD reality of Navi 48.
- Change: For the WMMA kernel pick BK=32 (halves K-loop sync count, feeds the matrix unit better) and size BM/BN so double-buffered FP16 LDS stays <=~32 KB to keep >=2 workgroups/CU. Sweep BM/BN in {64,128} and waves/WG in {4,8}; replace +1 scalar padding with a swizzle for the FP16 tiles. Verify __launch_bounds__ against the wave32 register file via --save-temps ISA inspection.
- Expected: 1.1-1.3x on top of WMMA from higher occupancy and fewer syncs.
- Risk: Low; needs a small offline autotune harness and ISA/occupancy inspection (rocprof), not algorithmic change.

### [P2] Fix wave32 assumption and WMMA-ize flash attention (seq >= 512)  (L effort, compute, lever=WMMA)
- Kernel: flash_attn_fwd_kernel (src/ops/hip/flash_attention.hip:17,33; scalar QK/PV dot loops 121-160)
- Current: flash_attention.hip:17 hardcodes `constexpr int WARP_SIZE = 64; // AMD Wavefront size is typically 64` - wrong for RDNA4 wave32 - and HEAD_DIM=64 is fixed. The score (Q@K^T) and output (P@V) are fully scalar nested-d loops (lines 121-160). Used by SDPA only for seq_len >= 512 (functional.cpp:423-427).
- Change: Correct all wavefront assumptions to 32 (affects register/occupancy math and any cross-lane reductions), stage K/V as FP16 in LDS, and replace the in-kernel QK^T and PV with 16x16x16 WMMA fragments (flash-attention-2 style with online softmax between the two matmuls). Generalize HEAD_DIM beyond 64.
- Expected: 2-4x for long-sequence attention; also corrects latent wave32 occupancy waste. Only matters once seq_len >= 512 (not TinyStories), hence P2.
- Risk: Medium - online-softmax rescaling interleaved with WMMA accumulation is fiddly; needs careful validation against test_flash_attention.

### [P2] FP8 WMMA path for inference (LM-head / weight-heavy GEMMs)  (L effort, compute, lever=dtype)
- Kernel: Inference GEMMs: LM-head + MLP (via the P0 WMMA kernel)
- Current: No FP8 anywhere. RDNA4 supports FP8 (E4M3/E5M2) WMMA at ~2x FP16. LM-head [B*T,C]@[C,vocab] and MLP projections are large and compute-bound, ideal FP8 candidates at inference.
- Change: Add an FP8 WMMA variant (FP8 inputs, FP16/FP32 accumulate) for inference-only GEMMs with per-tensor/per-channel scales; quantize weights offline, keep activations FP8 with calibration. Gate behind an inference precision flag.
- Expected: ~2x over FP16-WMMA on the FP8-eligible GEMMs (191 vs 96 TFLOPS peak), inference only.
- Risk: Higher - quantization accuracy needs calibration/validation; training stays BF16/FP16. Forward-looking, do after P0/P1 prove out.


## GPU attention kernels (Flash-Attention fwd/bwd + GQA repeat_kv) on AMD Radeon R9700 / gfx1201 (RDNA4): src/ops/hip/flash_attention.hip, src/ops/hip/attention_ops.hip, src/nn/gqa_attention.cpp, src/nn/functional.cpp.

**Bottleneck:** The live SDPA path (functional.cpp:425 -> ops::flash_attention for N>=512) is a pure scalar/SIMT FP32 kernel. For N>=512, D=64, attention is firmly COMPUTE-bound (arithmetic intensity ~O(N) flop/byte >> the ~75 flop/byte FP32 ridge of ~48 TFLOPS / ~640 GB/s), yet: (1) the matrix cores are 100% idle — QK^T and P*V run as per-thread scalar FMA loops on the VALU (verified: zero WMMA in the repo); (2) one-thread-per-query forces giant per-lane register arrays (fwd: q_reg[64]+acc_i[64]+scores[64] ~= 192 VGPR/lane; bwd: k_reg+v_reg+dk_acc+dv_acc = 256 floats/lane that spill to scratch), capping resident waves and killing latency hiding; (3) the backward serializes on 4096 global atomicAdd-to-dQ per thread; (4) everything is FP32, so DRAM traffic and LDS footprint are 2x what BF16 needs and matrix cores are unusable. Rough estimate: forward sustains <~5-8% of the card's ~48 TFLOPS FP32 peak and 0% of its ~95 TFLOPS BF16 WMMA peak; backward is worse (atomics + scratch spill). Net headroom on the SDPA path is roughly 8-20x. The single highest-impact lever is a BF16 WMMA (16x16x16) rewrite of flash-attention with FlashAttention-2 online-softmax scheduling; LDS-staged tiles and a larger cooperative block come along with it and fix the register/occupancy problems. repeat_kv (GQA) is a separate pure-bandwidth waste that matters most for decode.


### [P0] Rewrite forward QK^T and P*V as BF16 WMMA 16x16x16 matrix ops with FA2 online-softmax (the headline win)  (L effort, compute, lever=WMMA)
- Kernel: flash_attn_fwd_kernel (src/ops/hip/flash_attention.hip)
- Current: flash_attention.hip:71 maps one thread to one query row; :119-136 computes S=Q*K^T as a per-thread scalar loop (dot += q_reg[d]*sK[k][d], 64x64 FMA/thread); :144-161 computes the P*V update the same scalar way. All FP32 on the VALU; matrix cores 100% idle (grep: 0 WMMA in repo). This is the live path for N>=512 (functional.cpp:425-427).
- Change: Cast Q/K/V to BF16 and restructure to ONE block per BLOCK_M query tile, wavefronts cooperating via rocWMMA (or __builtin_amdgcn_wmma_f16_16x16x16_f32). QK^T: for the score tile S[BM x BN], loop d=0..D step 16 accumulating S_frag(FP32) += wmma(Q_frag, K_frag); D=64 -> 4 k-steps, and FP32 accumulators keep softmax accurate despite BF16 inputs. Run the online softmax on the FP32 S fragments (row-max/row-sum via LDS staging or fragment+DPP reductions), form P=exp2(S - m) and pack to BF16. P*V: O_frag(FP32) += wmma(P_frag, V_frag) looping BN step 16; on each max update rescale O_frag and l_i by exp2(m_old - m_new) (FlashAttention-2 schedule, fewer rescales than the current per-block rescale at :144-148). K/V tiles stay resident in LDS as BF16 and feed every wavefront. D=64 is a clean 4x16 fit. Verify the build targets gfx1201 first (see build item).
- Expected: ~4-8x on forward SDPA for N>=512. Reasoning: BF16 WMMA ~doubles matmul peak (~48->~95 TFLOPS) AND offloads QK^T/P*V off the VALU so the VALU is free for exp/softmax in parallel; combined with the occupancy recovery from dropping the 192-VGPR arrays, naive-scalar->WMMA FA rewrites on RDNA-class HW typically land 4-8x. QK^T+P*V are ~all the flops at D=64, so Amdahl headroom is the whole kernel.
- Risk: Numerical: BF16 QK^T inputs — mitigated by mandatory FP32 WMMA accumulators and FP32 LSE (already FP32 at flash_attention.cpp:74). Complexity: fragment<->lane mapping for the softmax reduction and causal mask; start with rocWMMA to avoid hand-coding lane layout. Requires head_dim % 16 == 0 (D=64 OK); keep the scalar kernel as a fallback for odd D (dispatch already stubs D!=64 at :316).

### [P0] Eliminate per-thread register arrays via cooperative LDS-staged tiling (kills spilling, restores occupancy)  (M effort, register-spilling, lever=LDS)
- Kernel: flash_attn_fwd_kernel + flash_attn_bwd_kernel_final (src/ops/hip/flash_attention.hip)
- Current: Forward keeps q_reg[HEAD_DIM] (:72), acc_i[HEAD_DIM] (:62) and scores[BLOCK_N] (:116) live per lane (~192 VGPR/lane). Backward is worse: k_reg+v_reg+dk_acc+dv_acc = 4xHEAD_DIM = 256 floats/lane (:203-204), at/over the wave32 256-VGPR architectural limit, so it spills to scratch (global-backed). Block is only 64 threads = 2 wave32 (:321,:351).
- Change: Move Q/K/V/S/P tiles into LDS (BF16, ~16KB for K+V) and let a 128-256 thread block cooperatively process a query tile, so each lane holds only WMMA fragments + scalars (~32-64 VGPR) instead of 64-192-element private arrays. This is the structural substrate the WMMA item builds on, and it independently removes the backward scratch spill. Vectorize the LDS<->global staging (float4/BF16x8) so the strided per-row loads at :101-102 become coalesced.
- Expected: Mostly enabling for the P0 WMMA item; standalone (SIMT-only, if WMMA is deferred) ~1.5-2.5x from dropping VGPR/lane ~192->~64 (more resident wave32 to hide DRAM/exp latency) and removing the backward scratch traffic.
- Risk: Done together with the P0 rewrite to avoid churning the kernel twice. LDS bank-conflict care needed on the S/P staging (pad to avoid 32-bank conflicts on BF16).

### [P1] Pin CMAKE_HIP_ARCHITECTURES=gfx1201 and confirm wave32 (prerequisite that unblocks WMMA)  (S effort, compute, lever=WMMA)
- Kernel: Build configuration (CMakeLists.txt) + flash_attention.hip:17
- Current: No explicit gfx target / CMAKE_HIP_ARCHITECTURES in CMakeLists.txt (grep found none). flash_attention.hip:17 hardcodes constexpr WARP_SIZE = 64 with comment 'AMD Wavefront size is typically 64' — wrong for RDNA4 (native wave32) and misleading, though currently unused in the kernel body.
- Change: Set the HIP offload arch explicitly to gfx1201 (--offload-arch=gfx1201 / CMAKE_HIP_ARCHITECTURES gfx1201) so __builtin_amdgcn_wmma_* / rocWMMA actually emit matrix instructions instead of being scalarized or failing to compile; confirm the default wave32 ABI. Delete/fix the WARP_SIZE=64 constant (set 32) to stop propagating the wave64 mental model. Do this before starting the P0 rewrite.
- Expected: 0x by itself, but it is the gate for the 4-8x P0 win. If the project is currently building for a default or mismatched arch, WMMA intrinsics won't lower to matrix ops at all.
- Risk: Low. Just ensure the CI/build box and the R9700 agree on gfx1201; keep a multi-arch list if other GPUs must build.

### [P1] Remove global atomicAdd-to-dQ and the 256-float spill; move backward to BF16 WMMA  (L effort, latency, lever=WMMA)
- Kernel: flash_attn_bwd_kernel_final (src/ops/hip/flash_attention.hip)
- Current: flash_attention.hip:280-285 does an atomicAdd to global dQ for every (kv_idx, m, d): BLOCK_M x HEAD_DIM = 4096 global atomics per thread, all KV blocks contending on the same dQ rows -> heavy serialization. Plus the 256-float/lane spill (:203-204) and FP32 scalar score/dp loops (:252-278).
- Change: (a) Kill the global atomics: use the FA2 backward schedule — a dKdV pass parallelized over KV tiles plus a separate dQ pass parallelized over query tiles (each writes its own rows once), or accumulate dQ in LDS per block and write once; dQ zero-init at :348 stays. (b) Recast score=Q*K^T, dP=dO*V^T, dS, dK=dS^T*Q, dV=P^T*dO as BF16 WMMA 16x16x16 with FP32 accumulators. (c) Replace persistent k_reg/v_reg/dk_acc/dv_acc with LDS-staged K/V + fragment accumulators to end the scratch spill.
- Expected: ~3-6x on backward. Reasoning: backward is currently the slowest kernel — global-atomic serialization + scratch spill + scalar FP32 stack — so it benefits from all three levers at once; it dominates training step time alongside forward.
- Risk: Backward math is fiddly; validate dQ/dK/dV against the CPU reference (flash_attention_backward_cpu_dispatch) within BF16 tolerance. Atomics removal changes the parallelization, so re-check causal boundary handling (:250).

### [P1] Fuse GQA KV-broadcast into the attention kernel instead of materializing expanded KV  (M effort, memory-bandwidth, lever=fusion)
- Kernel: repeat_kv_kernel / repeat_kv_kernel_vec4 (src/ops/hip/attention_ops.hip) + callsites (src/nn/gqa_attention.cpp:91-92,140-141)
- Current: gqa_attention.cpp:91-92 and :140-141 call ops::repeat_kv to physically expand K,V from [B,KV_H,T,D] to [B,Q_H,T,D] (attention_ops.hip:19-67), duplicating KV n_rep times in DRAM and bandwidth before SDPA. Plus three .contiguous() transposes at gqa_attention.cpp:80-82.
- Change: Don't materialize. Keep K,V at num_kv_heads and have the flash-attention kernel index the KV head directly: map grid.y over (B x Q_H) and compute kv_head = q_head / n_rep when forming the K/V global offsets (offset_q at flash_attention.hip:47). This removes the repeat_kv launch, the [B,Q_H,T,D] allocation, and the n_rep-fold redundant KV reads (the same KV tile is reused across the n_rep query heads that share it — load it once into LDS). Backward sums dK/dV over the n_rep group instead of repeat_kv_backward.
- Expected: Decode/inference (memory-bound, T small): ~1.5-3x on the attention step for n_rep=4-8 (KV traffic is the bottleneck there). Prefill/training (compute-bound): ~1.05-1.2x but removes a full kernel launch + a [B,Q_H,T,D] tensor allocation per layer per step.
- Risk: Indexing/backward-reduction bugs; the repeat_kv path can stay as a fallback. Watch KV-cache layout in the decode path (GQAKVCache stores num_kv_heads already, gqa_attention.cpp:170).

### [P1] Tune wave32 occupancy: larger cooperative block + BF16 LDS budget  (S effort, occupancy, lever=wave32)
- Kernel: flash_attn_fwd_kernel (src/ops/hip/flash_attention.hip)
- Current: Block is dim3(BLOCK_M)=64 threads = exactly 2 wave32 (:321). Forward LDS = sK[64][64]+sV[64][64] = 32KB FP32 (:82-83); with the ~64KB/workgroup LDS budget that caps resident workgroups to ~2/CU, and the ~192 VGPR/lane caps waves further.
- Change: After the LDS/WMMA refactor, use a 128-256 thread block (4-8 wave32) so a query tile is processed cooperatively and the K/V load is amortized across more lanes. BF16 K/V tiles drop LDS 32KB->16KB, allowing ~4 workgroups/CU (more in-flight waves to hide the ~hundreds-of-cycle DRAM latency on the ~640 GB/s bus). Cap registers via __launch_bounds__ to lock in the target wave count.
- Expected: ~1.2-1.5x in isolation (better DRAM/exp latency hiding); largely realized as part of the P0 rewrite — listed separately because block size + LDS budget are explicit tuning knobs to set, not assume.
- Risk: Over-large blocks can re-raise LDS/VGPR pressure and reduce occupancy; profile 128 vs 256 with rocprof (occupancy + VALU/MFMA utilization) and pick empirically.

### [P2] Skip fully-masked KV blocks branchlessly; special-case only the diagonal block for causal  (S effort, compute, lever=coalescing)
- Kernel: flash_attn_fwd_kernel (src/ops/hip/flash_attention.hip)
- Current: kv_end already block-skips beyond the diagonal (:86-89), but inside every surviving block the inner loop evaluates per-element 'if (is_causal && global_kv > q_idx)' and 'else if (global_kv >= N)' for all 64 k (:121-136), so below-diagonal blocks pay branch + -FLT_MAX setup they don't need, and the diagonal handling causes warp divergence.
- Change: Partition the KV loop into (a) full below-diagonal blocks computed with NO mask branch, and (b) exactly one diagonal block where the triangular mask is applied (in the WMMA path, mask the S fragment from the fragment's lane->(row,col) mapping; in scalar, a single predicated block). Hoist the global_kv>=N tail check out of the steady state. Improves causal load balance: lower query tiles iterate fewer KV blocks.
- Expected: ~1.1-1.3x on causal SDPA (removes per-element branching/divergence in the common non-diagonal blocks; block-level triangular skip already exists so this is the residual). Folds naturally into the WMMA rewrite.
- Risk: Off-by-one at the diagonal/tail; cover with N not a multiple of BLOCK_N tests.

### [P2] Use exp2 with log2(e) folded into scale instead of expf in the hot loop  (S effort, compute, lever=vectorization)
- Kernel: flash_attn_fwd_kernel (src/ops/hip/flash_attention.hip)
- Current: flash_attention.hip:154 calls expf(scores[k]-m_new) for every (thread,k) = 64x64 transcendentals per tile; :141 also expf. RDNA's native transcendental is v_exp_f32 (2^x); expf forces an extra multiply by log2(e) per call.
- Change: Pre-fold log2(e)=1.442695 into the scale (scale *= 1.442695f at dispatch, flash_attention.hip:326) so scores are already in log2 units, then use exp2f / __builtin_amdgcn_exp2 (and exp2 for the m-rescale at :141). One native instruction per element, no per-element multiply. Standard FlashAttention trick; also the natural form for the WMMA path's P fragment.
- Expected: ~1.05-1.15x on the softmax-bound portion; more valuable post-WMMA when exp stops being hidden behind scalar matmul. Cheap and safe.
- Risk: Minimal; keep LSE math consistent (store m*ln2 + ln(l) so saved LSE stays in natural units for the backward at :170).

### [P2] Coalesce/vectorize Q,K,V loads and O stores (float4 / BF16x8) across the wave  (M effort, memory-bandwidth, lever=coalescing)
- Kernel: flash_attn_fwd_kernel global loads (src/ops/hip/flash_attention.hip) 
- Current: K/V are staged with thread tx loading a full row k_ptr[(j+tx)*D + d] (:101-102): for a fixed d, consecutive lanes are D=64 floats = 256 B apart -> uncoalesced (each lane hits a distinct 128 B cache line per memory instr). Same strided pattern for Q load (:77) and O store (:174).
- Change: Stage tiles so consecutive lanes read consecutive global addresses: flatten the [BN x D] tile and have lane i load element i, i+blockDim, ... using float4 (FP32) or a BF16x8 vector load, then transpose in LDS. Aligns to the 128 B cache line and issues wide transactions on the ~640 GB/s bus. Mostly subsumed by the P0 LDS-staging item but applies even to the current scalar kernel.
- Expected: ~1.1-1.3x on the memory-staging portion (forward is compute-bound for large N, so this is secondary there but real for the K/V load phase and for short sequences / decode).
- Risk: LDS transpose needs bank-conflict-free layout (pad the inner dim); verify with a memory-throughput probe before/after.

### [P2] Fix silently-ignored dropout and the O(N^2) dropout-mask allocation  (S effort, memory-bandwidth, lever=dtype)
- Kernel: flash_attention_hip_dispatch / forward kernel (src/ops/hip/flash_attention.hip + src/ops/flash_attention.cpp)
- Current: flash_attn_fwd_kernel (:33-39) has no dropout parameters and the dispatch at :326 never forwards dropout_p/training/dropout_mask_out — dropout is silently dropped on the GPU path. Meanwhile the host allocates a [B,H,N,N] FP32 mask when training with dropout (flash_attention.cpp:83), an O(N^2) buffer that defeats flash-attention's O(N) memory promise and is never written.
- Change: Either (a) implement in-kernel dropout with a counter-based RNG (philox) keyed by (b,h,i,j) so no mask is materialized — apply the keep-mask to P before P*V and store only the RNG seed for backward; or (b) if dropout is out of scope for now, remove the [B,H,N,N] allocation and make the API reject dropout_p>0 on the flash path instead of silently allocating and ignoring. Correctness + memory bug, not throughput, but it currently reintroduces the exact O(N^2) memory flash-attention exists to avoid.
- Expected: Throughput-neutral, but removes a B*H*N*N FP32 allocation during training-with-dropout (e.g. multi-GB at N=4096) and fixes a silent correctness gap.
- Risk: Counter-based RNG must match the backward's mask (forward/backward determinism); if deferring, ensure callers don't rely on dropout being applied.


## RDNA4 (gfx1201 / R9700, native wave32, ~640 GB/s) HIP kernels: normalization, reduction, sampling, fused_ops — with GEMM and flash-attention cross-references

**Bottleneck:** Two regimes. (A) Memory-bandwidth/latency-bound: norm, softmax, reductions, sampling. They leave bandwidth on the table via non-vectorized scalar global loads (rmsnorm/layernorm/softmax-scalar path/reduction kernels are NOT float4 — only the fused compute_inv_rms and softmax_vec4 are), redundant passes (layernorm reads each row 3x, softmax reads 2x), shared-memory tree reductions instead of wave32 shuffles/DPP, single-address atomicMax/atomicAdd block reductions in sampling, and per-token hipMalloc+hipMemcpy+hipDeviceSynchronize in the decode loop (check_stop, all_true). Estimate ~40-55% of the ~640 GB/s peak today; float4 + fewer passes + wave32 shuffle/DPP reductions + removing host syncs should reach ~75-85%. (B) Compute-bound: GEMM (gemm.hip fp16 + fused gemm_with_norm) and flash-attention run on scalar FP32 / converted-FP16 ALU FMAs — grep for wmma/mfma/__builtin_amdgcn across src/ returns ZERO hits, so the RDNA4 WMMA 16x16x16 matrix cores are 100% idle, roughly 3-5% of the ~190 TFLOPS FP16 matrix peak. PLUS a correctness landmine that contradicts the prior audit: normalization.hip:13 hardcodes WARP_SIZE_HIP=64 on a native-wave32 build (verified: --offload-arch=gfx1201, no -mwavefrontsize64 anywhere), so its blockReduce partitions 256 threads into 4 logical-64 groups over 8 physical 32-lane waves and silently drops the partials of waves 1,3,5,7 — the softmax/norm sum denominator is ~halved. sampling.hip (offset=16) and fused_ops backward (warpSize builtin) are already wave32-correct, so normalization.hip is the lone inconsistent file. Highest-impact lever overall is WMMA on the GEMM/attention path; cheapest do-first is the wave32 reduction fix.


### [P0] Use RDNA4 WMMA 16x16x16 matrix cores for FP16/BF16 GEMM (currently 100% scalar ALU)  (L effort, compute, lever=WMMA)
- Kernel: gemm_with_norm_kernel (fused_ops.hip) + gemm_fp16_kernel (gemm.hip) + flash_attn_fwd_kernel
- Current: fused_ops.hip:163-180 inner loop is scalar rC[i][j] += rA[i]*rB[j] on FP32 ALUs. gemm.hip gemm_fp16_kernel (~:738) loads __half, __half2float-converts, and does the same scalar FMA into a float rC. grep for wmma/mfma/__builtin_amdgcn across src/ = 0 hits — matrix cores are completely idle. Build is --offload-arch=gfx1201 (RDNA4), which has hardware WMMA.
- Change: Replace the BK-loop register-FMA microkernel with __builtin_amdgcn_wmma_f32_16x16x16_f16 (or rocWMMA) fragments: stage A/B tiles as FP16 in LDS, issue WMMA with FP32 accumulate, keep the float rC accumulator. Each 16x16 output sub-tile = one WMMA op. For the fused kernel, fold the RMSNorm scale (inv_rms*norm_weight) into the FP16 A-fragment before the WMMA. Build one microkernel and reuse it in gemm_fp16, gemm_with_norm, and flash-attention.
- Expected: 3-6x on FP16 GEMM. RDNA4 FP16 matrix peak ~190 TFLOPS vs ~20 TFLOPS scalar FP32 FMA peak (~10x raw); realistically 3-6x after LDS-staging and bandwidth limits. GEMM is 70-90% of transformer FLOPs, so this is the single dominant throughput lever on this card.
- Risk: Needs FP16/BF16 tiles (model in half precision, or cast-on-load to FP16 with FP32 accumulate). WMMA fragment/LDS-swizzle layout is fiddly; FP32 accumulate keeps accuracy acceptable. Validate against test_fp16_gemm.

### [P0] Fix wave64 assumption in normalization.hip block reductions (WARP_SIZE_HIP=64 on a wave32 card drops half the reduction inputs)  (S effort, latency, lever=wave32)
- Kernel: blockReduceMax_hip / blockReduceSum_hip + softmax_online/vec4 (normalization.hip)
- Current: normalization.hip:13 hardcodes WARP_SIZE_HIP=64, but the build is native wave32 (warpSize==32; verified no -mwavefrontsize64 in repo). warpReduce loops start at offset=WARP_SIZE_HIP/2=32 (:18,:26) — a wasted/UB shuffle to non-existent lanes 32-63. blockReduce uses lane=tid%64, wid=tid/64, numWarps=(blockDim+63)/64=4 (:34-35,:43,:59): a 256-thread block has 8 physical 32-lane waves, but only even-wave leaders (tid 0,64,128,192) write shared[], so partials of waves 1,3,5,7 (tid 32-63,96-127,160-191,224-255) are silently dropped — the softmax/norm SUM denominator is ~halved. sampling.hip (:44-58, offset=16) and fused_ops backward (:295, warpSize) are already wave32-correct. This contradicts the prior-audit claim that these kernels use WARP_SIZE=32.
- Change: Set WARP_SIZE_HIP=32 (or use the warpSize builtin). warpReduce then starts at offset=16 and covers the full 32-lane wave with no wasted/UB shuffle; blockReduce partitions into 8 correct waves with shared[8]. Optionally use __builtin_amdgcn_mov_dpp / ds_swizzle (DPP) for the intra-wave reduction to drop the LDS round-trip entirely. This also makes the M3 s_block_max broadcast workaround (:96-101,:163-167) unnecessary.
- Expected: Primary value is CORRECTNESS: on wave32 the current sum reduction returns ~half the true denominator, so softmax/layernorm/rmsnorm output is numerically wrong (masked for max by softmax shift-invariance, NOT for the sum). Perf: removes one wasted shuffle iter per reduction and is the prerequisite for the float4/single-pass norm/softmax rewrites below. ~1.05-1.1x standalone on the norm kernels.
- Risk: Low, pure fix. Verify against CPU softmax/layernorm reference and the new tests/test_training_stability.cpp (the M3 fix history suggests this area is already suspected).

### [P1] Flash-attention: move QK^T and P·V to WMMA, widen blocks, cut per-thread register footprint  (XL effort, register-spilling, lever=WMMA)
- Kernel: flash_attn_fwd_kernel (flash_attention.hip)
- Current: flash_attention.hip:33-177 is scalar SIMT: QK^T is dot += q_reg[d]*sK[k][d] over HEAD_DIM=64 (:130-132), P·V is scalar (:157-160). Launch is dim3 block(BLOCK_M)=64 threads (:323) = only 2 wave32 wavefronts/block -> low CU occupancy. Each thread holds scores[BLOCK_N]=scores[64] (:116) + acc_i[64] + q_reg[64] ~ 200+ VGPRs -> register spilling caps occupancy further. All FP32; matrix cores idle. Attention-softmax IS already fused (online softmax in-kernel), so the win is matrix cores + occupancy, not fusion.
- Change: Tile scores as 16x16 WMMA fragments (Q[BLOCK_M,64]·K^T) and P·V as WMMA, keeping the online-softmax rescale between them. Store scores in LDS instead of a per-thread scores[64] array to kill the spill, and raise the block to 128-256 threads so several Q-rows share each K/V LDS tile. Cast Q/K/V tiles to FP16 in LDS for WMMA with FP32 accumulate.
- Expected: 2-4x. WMMA gives the matmul speedup; removing scores[64]/acc_i[64] spill and going 64->128/256 threads/block lifts occupancy so the per-block-N exp latency is hidden. Attention is the 2nd-largest FLOP sink after the projections.
- Risk: Highest-complexity kernel; causal masking + online rescale interacting with WMMA fragment layout is tricky. Validate against test_attention_comprehensive and the LSE stored for backward.

### [P1] Norm kernels: single-pass mean/var, float4 vectorized loads, wave32 shuffle reduction  (M effort, memory-bandwidth, lever=vectorization)
- Kernel: layernorm_kernel + rmsnorm_kernel (normalization.hip)
- Current: layernorm_kernel (:279-330) reads the row 3x (mean :291, var :307, normalize :324) with scalar loads and a log-depth shared-memory tree reduction (:297-302,:314-319 = 8 __syncthreads). rmsnorm_kernel (:356-387) reads 2x with the same scalar-load tree reduction. Neither uses float4, unlike the fused compute_inv_rms_kernel (fused_ops.hip:60-66) which does. Pure bandwidth-bound.
- Change: LayerNorm: compute sum and sum-of-squares in ONE pass (var = E[x^2] - E[x]^2) -> 2 global reads instead of 3. Use float4 (128-bit) loads for 128-byte-coalesced access. Replace the shared-mem tree with the fixed wave32 blockReduceSum / __shfl_xor reduction. Same for rmsnorm: 1 read for sum-sq + 1 to normalize, float4 + shuffle.
- Expected: LayerNorm ~1.4-1.5x (3 reads->2 plus vectorization), RMSNorm ~1.2-1.3x. Moves these from ~40-50% toward ~75-85% of 640 GB/s.
- Risk: Single-pass var via E[x^2]-E[x]^2 is slightly less stable than two-pass; fine with FP32 accumulate on FP32 activations. float4 needs 16B-aligned rows + a scalar remainder tail.

### [P1] Softmax: stage the row in LDS to drop the 2nd global read (3 BW passes -> 2)  (M effort, memory-bandwidth, lever=LDS)
- Kernel: softmax_online_kernel / softmax_online_vec4_kernel (normalization.hip)
- Current: softmax_online_kernel (:84-126) runs an online max+sum pass reading the row once, then a second pass (:124-126) that re-reads row_input from global to write exp(x-max)*inv_sum. vec4 kernel (:186-194) does the same. For a row that fits in LDS the re-read is avoidable. (Both also currently mis-reduce — see the WARP_SIZE P0.)
- Change: When cols*4B <= ~32KB, load the row into LDS once (float4), run online max/sum from LDS, then write output from LDS -> 1 global read + 1 write instead of 2 reads + 1 write (33% less global traffic). Keep the two-pass path for rows too large for LDS. Route the duplicate sampling.hip softmax_2d_kernel (:754) to this kernel.
- Expected: ~1.3-1.5x for LDS-resident rows (attention seq-len, vocab < ~8k); reaches near-peak BW as global traffic drops from 3 to 2 row-passes.
- Risk: LDS capacity caps the fast-path row size; need a clean fallback. Pad LDS to avoid bank conflicts during the reduction.

### [P1] Replace single-address atomic block reductions with wave32 shuffle reductions  (S effort, latency, lever=wave32)
- Kernel: softmax_2d_kernel + top_p_filter_opt_kernel (sampling.hip)
- Current: softmax_2d_kernel reduces the block max/sum by having all 256 threads hammer one __shared__ scalar with atomicMaxFloat (CAS loop, :772) and atomicAdd (:786). top_p_filter_opt_kernel does the same (:420 atomicMaxFloat, :435 atomicAdd). Heavy atomic serialization on a single address.
- Change: Use warp_reduce_max/sum (already present and wave32-correct at sampling.hip:44-58) + an 8-entry shared array + a final wave reduce — a proper blockReduce. Eliminates CAS contention. Even better: call the LDS-staged online softmax kernel and delete the duplicate softmax_2d path.
- Expected: ~1.5-2x on these kernels at vocab 32k-50k (256-way atomic CAS contention -> ~log2(32) shuffle steps). Matters for the per-decode-step softmax/top-p.
- Risk: Low; the wave32-correct warp primitives already exist in this file.

### [P1] Kill per-token hipMalloc/hipMemcpy/hipDeviceSynchronize in the generation loop  (M effort, launch-overhead, lever=async-stream)
- Kernel: check_stop_tokens_kernel + all_true_kernel dispatch (sampling.hip)
- Current: check_stop_tokens_hip_dispatch (:842-869) does hipMalloc + hipMemcpy(H2D) + launch + hipDeviceSynchronize + hipFree on EVERY call. all_true_hip_dispatch (:907-933) does hipMalloc + 2 hipMemcpy + hipDeviceSynchronize + hipFree every call. In autoregressive decode these run once per generated token, so every token forces a full device sync + allocation churn, serializing the pipeline.
- Change: Preallocate the stop-id buffer once (upload when sampling config is set, not per token) and a persistent result/flag buffer. Keep results on-device; copy the should-stop flag back only when actually testing termination, via pinned memory + async copy on the sampling stream. Remove the per-call hipDeviceSynchronize so decode stays asynchronous.
- Expected: Removes ~2 device syncs + ~3 hipMallocs per token. For small batch / short kernels these syncs can be 30-50% of per-token wall time -> ~1.3-1.6x end-to-end decode throughput depending on model size.
- Risk: Must keep termination semantics correct (when the host actually reads the flag). Manage the stop-id buffer lifetime across sampling-config changes.

### [P1] Fuse residual-add into norm, and bias+activation, to remove whole BW round-trips  (M effort, memory-bandwidth, lever=fusion)
- Kernel: rmsnorm_kernel / layernorm_kernel (normalization.hip) + activation.hip + elementwise.hip
- Current: Transformer blocks do residual-add (elementwise.hip) then RMSNorm (normalization.hip:356) as separate kernels — the residual sum is written to global then immediately re-read by the norm. bias-add + activation (activation.hip) are likewise separate elementwise passes. Each separate elementwise kernel is a full read+write of the activation tensor plus a launch. The existing fused RMSNorm+Linear (fused_ops.hip) already proves the pattern pays off.
- Change: Add a fused add_rmsnorm kernel that takes x and residual, writes y=x+residual (needed as the next block's residual) AND its normalized output in one pass. Fuse bias+activation (and the SwiGLU gate multiply) into the producing GEMM epilogue or a single elementwise kernel.
- Expected: Each fused pair removes one full tensor read+write (~2*activation_bytes of global traffic) and one launch. ~1.2-1.4x on the norm/elementwise portion; compounds across every layer of every step.
- Risk: Must still expose the pre-norm residual output for the next block, and save the correct intermediates for autograd backward.

### [P2] Vectorize column reductions and use DPP/shuffle tail instead of volatile-shared  (S effort, memory-bandwidth, lever=vectorization)
- Kernel: sum_cols_kernel + max_cols_kernel (reduction.hip)
- Current: sum_cols_kernel (:106-142) and max_cols_kernel (:161-197) do scalar grid-stride loads (:112,:167) then a shared-mem tree with a hand-unrolled volatile 32-lane tail (:129-137,:184-192). The volatile tail is wave32-safe (single 32-lane wave in lockstep) but rides on UB lockstep assumptions and round-trips through LDS; loads are not vectorized.
- Change: float4 the grid-stride loads (128-byte coalesced), and replace the volatile-shared tail with __shfl_xor / __builtin_amdgcn DPP across the 32-lane wave (registers only, no volatile). Reuse the same wave32 blockReduce as the fixed norm path.
- Expected: ~1.2-1.4x, mostly from float4 closing the BW gap, plus removing the LDS round-trips in the reduction tail.
- Risk: Low; float4 alignment + scalar remainder handling.

### [P2] Two-stage reduction instead of iterative multi-launch with per-level allocation  (M effort, launch-overhead, lever=async-stream)
- Kernel: sum_hip_dispatch / reduce_kernel (reduction.hip)
- Current: sum_hip_dispatch (:55-70) loops launching reduce_kernel, each iteration allocating a new intermediate tensor via empty() and reducing by 512x, until n==1, then a D2D memcpy. Large n => several launches + several device allocations per call. reduce_kernel (:17-39) also loads only ~2 elements per thread before reducing.
- Change: Two-stage: a grid-stride kernel where each thread accumulates many elements and each block reduces to one partial (atomicAdd to a single output, or gridDim partials), then one small finalize. One allocation, 1-2 launches regardless of n.
- Expected: ~1.3-2x for large reductions; removes O(log_512 n) kernel launches and the per-level intermediate buffers.
- Risk: float atomicAdd is order-nondeterministic; for bit-reproducibility keep a fixed 2-pass partial layout instead of atomics.

### [P2] Parallelize the top-p threshold search and stop recomputing softmax 3x  (M effort, latency, lever=LDS)
- Kernel: top_p_filter_opt_kernel (sampling.hip)
- Current: top_p_filter_opt_kernel finds the nucleus threshold with a serial loop on thread 0 (:443-464) while 255 threads idle, and recomputes expf(x-max)/sum three separate times (sum :429, threshold loop :455, output :469). The non-opt top_p_filter_kernel (:347-378) is worse (O(V^2) serial threshold on thread 0).
- Change: Store probs in LDS once (from the online-softmax pass), then find the nucleus boundary block-parallel: a histogram/partial-sum over probability buckets, or a block scan after a partial sort. At minimum parallelize the threshold loop across the block and reuse stored probs instead of 3x expf.
- Expected: ~2-4x on top-p at vocab 32k-50k: single-thread O(V) (or O(V^2)) scan -> block-parallel pass, plus removing 2 redundant exp sweeps.
- Risk: Exact nucleus set must match the reference; bucket/approx boundaries can shift by one token.

### [P2] Block-parallel prefix sum for the sampling CDF instead of serial thread-0 scan  (M effort, latency, lever=LDS)
- Kernel: multinomial_cdf_kernel (sampling.hip)
- Current: multinomial_cdf_kernel builds the CDF with a serial cumulative-sum loop on thread 0 over the whole vocab (:603-609) while the rest of the block idles, then threads binary-search it.
- Change: Replace the thread-0 scan with a block-wide inclusive scan (Hillis-Steele/Blelloch in LDS, or per-wave __shfl scans + a wave-offset pass). Keep the binary-search sampling.
- Expected: ~2-3x on the CDF build for vocab ~8k (serial O(V) -> O(V/threads + log) parallel scan); only the CDF path benefits.
- Risk: Scan edge cases and LDS bank conflicts.

### [P2] Rewrite fused RMSNorm+Linear backward: O(dim_in*dim_out) per-thread atomics + incomplete formula  (L effort, compute, lever=WMMA)
- Kernel: fused_rmsnorm_linear_backward_kernel (fused_ops.hip)
- Current: Each block (one row) loops j over dim_out and, inside, k over dim_in doing atomicAdd into grad_linear_weight[j*dim_in+k] (:351-356) — O(dim_in*dim_out) global atomics per row. grad_input recomputes W^T@grad_out scalar (:362-385). It is explicitly incomplete: :387 // TODO: missing the RMSNorm normalization correction term, so training gradients are wrong.
- Change: Compute grad_linear_weight as a real GEMM (grad_out^T @ normalized_input) via the WMMA microkernel instead of element-wise atomics; compute grad_input as grad_out @ W then apply the RMSNorm-backward Jacobian — the correct correction term already exists in rmsnorm_backward_kernel (normalization.hip:538-605); reuse it and finish the TODO.
- Expected: Order-of-magnitude on the weight-grad (per-element atomics -> GEMM/WMMA) and a CORRECTNESS fix for training (current grads omit the RMSNorm correction). Training-only; no inference impact.
- Risk: Correctness-critical; validate with the existing gradient-check tests (test_linear_grad_check) and against the separate rmsnorm_backward.


## RDNA4 (gfx1201 / R9700) elementwise, activation, RoPE, embedding, cast, copy & comparison HIP kernels — /home/daniel/Projects/vesper/src/ops/hip/

**Bottleneck:** These 7 files are memory-bandwidth-bound elementwise/gather/cast/copy kernels on a ~640 GB/s, ~64-CU, native-wave32 gfx1201. WMMA/matrix cores are NOT a lever here: none of these files contain matmuls (the WMMA opportunity lives in gemm.hip:728/821/920 and flash_attention.hip, which currently convert FP16->FP32 with __half2float and do scalar FMA — that is a separate, high-value roadmap). Build flags are bare: `--offload-arch=gfx1201 -O3 -DNDEBUG` with NO -ffast-math / -ffp-contract / -fno-math-errno (build-hip/.../flags.make:9). Current state within scope: residual adds (same-shape Add), SwiGLU silu*mul (elementwise.hip:608-672), Adam/lerp/addcmul (elementwise.hip:794-958) and RoPE (rope.hip:87-174) already use float4 and are roughly 70-85% of peak BW. The real gaps: (a) GELU/sigmoid evaluate transcendentals in FP64 (activation.hip:11 exp(), elementwise.hip:79/85/447/462/478 tanh()) — RDNA4 FP64 runs at ~1/16 of FP32, so those kernels are ALU-bound at well under 10% of FP32 capability instead of BW-bound; (b) every broadcast/strided/scalar fallback does per-element 64-bit modulo+divide coordinate decomposition with scalar 4-byte loads (elementwise.hip:154-160, 316-321, 195-200; copy.hip:29-46) — bias-add ([M,N]+[N]) and any non-%4 tail land here at ~25-40% of peak BW; (c) sigmoid/relu/comparison/cast/copy are pure scalar with no float4 (activation.hip, comparison.hip, cast.hip, copy.hip:48); (d) FP16/BF16 elementwise is silently reinterpreted as float — launch_* hardcode data_ptr<float>() and instantiate <float,Op> (elementwise.hip:264-299, 358-389, 408-437) — both a correctness gap and zero packed-half throughput; (e) activation/comparison/copy/reduction launch on default stream 0 (0 Stream::current refs) creating a cross-stream hazard and blocking overlap; (f) RoPE launches dim3 threads(head_dim/4) = 16-32 threads/block = 0.5-1 wave32, occupancy-starved on 64 CUs. Net: the already-vectorized FP32 paths are healthy; the wins are concentrated in the scalar/broadcast/FP64 fallbacks and RoPE launch geometry.


### [P0] Add a contiguous/last-dim-broadcast fast path and vectorize the broadcast adds; stop doing per-element coordinate decomposition  (M effort, memory-bandwidth, lever=vectorization + coalescing (float4/dwordx4) + fast contiguous path)
- Kernel: elementwise.hip — broadcast/scalar/unary fallback kernels (elementwise_broadcast_kernel, elementwise_scalar_kernel, elementwise_unary_kernel) and their vectorization gate
- Current: The vectorized path only fires when BOTH inputs are the exact same shape as out and contiguous (elementwise.hip:259-267). Any broadcast (e.g. Linear/conv bias-add [M,N]+[N], LayerNorm-style affine) falls into elementwise_broadcast_kernel which, per output element, runs a runtime-bound loop doing 64-bit `remaining % shape[i]` / `remaining /= shape[i]` over up to 8 dims plus scalar 4-byte loads (elementwise.hip:154-160). Same scalar+divide pattern in elementwise_scalar_kernel (316-321) and elementwise_unary_kernel (195-200). The functor already defines an op(float4,float) scalar-broadcast overload (e.g. Add at :28-30) that is never wired into any broadcast kernel.
- Change: 1) Add a 'fully contiguous, same numel' fast path for binary/unary/scalar that uses the existing float4 kernels even when shapes differ but layouts are contiguous and numel matches (covers reshaped/viewed-but-contiguous tensors that currently miss the gate). 2) Add a vectorized last-dim-broadcast kernel: when b broadcasts only along leading dims and the contiguous inner dim matches (the bias-add case), load a as float4 and broadcast b per-row using the existing op(float4,float) overload — float4 loads/stores, no coordinate decomposition. 3) For the genuinely strided residual path, precompute collapsed/merged contiguous dim-groups on the host so the in-kernel loop runs over far fewer dims, and hoist the divisions out where strides are unit.
- Expected: 3-6x on broadcast bias-adds and any contiguous-but-unmatched case (eliminates ~10-20 integer-divide ALU ops/element and moves 4B->16B/thread, lifting ~30% -> ~85% of the 640 GB/s peak). For Llama-style configs with bias=false the win is smaller; for bias=true Linear/Conv and affine norms it is large.
- Risk: Broadcast/stride edge cases and 16B alignment must be guarded (keep the scalar kernel as correctness fallback). Medium.

### [P0] Kill FP64 transcendentals — use f32 fast intrinsics (__expf / tanhf) on the f64-starved RDNA4  (S effort, compute, lever=dtype (FP32 vs FP64) + fast f32 intrinsics mapping to v_exp_f32 / v_rcp_f32)
- Kernel: activation.hip sigmoid_kernel; elementwise.hip Gelu / GeluBackward (and SiLU family)
- Current: activation.hip:11 computes `static_cast<T>(1)/(static_cast<T>(1)+exp(-in[idx]))` — `exp` is double, so float promotes to FP64, runs an FP64 exp, then truncates. elementwise.hip:79 and :85 (Gelu) and :447/:462/:478 (GeluBackward) call `tanh(...)` (double) on float args. With no -ffast-math (flags.make:9) these are full FP64 transcendentals; RDNA4 FP64 throughput is ~1/16 of FP32, so these kernels are ALU-bound, not BW-bound. SiLU/silu_mul (elementwise.hip:122,127,577,613) correctly use expf (f32) but still the accurate library expf, not the hardware __expf.
- Change: Replace `exp(` -> `__expf(` in sigmoid (activation.hip:11) and `tanh(` -> `tanhf(` (or a 2-op rational/`__expf`-based tanh) in Gelu/GeluBackward (elementwise.hip:79,85,447,462,478). Switch SiLU/silu_mul/silu_inplace expf -> __expf (elementwise.hip:122,127,577,584,613,624,701,712) and sigmoid in SiLUBackward. Prefer explicit __ intrinsics over a global -ffast-math so host code numerics are untouched.
- Expected: 4-8x on GELU/sigmoid kernels (converts them from FP64-ALU-bound back to BW-bound). ~1.1-1.3x on the already-BW-bound SiLU/silu_mul path. Removes a latent FP64 cliff that hits hard whenever a GELU-MLP or sigmoid-gated config is selected.
- Risk: Fast intrinsics carry ~1-2 ULP error; negligible for training/inference. Low.

### [P1] Launch on the current Vesper stream, not default stream 0 (fix cross-stream hazard + enable overlap)  (S effort, latency, lever=async-stream correctness / overlap)
- Kernel: activation.hip (sigmoid/relu), comparison.hip (all 3), copy.hip (copy_strided), reduction.hip (sum_rows/sum_cols/max_cols)
- Current: These dispatchers never call Stream::current (grep: 0 references) and pass literal `0` as the stream arg: activation.hip:18,33; comparison.hip:33,40,47; copy.hip:76; reduction.hip:97,154,209. Everything else in the codebase (elementwise/cast/embedding/rope) correctly fetches Stream::current(Device::HIP). Mixing stream-0 kernels with the model's working stream forces implicit serialization and risks read-before-write hazards across streams.
- Change: In each of these dispatchers fetch `hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());` and pass it as the 4th hipLaunchKernelGGL arg instead of 0 (include vesper/core/stream.h where missing).
- Expected: Correctness fix (eliminates a real cross-stream race) plus removes false serialization; enables kernel/copy overlap on the training stream. Throughput effect is workload-dependent but unblocks all later pipelining work.
- Risk: Verify nothing implicitly relied on stream-0 global sync. Low.

### [P1] Fix RoPE block size — 16-32 threads/block is 0.5-1 wave32; pack sequence positions to fill ≥128-256 threads  (M effort, occupancy, lever=grid/block sizing tuned to native wave32 (multiple waves per block))
- Kernel: rope.hip rope_vectorized_kernel / rope_inverse_vectorized_kernel launch config
- Current: apply_rope_hip launches `dim3 threads(num_vec)` with num_vec = head_dim/4 (rope.hip:188,195,221,228). For head_dim=64 that is 16 threads (half a wave32, so half the lanes are masked off on the very first instruction); head_dim=128 gives 32 threads (exactly one wave). Grid is (batch*heads, seq_len). With one short wave per block, occupancy and latency-hiding on 64 CUs are starved even though the kernel itself is correctly float4-vectorized and uses __ldg.
- Change: Make the block 2D: threadIdx.x over num_vec, threadIdx.y over a tile of sequence positions (e.g. 8-16 positions) so blockDim = num_vec * positions_per_block reaches 128-256 threads; collapse blockIdx accordingly. This packs multiple positions per workgroup, fills several waves per CU, and keeps loads coalesced across the head_dim.
- Expected: 1.5-3x on RoPE (runs every layer every step). Largest at head_dim=64 where current blocks waste half their lanes.
- Risk: Index/bounds math must stay exact for ragged seq_len tiles. Low-medium.

### [P1] Vectorize the remaining pure-bandwidth scalar kernels to float4 / packed-half + contiguous fast path  (M effort, memory-bandwidth, lever=vectorization (float4 / dwordx4) + packed half2 conversion + coalescing)
- Kernel: activation.hip (sigmoid/relu), comparison.hip (greater_than/equal/equal_scalar), cast.hip (all conversion kernels), copy.hip contiguous path
- Current: These are all scalar one-element-per-thread (4 B/thread): sigmoid/relu (activation.hip:8-35), comparison kernels (comparison.hip:6-28), every cast kernel incl. the hot FP32<->FP16 path (cast.hip:64-93), and copy.hip's contiguous branch still does scalar dst[idx]=src[idx] (copy.hip:48). On a BW-bound card these run at roughly 25-40% of peak because each thread issues a single narrow load/store.
- Change: Add float4 kernels for sigmoid/relu (relu via fmaxf on float4) and for comparison (compare 4 lanes -> 1.0/0.0). For cast, process 8 halves/thread: load float4, convert with __float22half2_rn to two half2 and store 8 B (or float4 on widen), and use __half22float2 on the reverse — coalesced 16 B transactions. Gate all on 16 B alignment + numel%vec==0 with the existing scalar kernels as fallback.
- Expected: 2-4x toward peak BW on each (most impactful on FP32<->FP16 cast in mixed-precision and on .contiguous() copies).
- Risk: Alignment and tail handling; keep scalar fallback. Low-medium.

### [P2] Add a native FP16/BF16 packed-half elementwise path (currently half tensors are silently reinterpreted as float)  (L effort, memory-bandwidth, lever=dtype + half2 packed math (v_pk_add/mul/fma_f16) + vectorization)
- Kernel: elementwise.hip launch_broadcast_kernel / launch_scalar_kernel / launch_unary_kernel (dtype handling)
- Current: All three launchers hardcode data_ptr<float>() and instantiate the kernels as <float, Op> (elementwise.hip:264-299, 358-389, 408-437), and the host add()/mul()/... allocate output with a.dtype() then call these unconditionally (src/ops/elementwise.cpp:97-122). A Float16/BFloat16 tensor therefore has its bytes read/written as float — wrong values and out-of-bounds. There is no packed-half compute path at all, so even once corrected, FP16 elementwise would leave 2x bandwidth on the table.
- Change: Templatize the launchers on dtype and add half/bf16 kernels that process 8 elements/thread as 4x __half2 using __hadd2/__hmul2/__hfma2 (and __half22float2 for transcendental functors that need f32 intermediates). Route DType::Float16/BFloat16 from the host dispatch to these kernels instead of the float path. This both fixes the correctness gap and gives ~2x BW (half the bytes) for mixed-precision training once enabled (train_tinystories.cpp:96 notes FP16 path is pending).
- Expected: Correctness fix today; ~2x bandwidth on FP16/BF16 elementwise once mixed precision is turned on (half the bytes moved + packed ALU).
- Risk: dtype plumbing through host dispatch + numeric validation needed; net-new test coverage. Medium.

### [P2] Restructure embedding gather to block-per-row coalesced + vectorized copy; special-case L2 max_norm  (M effort, memory-bandwidth, lever=coalescing + vectorization (float4) + removing 64-bit integer divide)
- Kernel: embedding.hip embedding_forward_kernel and embedding_max_norm_kernel
- Current: embedding_forward_kernel uses one thread per output element and recomputes `row = i/embedding_dim` and `col = i%embedding_dim` with 64-bit integer divide/modulo per element (embedding.hip:82-83), then a scalar gather. embedding_max_norm_kernel computes `powf(fabsf(val), norm_type)` per element and `powf(sum, 1/norm_type)` (embedding.hip:36,53) — a transcendental pow even for the common L2 case where it should be x*x and sqrtf.
- Change: Launch one block (or wave) per index row; threads stride the embedding_dim contiguously and copy the source row as float4 — coalesced, vectorized, and the row index comes from blockIdx so the per-element 64-bit divide disappears. In max_norm, branch norm_type==2.0f to use val*val + sqrtf instead of powf.
- Expected: 2-3x on embedding forward (gather becomes coalesced float4 instead of scalar with per-element divide); max_norm path drops a transcendental when it runs.
- Risk: padding_idx / out-of-range index handling must be preserved per row. Low-medium.

### [P2] Route pure contiguous moves through hipMemcpyAsync / float4 instead of a scalar copy kernel  (S effort, memory-bandwidth, lever=vectorization + DMA copy engine (hipMemcpyAsync) / launch-overhead avoidance)
- Kernel: copy.hip copy_strided_hip_dispatch (contiguous case) and cast.hip Float32->Float32 / same-dtype paths
- Current: When src and dst are both contiguous, copy_strided_kernel still runs a scalar element-wise kernel (copy.hip:48), and cast.hip dispatches cast_float_to_float_kernel (cast.hip:42, :165-168) for a same-type 'cast'. Both move data at scalar (4 B/thread) rate when a DMA copy engine or float4 kernel would saturate BW; copy.hip also launches on stream 0.
- Change: Detect contiguous src==dst layout and same-dtype cast and issue hipMemcpyAsync(DeviceToDevice) on the current stream (cast.hip already does this for Float16->Float16 at :203 and BF16->BF16 at :232 — extend to Float32->Float32). For strided-but-inner-contiguous copies, use a float4 kernel on the contiguous inner dimension.
- Expected: Up to ~3-4x on large contiguous copies/identity-casts (saturates BW / uses the copy engine vs a scalar kernel) and removes a redundant kernel launch.
- Risk: Confirm aliasing rules (no in-place overlap) before memcpy. Low.


## RDNA4 (gfx1201 / AMD Radeon R9700, 64 CU, ~32GB GDDR6 ~640 GB/s, ROCm 7.2.4 hipcc) GPU kernel + dispatch optimization roadmap for Vesper

**Bottleneck:** Dominant cost is GEMM + attention, and both run as pure scalar SIMT on the vector ALUs: the FP16 GEMM (gemm.hip:721-735) converts each half to float with __half2float and does scalar FMA, the FP32 path (gemm.hip:1316-1342) is register-tiled scalar, and flash-attention QK^T/P*V (flash_attention.hip:253-258) are scalar dot products. The ~96 TFLOPS FP16/BF16 WMMA matrix cores are 100% idle. FP32 GEMM realistically hits ~30-40% of the ~48 TFLOPS FP32 vector peak (~15-19 TFLOPS), so matmuls run at roughly ~15-20% of the card's achievable FP16 matrix throughput — that compute gap is the headline. Secondary limiter is latency/launch-overhead on the decode + auxiliary path: cat/index_ops/sampling each do per-call hipMalloc+hipMemcpy+hipDeviceSynchronize+hipFree (cat.hip:182-278, index_ops.hip:289-333, sampling.hip:850-930), the decode loop does a per-token [B,vocab] D2H + CPU sample + H2D round trip (transformer.cpp:256-305), and Stream::set_current is never called so the entire library executes on the null default stream 0 (stream.cpp:107) — zero async overlap, with explicit device syncs flushing the pipeline. Build is -O3 with --offload-arch=gfx1201 but no fast-math/FTZ. Memory BW (~640 GB/s) is not the primary limiter for the compute kernels. Net: heavily compute-bound on idle matrix cores, plus avoidable host-sync/launch latency in decode.


### [P0] Replace scalar half-to-float FMA GEMM with RDNA4 WMMA 16x16x16 matrix cores  (L effort, compute, lever=WMMA)
- Kernel: gemm_fp16_kernel / gemm_fp16_fp32out_kernel / gemm_fp16_batch_kernel (FP16/BF16 matmul; also the FP32 register-tiled path)
- Current: gemm.hip:721-735 (repeated at 813-827, 912-926): inner K-loop does rA[i]=__half2float(sA[...]) then scalar rC[i][j]+=rA[i]*rB on the vector ALU; tiles BM_FP16=64,BN_FP16=64,BK_FP16=16,TM/TN=4 (gemm.hip:43-47). Reached via dtype check at gemm.hip:1205. The FP32 path (gemm.hip:1316-1342, 1272-1283) is likewise pure SIMT. Matrix cores are never touched anywhere in the file (no wmma/mfma intrinsics present).
- Change: Rewrite the K-loop to use __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 (or rocWMMA): stage 16x16 __half fragments of A/B in LDS, accumulate into a float32 WMMA fragment, store back. BK_FP16=16 already matches the WMMA K=16 depth, so the tiling barely changes. For the FP32/training path, cast inputs to BF16 on the fly and use the bf16 WMMA variant with FP32 accumulate (numerically standard for training). Keep the existing scalar kernel as a tail/remainder fallback for dims not multiple of 16. Precondition: route hot matmuls through DType::Float16/BFloat16 (the AMP scaffolding in nn/amp.cpp:13-46 and the cast kernels already exist).
- Expected: 3-5x on the GEMM kernel itself (scalar+convert ~15-20 TFLOPS effective vs ~50-67 TFLOPS achievable WMMA against a ~96 TFLOPS FP16 dense peak), and ~2-4x on end-to-end transformer fwd/bwd because GEMM+attention are ~80-90% of model FLOPs.
- Risk: WMMA fragment layout + LDS swizzle is fiddly; BF16-with-FP32-accumulate must be validated against the existing gradient checks for training stability; needs multiple-of-16 tiling with a scalar remainder path. rocWMMA dependency or hand-written intrinsics both viable on ROCm 7.2.4.

### [P0] Pass cat metadata as kernel args and drop the per-call malloc/memcpy/device-sync  (S effort, launch-overhead, lever=pass-by-value kernel args / async-stream)
- Kernel: cat_hip_dispatch (KV-cache concat, runs every decode step)
- Current: cat.hip:182-188 does hipMalloc x2 (input_ptrs_device, dim_sizes_device) + hipMemcpy x2 H2D, launches on hardcoded stream 0 (cat.hip:211/228/248), then hipDeviceSynchronize() at cat.hip:275 and hipFree x2 at cat.hip:277-278 — every single cat call.
- Change: For the common small case (a handful of input tensors), pass the pointer list and dim_sizes by value inside a small POD struct kernel argument (HIP allows ~4KB of kernel args) or fixed-size arrays — eliminating both hipMalloc and both hipMemcpy. Remove the hipDeviceSynchronize entirely (the downstream consumer kernel already orders on the same stream). For large input counts, take a reusable scratch buffer from the existing CachingAllocator instead of malloc/free per call. Launch on Stream::current() rather than 0.
- Expected: Eliminates ~2 hipMalloc + 2 hipMemcpy + 1 full-device sync (tens of microseconds of driver/sync latency) per cat; in a token-by-token decode loop where cat appends KV each step this is a direct per-token latency cut and removes a pipeline flush.
- Risk: Kernel-arg struct size cap for very large tensor lists — keep the scratch-buffer path as a fallback above a threshold. Removing the sync is only safe once consumers read on the same stream (true today since all is stream 0).

### [P0] Sample on device and copy back only the chosen token, not the full [B,vocab] probabilities  (M effort, latency, lever=fusion / async-stream)
- Kernel: Transformer::generate / Generator decode loop (per-token sampling)
- Current: transformer.cpp:256-264 computes softmax then probs.to(Device::CPU) — a full [B,vocab] D2H plus implicit device sync — every token, samples on CPU with std::discrete_distribution (transformer.cpp:266-295), then H2D the token back (transformer.cpp:299). generator.cpp also does stop_mask.to(CPU)/done_mask.to(CPU) (generator.cpp:100-103) and next_token.to(CPU) (generator.cpp:193-194) each step.
- Change: Use the already-implemented device sampling kernels (sampling.hip: argmax/top_k/top_p/multinomial) to produce the token id on-GPU and copy back only B int32 ids. Wire a device-side RNG (or pass a per-step seed) to preserve reproducibility. Keep stop-token / done-mask checks on device and reduce to a single small flag instead of full-mask D2H every step.
- Expected: Replaces a vocab-sized D2H (e.g. B=1, vocab=32k -> 128KB) + full sync + host work + H2D with a 4-byte readback per token. Decode is latency-bound, so 1.5-3x tokens/s at small batch.
- Risk: Must reproduce existing sampling semantics (temperature/top-k/top-p) and seed-based reproducibility exactly; device multinomial exists but RNG wiring and numerical parity need testing.

### [P1] Activate the currently-dead stream plumbing and remove blocking device syncs from the hot path  (M effort, latency, lever=async-stream)
- Kernel: Stream layer + all *_dispatch launches
- Current: Stream::set_current (stream.h:39) is never called anywhere in src/examples/web, so Stream::current(Device::HIP) always returns the null default stream (stream.cpp:107, raw_handle()==nullptr==stream 0). Every kernel runs on stream 0; cat/sampling/comparison/activation/reduction even hardcode 0 in the launch (cat.hip:211, sampling.hip:129/482/643, reduction.hip:97). Blocking hipDeviceSynchronize in cat.hip:275, index_ops.hip:328/385/439, sampling.hip:864/926 flush the whole device.
- Change: Add a StreamGuard that sets a real per-model hipStream as current; replace the hardcoded 0 in cat/sampling/reduction/comparison/activation launches with Stream::current().raw_handle(). Remove hipDeviceSynchronize except where a host actually reads a result, and make CachingAllocator::free stream-ordered (record an event before returning a block to the free list) so async reuse is safe once multiple streams exist.
- Expected: Enables overlap of H2D/compute/D2H and removes full-device flushes; ~5-15% on its own but it is the prerequisite that unlocks device-sampling overlap, pinned-memory async copies, and graph capture.
- Risk: Correctness: removing syncs requires the allocator to be stream-aware, otherwise a buffer can be recycled while still in flight. Sequence this with the allocator change; validate with race-prone tests.

### [P1] Pass shape/stride arrays by value as kernel args; drop per-call malloc/memcpy/sync  (S effort, launch-overhead, lever=pass-by-value kernel args)
- Kernel: gather_hip_dispatch / scatter_hip_dispatch / scatter_scalar_hip_dispatch (index ops)
- Current: index_ops.hip:289-297 does hipMalloc x4 + hipMemcpy x4 for in_shape/in_strides/idx_shape/out_strides, then hipDeviceSynchronize() at index_ops.hip:328 and hipFree x4 at 330-333; identical pattern in scatter (346-390) and scatter_scalar (402-443).
- Change: ndim is tiny (<=8 for any real tensor): pass the four shape/stride arrays as fixed-size POD arrays in the kernel signature instead of allocating/copying device buffers. Remove the hipDeviceSynchronize (the kernel orders on the stream). Launch on Stream::current() instead of 0.
- Expected: Removes 4 hipMalloc + 4 hipMemcpy + a full device sync + 4 hipFree per gather/scatter; gather feeds the sampling/top-k and embedding-grad paths, so this cuts tens of microseconds of pure driver overhead per call.
- Risk: Must enforce an ndim cap (e.g. assert ndim<=8) matching the fixed array size; trivial otherwise.

### [P1] Add RDNA4 fast-math/FTZ flags and bake --offload-arch=gfx1201 into CMake  (S effort, compute, lever=dtype (fast-math / FTZ))
- Kernel: All HIP kernels (build system)
- Current: build-hip/CMakeCache shows CMAKE_CXX_FLAGS=--offload-arch=gfx1201 passed manually at configure time (it is absent from src/CMakeLists.txt), plus -O3 -DNDEBUG. No -ffast-math, -fno-math-errno, or -fgpu-flush-denormals-to-zero. .hip files are compiled LANGUAGE CXX via hipcc (src/CMakeLists.txt:139-160).
- Change: In src/CMakeLists.txt attach per-source COMPILE_OPTIONS to the .hip TUs: -O3 -ffast-math -fno-math-errno -fgpu-flush-denormals-to-zero -munsafe-fp-atomics, and set --offload-arch=gfx1201 (and GPU_TARGETS=gfx1201) in-tree so a fresh clone builds for the right arch without manual flags. Wave32 is already the gfx1201 default, so no -mwavefrontsize flag is needed.
- Expected: 5-20% on transcendental-heavy kernels (softmax expf, silu/gelu sigmoid, rope sin/cos, rmsnorm/layernorm rsqrt) from faster intrinsics; FTZ removes denormal stalls; -munsafe-fp-atomics enables hardware FP atomics in scatter/embedding-backward.
- Risk: fast-math relaxes IEEE NaN/Inf handling — verify softmax/layernorm numerics and training loss stability; scope the flags to device TUs, not host code.

### [P1] Use WMMA matrix cores for QK^T and P*V in flash attention  (XL effort, compute, lever=WMMA)
- Kernel: flash_attn_fwd_kernel / flash_attn_bwd_kernel (attention)
- Current: flash_attention.hip:253-258 computes scores with a scalar loop score += sQ[m][d]*k_reg[d] over HEAD_DIM, and the P*V accumulation (around 261-288) is likewise scalar on the vector ALU; no matrix instructions. Same scalar pattern in the fwd path near 116-164.
- Change: Tile Q/K/V to 16x16 and use wmma_f32_16x16x16_f16 for S=QK^T and O=P*V, keeping the online-softmax rescale (running max/sum) between the two matmuls in FP32 accumulators. Follow the rocWMMA flash-attention reference for the fragment relayout and causal masking.
- Expected: 2-4x on attention for prefill / longer sequences; smaller at seqlen=1 decode where attention is KV-memory-bound rather than compute-bound.
- Risk: Hardest kernel to get right — online softmax sandwiched between two WMMA matmuls with fragment relayout, plus correct causal masking and backward pass. Requires careful numerical validation.

### [P1] Use pinned (hipHostMalloc) host staging buffers and async copies for H2D/D2H  (M effort, memory-bandwidth, lever=pinned DMA / async-stream)
- Kernel: Dataloader collate + decode token transfers + CachingAllocator(CPU)
- Current: allocator.cpp:142 allocates CPU memory with new char[size] (pageable). Dataloader collates batches with ops::stack (dataloader.cpp:51-52) then they reach the GPU via pageable copies; the decode loop copies tokens/probs host<->device every step. No hipHostMalloc anywhere in the tree.
- Change: Add a pinned-host allocator (hipHostMalloc) for training-batch staging and decode token buffers, and issue transfers with hipMemcpyAsync on a dedicated copy stream so H2D/D2H overlap compute. Cap the pinned pool to avoid exhausting pinned memory.
- Expected: Pinned transfers run ~1.5-3x the pageable bandwidth and are DMA-overlappable, removing input-pipeline stalls and shrinking the decode token round-trip.
- Risk: Pinned memory is a limited OS resource — bound the pool; async copies are only safe after the stream work (item: activate async streams) lands.

### [P1] Fix hardcoded wave64 constants to wave32 for correct/fast warp reductions on gfx1201  (S effort, compute, lever=wave32)
- Kernel: softmax/rmsnorm/layernorm warp reductions (normalization.hip) + flash-attention
- Current: normalization.hip:13 defines WARP_SIZE_HIP=64 and uses it in __shfl_xor reductions (normalization.hip:18/26 loop offset starts at 32) and in lane/warp indexing + shared-mem sizing (normalization.hip:34-43, 263: shared_mem_size=(threads/64+1)). flash_attention.hip:17 hardcodes WARP_SIZE=64. gfx1201 wavefronts are 32 lanes, so the first shuffle iteration (offset=32, width defaults to 32) resolves to the lane's own value (laneId XOR 32 masked by width-1 == laneId), and 64-thread 'warps' span two physical wavefronts that __shfl cannot cross.
- Change: Set the wavefront constant to 32 (or use the runtime warpSize / __AMDGCN_WAVEFRONT_SIZE as fused_ops.hip:295 already does) in normalization.hip and flash_attention.hip, and size per-warp shared arrays as threads/32. This removes the wasted/own-value shuffle step and makes block reductions count all wavefronts.
- Expected: Small direct throughput gain (drops a redundant shuffle round and right-sizes LDS) but, more importantly, eliminates a likely latent correctness bug: on wave32 the FP32 warp-sum reduction over-counts/mis-counts, which corrupts softmax/normalization sums used in every transformer layer. Verify against reference outputs.
- Risk: If a path currently 'works' only because of a compensating block configuration, changing the constant will shift behavior — validate softmax/rmsnorm/layernorm numerics before and after. Cannot confirm at runtime here (GPU reserved), so treat as verify-then-fix.

### [P2] Capture the steady-state decode step into a HIP graph and replay per token  (L effort, launch-overhead, lever=async-stream (graph capture))
- Kernel: Per-token decode step (forward_with_cache + sampling kernel sequence)
- Current: transformer.cpp:243-305 relaunches the full per-token kernel sequence from the host every token (dozens of launches: rmsnorm, QKV gemm, rope, attention, MLP, sampling). No hipGraph / stream capture anywhere in the tree.
- Change: After host syncs are removed from the step (depends on the device-sampling and async-stream items), hipStreamBeginCapture the steady-state decode step once with fixed KV-cache shapes, instantiate a hipGraphExec, and hipGraphLaunch per token, updating only the token id and position inputs. Re-capture on shape change (e.g. prefill vs decode).
- Expected: Collapses dozens of per-token kernel launches (each ~a few microseconds of CPU launch latency) into one graph launch; at small models / batch=1 where launch overhead dominates, 1.2-2x tokens/s.
- Risk: Requires fully static shapes per captured step and zero host syncs inside the captured region; KV-cache buffer pointers must remain stable across tokens. Brittle if the allocator hands out different pointers per step.

### [P2] Add post-launch hipGetLastError checks (debug builds only)  (S effort, launch-overhead, lever=async-stream (debuggability))
- Kernel: All *_dispatch kernel launches (112 hipLaunchKernelGGL sites)
- Current: Only 3 hipGetLastError calls exist across 112 hipLaunchKernelGGL launches (copy.hip:89, rope.hip:203/233). Every other launch (gemm, attention, normalization, sampling, cat, index_ops, elementwise...) silently ignores launch failures such as invalid launch config or too-much-shared-memory.
- Change: Wrap launches in a VESPER_HIP_LAUNCH macro that, under !NDEBUG, calls hipGetLastError() (and optionally a scoped hipDeviceSynchronize) and reports the kernel name + config on failure; compile to a bare launch in release so there is no runtime cost.
- Expected: No runtime speedup (and zero overhead in release); surfaces silent launch failures and prevents wrong-result debugging sessions — strictly a correctness/dev-velocity guardrail.
- Risk: Must gate the sync strictly behind NDEBUG, otherwise it would serialize the pipeline in release; the error check alone is cheap.

### [P2] Replace pure power-of-2 binning with finer size classes to cut VRAM waste  (M effort, occupancy, lever=memory pooling / size-classes)
- Kernel: CachingAllocator::allocate (device + host memory pool)
- Current: allocator.cpp:23-34 round_to_bin_size rounds every request up to the next power of 2 (min 512B). A 1.05 GB activation reserves 2 GB; worst-case ~2x over-allocation, and free blocks are never split/coalesced (allocator.cpp:209-241).
- Change: Adopt PyTorch-style rounded size classes (e.g. round large blocks to the nearest 2MB, finer kMinBlock granularity below a threshold) or cap rounding to ~1.25x, and add split/coalesce of cached blocks so a large free block can satisfy smaller requests.
- Expected: Recovers up to ~30-50% of reserved VRAM in the worst case on a 32GB card, enabling larger batch/seqlen -> higher GEMM arithmetic intensity and occupancy. No direct per-kernel speedup; indirect throughput via bigger tiles/batches.
- Risk: More complex free-list bookkeeping; must keep allocate near-O(1) and avoid introducing fragmentation regressions. Add allocator stats/tests before and after.