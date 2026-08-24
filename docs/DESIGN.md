# Design: Vesper inference engine

Vesper is a local LLM inference engine. HIP/AMD is the first GPU target.
The CPU backend is the correctness oracle and the only path that must
build in CI.

The archived training library is not a dependency. Nothing here links it.

## Shape

```
prompt tokens
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  tokenizer  │────▶│    engine    │────▶│   sampler   │
│  (byte v0)  │     │  prefill +   │     │ greedy /    │
└─────────────┘     │  decode loop │     │ top-k later │
                    └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         weights       KV cache      kernels
         (F32 v0)      (linear)      CPU now
                                     HIP next
```

The engine owns one model, one KV arena, and a handful of scratch
buffers. There is no autograd graph and no per-op Tensor object on the
hot path.

## Layers

| Layer | Owns | Must not own |
| --- | --- | --- |
| `Buffer` | Contiguous F32 storage + device | Views, broadcasting, grads |
| `kernels` | RMSNorm, RoPE, GEMV, attention, SwiGLU, softmax | Model structure |
| `ModelWeights` | Embedding, per-layer projections, norms | How they are launched |
| `KVCache` | K/V per layer, write cursor | Attention math |
| `Engine` | Prefill, decode, stats | HTTP, tokenization policy |
| `cli` | Flags, printing tok/s | Kernels |

Backends are selected at `Buffer` allocation and kernel dispatch. Adding
HIP means a sibling `kernels_hip` and a device allocator, not a second
engine.

## Model target

v0 implements a **Qwen3-style dense decoder**:

- pre-norm RMSNorm
- GQA (Q heads ≥ KV heads)
- optional QK-norm (Qwen3)
- RoPE on Q/K, NeoX pair style
- SwiGLU MLP, no biases
- tied or untied lm_head
- linear KV cache, one sequence

That is enough to load Qwen3-0.6B/8B weights once a safetensors/GGUF
reader exists. It is **not** Qwen3.8. Qwen3.8-27B adds Gated DeltaNet
layers, gated full attention, MTP draft heads, and a vision encoder.
Those land after the dense loop is correct and a HIP GEMV exists.

Config factories in `config.cpp` name the real Qwen3-0.6B and Qwen3-8B
shapes so weight loading has a place to land. The runnable demo is a
tiny random model (`ModelConfig::tiny_demo`).

## Decode step

For token `t` at position `p`:

1. embed `t` → `x`
2. for each layer:
   - RMSNorm
   - Q, K, V GEMVs
   - per-head RMSNorm on Q and K if `qk_norm`
   - RoPE at `p`
   - append K, V to that layer's cache
   - GQA attention over `cache[0..p]`
   - output projection + residual
   - RMSNorm
   - gate/up GEMVs, SwiGLU, down GEMV + residual
3. final RMSNorm
4. lm_head GEMV → logits
5. sample

Prefill is the same function in a loop. No separate "training forward".

The required correctness gate: last-token logits from cached decode
match an uncached full-sequence attention over the same written K/V.
See `tests/test_engine.cpp`.

## Memory

v0 KV is dense `[n_layers, max_seq, n_kv_heads, head_dim]` for K and V.
Paged KV comes after HIP decode works. Quantized weights come before
Qwen3-8B will fit on a 24 GiB card in this engine; until then the CPU
path stays F32 and tiny.

## What FreeToken changes

FreeToken's speed is not "faster GEMV on NVIDIA." It is a different
machine model: host RAM holds every MoE expert, VRAM holds a working
set, and decode misses are split between PCIe fill and CPU compute
using two measured bandwidths (\(B_P\), \(B_H\)).

That is the path to FreeToken-like numbers on a Radeon. A dense
Qwen3.8-27B still cannot beat the memory-bandwidth roofline without
MTP. A Qwen3.6-35B-A3B-class MoE can, if we implement:

- a host-resident expert pool (source of truth)
- a GPU LRU of complete `(layer, expert)` slots
- a `ft bench bw` equivalent that profiles HIP copy vs CPU expert GEMV
- the \(q^\star\) split, overlapped
- later: semantic anchors for Gated DeltaNet, once hybrid layers exist

Do not start with NVFP4 or CUDA-graph theatrics. Start with F32/Q4
experts, one sequence, and a correctness gate: hybrid CPU+GPU MoE
output matches fused-GPU MoE output.

## Roadmap

1. **CPU oracle (this tree)** — dense Qwen3 block, KV decode, tests, CLI tok/s.
2. **Weight load** — safetensors F16/BF16 → F32, then GGUF Q4_K / Q8_0.
3. **HIP GEMM/GEMV + RMSNorm + RoPE** for the user's gfx target, with CPU comparison gates.
4. **Fused decode kernel** — norm + QKV + RoPE in one launch; HIP graph the decode step.
5. **Qwen3-8B** on a real AMD card, report decode tok/s vs llama.cpp HIP and Vulkan.
6. **FreeToken-style MoE offload** — host expert pool, GPU LRU, \(q^\star\) hybrid, aimed at Qwen3.6-35B-A3B-class models.
7. **Qwen3.8 hybrid + MTP** — Gated DeltaNet, gated attention, draft/verify, semantic anchors.
8. **Optional Vulkan/RADV backend** if HIP decode loses on RDNA3 the way llama.cpp does.

Dense decode (steps 1–5) stays first. MoE offload is how we chase
FreeToken's numbers; it is not how we debug the first kernel.

## Why not hipify the old kernels

The archived HIP GEMM/attention kernels were written for a training
Tensor API (batched GEMM, backward, dropout). Decode is GEMV, cache
writes, and a 1×seq attention. New kernels, new layouts.
