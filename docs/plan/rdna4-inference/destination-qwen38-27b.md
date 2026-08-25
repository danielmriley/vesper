# Product. Qwen3.8-27B on the R9700

Back to [overview](overview.md).

This is the model you want to run. It is not Qwen3-8B and it is not
a 3.8B. It is a dense 27B hybrid. The GEMV ladder in phases 1 to 10
is how you get a kernel that can stream those weights. This file is
what you build after that kernel is honest.

## What the model is

`Qwen/Qwen3.8-27B`. Apache-2.0. `model_type` is `qwen3_5`.

- 64 layers, hidden 5120, vocab 248320
- Layout is 16 repeats of three Gated DeltaNet plus FFN, then one
  Gated Attention plus FFN. That is 48 linear-attention layers and
  16 full-attention layers.
- Gated DeltaNet. 48 V heads, 16 QK heads, head dim 128.
- Gated Attention. 24 Q heads, 4 KV heads, head dim 256.
- FFN intermediate 17408, SwiGLU.
- Native MTP heads.
- Native 262K context. Vision encoder on the same stack.

A GQA-plus-SwiGLU loop will load some tensors and then die on the
DeltaNet layers. That is the current engine.

## What efficient means on this card

The R9700 has 32 GB and about 640 GB/s. 32 GB is why Q4 fits. It
does not raise tok/s.

| Quant | Weights | Fit on 32 GB | Decode roofline at 70% |
| --- | --- | --- | --- |
| F16 / BF16 | ~56 GB | no | n/a |
| Q8_0 | ~28 to 31 GB | tight, short context | ~16 tok/s |
| Q6_K | ~23 GB | yes, room for KV | ~20 tok/s |
| Q4_K | ~14 to 18 GB | yes | ~28 to 32 tok/s |

llama.cpp Vulkan already sits on that Q4 band for a dense 27B stream
on this card. That is the one-token ceiling.

Three things make a 27B *feel* faster than 30 tok/s.

1. **A Q4 GEMV that actually reaches ~70% of 640 GB/s.** If you are
   at 40%, you have a 16 tok/s model and no trick saves you.
2. **Gated DeltaNet.** Most layers do not grow a full KV cache.
   Long context stays possible. This does not shrink the weight
   stream. Every token still reads ~15 GB of Q4 weights.
3. **MTP.** Draft K tokens from the trained heads, verify in one
   forward, accept a prefix. Same weight stream, more accepted
   tokens. Typical published gain is about 1.6 to 1.8 times when
   the GEMV is already honest. A 30 tok/s GEMV can look like 50
   accepted tok/s. A 16 tok/s GEMV looks like 25.

Hundreds of tok/s on this dense 27B would need much more bandwidth
or a much smaller active set. This model is not a MoE. FreeToken
numbers do not apply.

Vision stays out until text decode is boring. The mmproj is a
second program.

## Wave 2, after the GEMV gate

Start these only after [phase 10](phase-10-r9700-gate.md) says the
Q4 GEMV is within 15% of llama.cpp Vulkan, or you have chosen a
Vulkan peer and that GEMV is honest.

1. **Config and GGUF map.** `ModelConfig` for hidden 5120, 64 layers,
   the 3-to-1 block, GDN and gated-attn head counts, vocab 248320.
   Bind `qwen3_5` tensor names. Fail on missing DeltaNet tensors.
   Do not run generate yet.
2. **Gated attention, CPU.** 24Q / 4KV, head dim 256. Oracle against
   a recorded reference or a tiny hybrid fixture.
3. **Gated DeltaNet, CPU.** Recurrent state per layer, conv kernel 4,
   linear step. The extra state is not KV. Correctness gate is
   one-step equals a slow reference.
4. **Hybrid block.** One 3-plus-1 group on CPU, Q4 weights, tiny
   fixture. Interleave is a data structure, not a pile of ifs.
5. **HIP for GDN and gated attn.** CPU equals HIP. Same gfx1201
   rules. Do not fuse until unfused matches.
6. **Load Qwen3.8-27B Q4_K, text only.** Generate. Print
   `DecodeReport`. Bytes per token must sit near 14 to 18 GB plus
   the small GDN state and the 16-layer KV.
7. **MTP.** Draft from the native heads, verify, `--strict` equals
   one-token decode. Report accepted tok/s besides decode tok/s.
8. **R9700 gate on this model.** Same Q4_K file against llama.cpp
   HIP and Vulkan, with and without MTP. This is the product
   number.

Write a phase file when you start each item. Do not grow this list
into HTTP, a zoo, or the vision tower.

## Data structures to add then

`HybridBlock` is three `DeltaNetLayer` then one `GatedAttnLayer`.
`DeltaNetState` is the recurrent tensor per layer, sized at load.
`GatedAttnKV` stays linear and only for the 16 full-attn layers.
`MtpHead` is the draft projection already in the GGUF.

Do not store a full `[64, seq, kv, dim]` cache. That layout is
wrong for this model.
