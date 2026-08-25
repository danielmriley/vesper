# R9700 from-scratch engine

This is the product architecture. The first code is a GGUF loader.
It is not a llama.cpp fork and it is not an Ollama wrapper.

## How the current tree works

`vesper-infer --demo` builds `ModelWeights::random`, constructs
`Engine`, and calls `generate`. Every weight and activation is an
F32 `Buffer`. Kernels live in `src/kernels_cpu.cpp` and optional
F32 HIP twins. There is no file load and no `--model`.

That loop is the CPU oracle. It stays. GGUF does not go into
`Buffer`. `Buffer` is float storage for activations. Weights that
arrive as Q4_K blocks need a typed view of file bytes.

## What we are building

A C++20 HIP engine on one card, the Radeon AI Pro R9700
(`gfx1201`, 64 CUs, 32 GB, about 640 GB/s, wave32, 256 B
cacheline, 64 KB LDS per CU, 64 MB Infinity Cache).

The user-facing loop is `vesper-infer --model FILE --prompt TEXT`.
A later chat REPL can sit on the same `Engine`. The engine must
load GGUF itself.

We study the public GGUF and GGML block sizes. We do not vendor
ggml, copy llama.cpp kernels, or link ROCm-llama.

## Hardware we will not invent

`docs/TARGET.md` is the pin. LDS is 64 KB per CU, not 128 KB.
Wave32 is the first launch. Wave64 is a later measure, not the
default. There is no CDNA FP8 MFMA. Decode is bandwidth. Prefill
may use WMMA. HIP sets `GPU_MAX_HW_QUEUES=1`.

Honest Q4 decode on a dense 27B at 70% of 640 GB/s is about 28 to
32 tok/s. Beating llama.cpp Vulkan by 15 to 25% is the hillclimb
after a fused Q4 GEMV is honest. It is not a milestone-1 gate.

C++20 is the language pin. C++23 modules wait.

## Data shape

`GgufFile` owns a POSIX mmap of one file. That is the organizing
structure. Metadata is a key map. Tensors are a name map of views
into the map. Nothing copies a 19 GB weight blob on load.

```
GgufFile
  path, version, alignment, file_size
  kv: map<string, GgufValue>
  tensors: vector<GgufTensor>
  by_name: map<string, index>

GgufValue
  kind (scalar, string, array)
  payload

GgufTensor
  name, type, dims[1..4], offset, nbytes
  data(): const byte* into the map
```

`ggml_nbytes(type, dims)` is a table. Unknown types fail at parse.
A later `bind_qwen3(const GgufFile&)` fills `ModelConfig` from
`{arch}.*` keys and keeps tensor views. That bind is milestone 2.

## Alternatives

**A. mmap views (this plan).**
One `GgufFile` owns the map. Tensors are pointers. A 19 GB Q4
file fits the 32 GB card without a host copy of the same bytes.

**B. Parse, then copy every tensor into owned `vector<byte>`.**
Simpler lifetime. Doubles host RAM on load. Rejected for this
card.

**C. Link llama.cpp and wrap `llama_model`.**
You get chat this week. You do not own the GEMV. The user asked
for an original runtime. Rejected for the engine. A later product
shell may still *compare* against llama.cpp.

## First three milestones

### M1. Inspect a GGUF

`GgufFile::open`, a synthetic writer in tests, and
`vesper-infer --inspect FILE`.

**Done.** `ctest` writes a tiny GGUF, opens it, and checks magic,
version, one KV, one Q8_0 tensor, and one F32 tensor. The inspect
CLI prints architecture, tensor count, and each name, type, shape,
and nbytes. A truncated file and a bad magic fail with a clear
error. CI never downloads a real model.

### M2. CPU Q8_0 GEMV and one loaded decode

Dequant + GEMV on CPU against F32. Bind a tiny Q8 GGUF (or the
demo written as Q8) into the existing decoder. Greedy ids match
the F32 oracle on the same weights.

**Done.** A test builds Q8_0 blocks from known F32, runs GEMV, and
stays within a tight error. `vesper-infer --model tiny.gguf`
generates without `--demo`.

### M3. HIP fused Q8_0 GEMV on gfx1201

One kernel. Wave32. 256 B aligned rows. Print bytes per token and
achieved GB/s. CPU equals HIP on a fixture.

**Done.** On the R9700, achieved GB/s is at least 40% of 640 on
the GEMV microbench, or we write why and change the workgroup
before any fusion or FlashAttention. Then Q4_K.

Qwen2, Llama, Mistral, MoE, quantized KV, and a chat REPL come
after M3 is honest. Qwen3.8 hybrid (Gated DeltaNet) is a later
bind, not M1.

## Applicable skills

- `how` before editing `Engine` or `Buffer`
- `architect` before a second storage type
- `/deslop` and `unslop` on each diff
- `control-cli` for `--inspect` and later `--model`
- `interrogate` before we claim a tok/s win against llama.cpp
