# Vesper

AMD-first local LLM inference engine. C++17, no Python on the hot path,
CPU backend as the correctness oracle. v1 GPU target is the
**Radeon AI Pro R9700 (RDNA 4, gfx1201)** — [docs/TARGET.md](docs/TARGET.md).

The previous Vesper project — a PyTorch-like training library — is frozen
on [`cursor/clearmain-47fb`](https://github.com/danielmriley/vesper/tree/cursor/clearmain-47fb).
This tree starts empty and only implements inference.

## Status

v0 is a working dense decoder (Qwen3-style: RMSNorm, GQA, RoPE, SwiGLU,
QK-norm) with a linear KV cache. It runs a tiny random model on CPU and
checks that cached decode matches full-sequence attention.

It does **not** yet load Qwen weights or claim tok/s on Qwen3.8.
The v1 method is the phase plan in
[docs/plan/rdna4-inference/overview.md](docs/plan/rdna4-inference/overview.md).
Notes: [hardware target](docs/TARGET.md),
[architecture](docs/ARCHITECTURE.md), [research](docs/RESEARCH.md),
[v0 design](docs/DESIGN.md), [engine landscape](docs/ENGINES.md),
[tok/s order](docs/TOKS.md), [small engines](docs/SMALL_ENGINES.md).

## Build

CPU oracle (CI / machines without ROCm):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

R9700 (RDNA 4), ROCm 7.x:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVESPER_USE_HIP=ON
cmake --build build -j
./build/vesper-infer --hip-info
```

Needs a C++17 compiler. HIP is optional and gfx1201-only.

## Run

```bash
./build/vesper-infer --demo --prompt "hello" --tokens 32
./build/vesper-infer --demo --device hip --prompt "hello" --tokens 32
```

The demo uses a 2-layer random model and a byte tokenizer. Output is not
a language model. It exists to exercise prefill/decode and print tok/s.

## Why this exists

[FreeToken](https://github.com/FlashML-org/FreeToken) is the engine
behind the consumer-GPU tok/s posts. It is NVIDIA-only (RTX 30/40/50,
CUDA 13) and gets its speed from MoE expert caching plus CPU–GPU
offload, not from running a dense 27B at hundreds of tok/s. The useful
AMD references are llama.cpp (HIP and Vulkan) and hipEngine. Vesper's
job is a small MIT-licensed engine we can tune on Radeon: dense loop
first, then a FreeToken-style expert cache on HIP.

## License

MIT. See [LICENSE](LICENSE).
