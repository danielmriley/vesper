# Vesper

AMD-first local LLM inference engine. C++20, no Python on the hot path,
CPU backend as the correctness oracle. v1 GPU target is the
**Radeon AI Pro R9700 (RDNA 4, gfx1201)** — [docs/TARGET.md](docs/TARGET.md).

The previous Vesper project — a PyTorch-like training library — is frozen
on [`cursor/clearmain-47fb`](https://github.com/danielmriley/vesper/tree/cursor/clearmain-47fb).
This tree starts empty and only implements inference.

## Status

v0 is a working dense decoder (Qwen3-style: RMSNorm, GQA, RoPE, SwiGLU,
QK-norm) with a linear KV cache. It runs a tiny random model on CPU and
checks that cached decode matches full-sequence attention.

It loads official `qwen35` / `qwen3_5` GGUFs (Gated DeltaNet, gated
attention, SwiGLU, packed Q4_K/Q5_K/Q6_K/Q8_0). Tok/s vs llama.cpp
is measured on the R9700 with `scripts/compare-qwen38/compare.sh`,
not on CI. The product is a from-scratch GGUF engine, not a
llama.cpp fork. Architecture notes are in
[docs/plan/r9700-engine/overview.md](docs/plan/r9700-engine/overview.md).
The older GEMV ladder and compare notes stay in
[docs/plan/rdna4-inference/overview.md](docs/plan/rdna4-inference/overview.md)
and
[docs/plan/qwen38-compare/overview.md](docs/plan/qwen38-compare/overview.md).
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

Needs a C++20 compiler. HIP is optional and gfx1201-only.

## Run

```bash
./build/vesper-infer --inspect path/to/model.gguf
./build/vesper-infer --write-tiny /tmp/vesper-tiny.gguf
./build/vesper-infer --model /tmp/vesper-tiny.gguf --prompt "hello" --tokens 32
./build/vesper-infer --demo --prompt "hello" --tokens 32
./build/vesper-infer --demo --device hip --prompt "hello" --tokens 32
./build/vesper-infer --bench-q8
```

`--inspect` maps a GGUF v3 file and prints architecture plus tensors.
`--write-tiny` writes the 2-layer demo as a Q8_0 `vesper_tiny` GGUF.
`--model` loads `vesper_tiny`, `vesper_hybrid`, `qwen35`, or `qwen3_5`.
On the R9700, point it at `ggml-org/Qwen3.8-27B-GGUF` `Q4_K_M` and
`--device hip`. Same-file table:

```bash
COMPARE_GGUF=/path/to/Qwen3.8-27B-Q4_K_M.gguf ./scripts/compare-qwen38/compare.sh
```

`--bench-q4` times the fused Q4_K GEMV against the 640 GB/s pin.

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
