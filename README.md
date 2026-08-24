# Vesper

AMD-first local LLM inference engine. C++17, no Python on the hot path,
CPU backend as the correctness oracle, HIP planned for Radeon GPUs.

The previous Vesper project — a PyTorch-like training library — is frozen
on [`cursor/clearmain-47fb`](https://github.com/danielmriley/vesper/tree/cursor/clearmain-47fb).
This tree starts empty and only implements inference.

## Status

v0 is a working dense decoder (Qwen3-style: RMSNorm, GQA, RoPE, SwiGLU,
QK-norm) with a linear KV cache. It runs a tiny random model on CPU and
checks that cached decode matches full-sequence attention.

It does **not** yet load Qwen weights, talk to a GPU, or claim tok/s on
Qwen3.8. See [docs/RESEARCH.md](docs/RESEARCH.md) and
[docs/DESIGN.md](docs/DESIGN.md).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Needs a C++17 compiler. No BLAS, no ROCm, no Python.

## Run

```bash
./build/vesper-infer --demo --prompt "hello" --tokens 32
```

The demo uses a 2-layer random model and a byte tokenizer. Output is not
a language model. It exists to exercise prefill/decode and print tok/s.

## Why this exists

Projects such as TokenSpeed and FlashQwen publish very high decode
numbers on NVIDIA. They do not run on Radeon. The useful AMD references
are llama.cpp (HIP and Vulkan) and hipEngine (HIP-first, gfx1100/gfx1151).
Vesper's job is a small MIT-licensed engine that we can tune on AMD
hardware, starting from a CPU-correct loop.

## License

MIT. See [LICENSE](LICENSE).
