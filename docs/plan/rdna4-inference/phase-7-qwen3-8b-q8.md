# Phase 7. Qwen3-8B Q8

Back to [overview](overview.md).

## Goal

Vesper loads a real Qwen3-8B Q8_0 GGUF, generates on the R9700, and
prints a `DecodeReport`. You can compare that report to llama.cpp by
hand. This phase does not require a win.

## Changes

`src/weights.cpp` maps GGUF names onto `ModelConfig::qwen3_8b()`
slots. `src/cli.cpp` grows a `--model PATH` flag. Keep the byte
tokenizer if the real tokenizer is not ready. If you need a tokenizer,
that is a split phase. Do not hide it here.

## Data structures

`ModelConfig::qwen3_8b()` is already the shape. `GgufFile` plus
`PackedMatrix` per projection. Fail if a required tensor is missing
or the type is not Q8_0.

## Verification

**Static.** Fixture test that a recorded tensor name list binds to
the 8B slots.

**Runtime.** On a R9700, with a local Qwen3-8B Q8 GGUF:

```bash
./build/vesper-infer --model /path/to/qwen3-8b-q8.gguf --prompt hello --tokens 32 --device hip
```

`control-cli` captures the `DecodeReport`. Bytes per token must sit
near 8.5 GB of weights plus KV for that context. If achieved GB/s is
under 40% of 640, stop and fix the GEMV. Do not start Q4.
