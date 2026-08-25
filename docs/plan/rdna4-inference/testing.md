# How to verify the RDNA4 plan

Back to [overview](overview.md).

## Merge gate, every phase

These commands run on a machine without a GPU.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/vesper-infer --demo --prompt hello --tokens 8
./build/vesper-infer --hip-info
```

Drive `vesper-infer` with the `control-cli` skill. Check the printed
fields, not only the exit code.

## After phase 1

The demo print includes every `DecodeReport` field. A test covers
the roofline math.

## After phase 5, on a R9700

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVESPER_USE_HIP=ON
cmake --build build -j
./build/vesper-infer --hip-info
ctest --test-dir build --output-on-failure
```

`--hip-info` must show `gfx1201`. Any other arch must refuse HIP.

## After phase 7 and phase 9, on a R9700

Run the real GGUF through `--device hip`. Read `DecodeReport`.
If achieved GB/s is under 40% of 640, the GEMV is not done.

## Phase 10, on a R9700

This wave proves the GEMV on Qwen3-8B Q4_K. The product compare on
Qwen3.8-27B is in [destination-qwen38-27b.md](destination-qwen38-27b.md).

Same Qwen3-8B Q4_K file, prompt, and token count:

1. `vesper-infer --model ... --device hip`
2. llama.cpp HIP
3. llama.cpp Vulkan

Keep the three transcripts. The 15% Vulkan gate is the product
check. CI cannot run it.

## Surfaces this plan does not cover

- HIP occupancy and wave counts. No control skill. Use `rocprof` on
  the R9700 if a GEMV misses 40% of peak.
- Browser or UI. None in v1.
