# v1 hardware target: Radeon AI Pro R9700

The first GPU we optimize for is the **AMD Radeon AI Pro R9700**.
It is **RDNA 4**, not RDNA 3.

| | |
| --- | --- |
| Card | [Radeon AI Pro R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) |
| Architecture | RDNA 4 (Navi 48) |
| LLVM / HIP target | `gfx1201` |
| CUs / SPs | 64 / 4096 |
| Wavefront | **Wave32** |
| VRAM | 32 GB GDDR6 |
| Peak bandwidth | **640 GB/s** (measured MCLK 1258 MHz on community benches) |
| LDS / CU | 64 KB |
| L1 / L2 / L3 | 32 KB / 8 MB / 64 MB |
| Cacheline | **256 B** (stricter than RDNA3's 128 B) |
| FP16 matrix | ~191 TFLOPS |
| Official ROCm | Yes (Radeon AI Pro, Ubuntu 24.04 / 22.04, RHEL 9/10) |

RDNA 3 (7900 XTX / W7900, `gfx1100`, ~960 GB/s, 24 GB) and Strix Halo
(`gfx1151`) stay in the notes as later peers. Do not write the first
HIP kernels as a gfx1100 port. Compile with
`-DCMAKE_HIP_ARCHITECTURES=gfx1201` / `VESPER_HIP_ARCH=gfx1201`.

## What 32 GB and 640 GB/s change

Compared with a 24 GB 7900 XTX, this card has **more VRAM and less
bandwidth**. Decode tok/s is bandwidth / bytes. Prefill and "does it
fit" are the 32 GB win.

Honest ~70% of 640 GB/s is ~450 GB/s.

| Model | Quant | Weights | Fits? | Decode roofline (~70%) |
| --- | --- | --- | --- | --- |
| Qwen3-8B | F16 | ~16.4 GB | yes, with room for KV | ~27 tok/s |
| Qwen3-8B | Q8 | ~8.5 GB | yes, long context | ~53 |
| Qwen3-8B | Q4 | ~4.5–5 GB | yes | ~90–100 |
| Qwen3.8-27B | Q8 | ~28 GB | tight; short context | ~16 |
| Qwen3.8-27B | Q4 | ~14–16 GB | yes | ~28–32 |
| Qwen3.6-35B-A3B | Q4 active ~1.5 GB | full pack ~18 GB | yes | hundreds if routing/dequant stay cheap |

llama.cpp on this card already sees dense-27B decode around
**29–33 tok/s** (Vulkan), which is ~70% of 640 GB/s on a ~15.6 GiB
stream. That is the ceiling to beat or match, not 7900 XTX math.

F16 Qwen3-8B is a legal first GPU target here. On 24 GB it was not.

## Kernel rules for gfx1201

From AMD's ROCm docs, community R9700 llama.cpp work, and the
[Hugging Face R9700 kernel guide](https://github.com/huggingface/kernels/blob/main/kernel-builder/skills/rocm-kernels/references/r9700-optimization-guide.md):

- **Wave32.** Reductions are 16…1, not 32…1. Do not copy CDNA wave64
  shuffle distances. `__launch_bounds__` and occupancy assume 32-wide
  waves. Max 32 waves/CU.
- **Align 256 B.** Weight rows, KV tiles, and staging buffers. 128 B
  packing that was "good enough" on gfx1100 will miss more here.
- **LDS 64 KB.** Prefill GEMM tiles stay small (64–128). `num_stages=2`.
  A 128×128×64 FP16 double-buffered tile is already the LDS wall.
- **Grid multiples of 64** (one block per CU as a starting guess).
- **FP16 for matrix/prefill.** AMD lists FP8 as a software type on
  this card; there is **no CDNA-style FP8 MFMA**. Decode stays Q8/Q4
  integer-dot + FP32 accumulate until a measured FP16 path wins.
- **L3 is 64 MB.** A Q4-8B working set can live there. Do not assume
  the 8 MB L2 holds a layer.
- **HIP idle power.** Community ROCm on R9700 sits at 70–100 W with
  a model loaded and nothing running. `GPU_MAX_HW_QUEUES=1` is the
  known workaround. Vulkan does not have that bug. Document it on
  the HIP path; do not "fix" it by adding queues.
- **Vulkan is still a peer.** RDNA4 did not make HIP automatically
  faster than RADV on decode. Measure both. Ship HIP first because
  that is the engine we are writing; add a Vulkan backend only if
  HIP loses by a real margin on the same Q4.

## Build pin

```text
VESPER_HIP_ARCH=gfx1201
CMAKE_HIP_ARCHITECTURES=gfx1201
GPU_TARGETS=gfx1201
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVESPER_USE_HIP=ON
cmake --build build -j
./build/vesper-infer --hip-info
./build/vesper-infer --demo --device hip --prompt "hello" --tokens 32
```

ROCm 7.x with native gfx1201. No `HSA_OVERRIDE_GFX_VERSION` hacks.
The runtime sets `GPU_MAX_HW_QUEUES=1` if unset (R9700 idle-power bug)
and refuses any arch other than gfx1201.

Sources: [AMD R9700 product page](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html),
[ROCm system requirements](https://rocmdocs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html),
[llama.cpp R9700 decode ceiling](https://github.com/ggml-org/llama.cpp/discussions/21043).
