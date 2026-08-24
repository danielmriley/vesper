# How current inference engines are put together

Date: 2026-08-24.

FreeToken is one point on a map. This note is the map: who else exists,
how big they are, where the speed actually lives, and what a new AMD
engine should copy.

Deep dives:

- [ARCHITECTURE.md](ARCHITECTURE.md) — from-scratch design for max tok/s
- [TOKS.md](TOKS.md) — leverage-ordered tricks for one consumer GPU
- [SMALL_ENGINES.md](SMALL_ENGINES.md) — FlashQwen, Fastgen, ds4, and the rest
- [RESEARCH.md](RESEARCH.md) — FreeToken, Qwen3.8, AMD reality

## Three families

| Family | Job | Typical size | Examples |
| --- | --- | --- | --- |
| **Production servers** | Many users, OpenAI HTTP, datacenter | Huge (50k–300k+ LoC once you count kernels and the model zoo) | vLLM, SGLang, TensorRT-LLM, TokenSpeed, LightLLM, TGI |
| **Local / edge runtimes** | One user, one box, any GGUF tonight | Huge if they own a backend zoo; medium if they wrap someone else's | llama.cpp, Ollama, MLC-LLM, ExLlama, hipEngine, FastLLM, MLX |
| **Small / from-scratch** | One architecture, readable loop | 0.7k–18k | llama2.c, FlashQwen, Fastgen, nano-vllm, ds4 |

Peak tok/s lives in **kernels** (quantized GEMV, attention, MoE dispatch).
Engines become huge in the **scheduler + model zoo + serving shell**.
The shared core that every serious engine reinvented is small:

- continuous / in-flight batching (multi-user)
- paged or token-level KV
- chunked prefill / a constant token budget
- prefix reuse
- CUDA / HIP graphs
- keep the CPU scheduler off the GPU critical path

A new engine that copies the zoo will become vLLM. A new engine that
copies the core, for one model and one GPU class, can stay FlashQwen-sized.

## Production servers

| Engine | Host / kernels | Size | GPU | Distinctive trick |
| --- | --- | --- | --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) | Python + Rust; CUDA / HIP / Triton | Huge | NVIDIA, AMD ROCm, others | PagedAttention, V1 token-budget scheduler, prefix cache |
| [SGLang](https://github.com/sgl-project/sglang) | Python + Rust; CUDA / HIP / FlashInfer | Huge | NVIDIA, MI300-class AMD | RadixAttention, overlap scheduler, structured decode |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Python + C++ + CUDA | Huge | NVIDIA only | In-flight batching, graph padding, NVFP4, PD disagg |
| [TokenSpeed](https://github.com/fw-ai/tokenspeed) | Python exec + C++ FSM | Medium–large | Blackwell + MI350 | Kernel registry, typed KV FSM, agentic TPS floor |
| [LightLLM](https://github.com/ModelTC/LightLLM) | Python + Triton | Medium | NVIDIA-first | TokenAttention (block size 1), small process graph |
| [TGI](https://github.com/huggingface/text-generation-inference) | Rust router + Python engine | Medium | NVIDIA + ROCm + others | Clean serving/engine split. Now maintenance-mode. |
| DeepSpeed-FastGen / MII | Python + DeepSpeed CUDA | Small frontend | NVIDIA sm80+ | Dynamic SplitFuse. Idea won; codebase did not. |

vLLM and SGLang are the default answers to "how do I serve many users."
They are the wrong starting tree for a from-scratch Radeon engine. Steal
the *ideas* (paged KV, radix prefix, token budget). Do not port the
plugin matrix.

TokenSpeed is the one production engine that *refuses* the zoo: C++
owns request/KV lifecycle, Python runs the step, kernels sit behind a
registry. Closest "do not become vLLM" architecture in this row.

TGI's useful lesson is the process split: a non-Python front (Rust/C++)
so HTTP and batching are not GIL-bound. HF now tells people to use
vLLM or SGLang.

## Local and edge

| Engine | Host / kernels | Size | Weights | AMD / other |
| --- | --- | --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C/C++; CUDA, HIP, Vulkan, Metal, CPU SIMD | Huge (~320k if you count every backend) | GGUF | HIP + Vulkan + Metal |
| [Ollama](https://github.com/ollama/ollama) | Go + vendored llama.cpp | Huge as a product | GGUF catalog | ROCm / Vulkan; often lags mainline |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | Python + TVM codegen | Huge with TVM | MLC checkpoint | ROCm, Vulkan, Metal, WebGPU |
| [ExLlamaV2/V3](https://github.com/turboderp-org/exllamav3) | Python + CUDA | Medium | EXL2 / EXL3 | NVIDIA. ROCm is TODO. |
| [koboldcpp](https://github.com/LostRuins/koboldcpp) | llama.cpp + UI | Huge (fork) | GGUF | Vulkan first |
| [llamafile](https://github.com/mozilla-ai/llamafile) | Cosmopolitan + llama.cpp | Medium wrapper | GGUF in a ZIP | ROCm DSO + Vulkan |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | Python + AMX/CUDA | Large | ST + GGUF experts | Hybrid CPU+GPU MoE, not RDNA decode |
| [hipEngine](https://github.com/shisa-ai/hipengine) | Python host + HIP | Small–medium | GGUF, ParoQuant | gfx1100 / gfx1151 only |
| [FastLLM](https://github.com/ztxz16/fastllm) | C++ + CUDA | Medium-large | HF, AWQ, some GGUF | ROCm claimed, not RDNA-tuned |
| [MLX](https://github.com/ml-explore/mlx) | C++ + Metal | Medium | MLX safetensors | Apple only |
| [ORT GenAI](https://github.com/microsoft/onnxruntime-genai) | C++ + ORT EPs | Huge with ORT | ONNX | DirectML on Windows AMD |

llama.cpp is the engine every local project is compared to. Decode is
one fused quantized GEMV (`mul_mat_vec_q`) that community profiles put
near 90% of GPU time. It is hard to beat on single-user decode because
of bits-per-weight, mmap'd GGUF, no Python on the token, and years of
per-backend occupancy work — not because the HIP backend is clever.
On RDNA3, **Vulkan/RADV often wins decode**; **HIP often wins long
prefill**. See [TOKS.md](TOKS.md).

Ollama is a catalog and a Go server around a vendored llama.cpp. It is
not a kernel project. Tuned llama.cpp regularly beats it on the same
file because flags and backend snapshots lag.

hipEngine is the AMD-native sibling of FlashQwen: short model list,
hand-tuned HIP, CPU oracle, evidence-backed benches. AGPL.

ExLlama is the NVIDIA consumer speed king (fused INT4, quantized KV).
Do not hipify it. The PTX (`mma.sync`, `cp.async`, `ldmatrix`) is the
product.

## Small engines (the templates)

Full notes: [SMALL_ENGINES.md](SMALL_ENGINES.md).

| Engine | LoC-ish | GPU | The one trick |
| --- | --- | --- | --- |
| [llama2.c](https://github.com/karpathy/llama2.c) | ~700 C | CPU | The decode loop *is* the program. That is Vesper v0. |
| [FlashQwen](https://github.com/frankkk96/FlashQwen) | <2k C++/CUDA, then a Go gateway | NVIDIA | One model. Token ids in, token ids out. Fuse then graph. |
| [Fastgen](https://github.com/facebookresearch/fastgen) | ~3k Python | NVIDIA | Mini-vLLM inventory (paged KV, graphs, chunked prefill) without the zoo. |
| [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | ~1.2k Python | NVIDIA | Scheduler + block table, readable in an afternoon. Speed is vendor kernels. |
| [mini-sglang](https://github.com/sgl-project/mini-sglang) | ~5k Python | NVIDIA | Overlap scheduling + radix cache, stripped. |
| [ds4](https://github.com/antirez/ds4) | C, grown into a product | Metal, CUDA, **HIP on Strix Halo** | Host/SSD expert cache + native MTP. FreeToken's machine model in C. |
| [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) | llama.cpp fork | NVIDIA + HIPBLAS | Hot/cold *neurons*. Needs ReLU-sparse models. Not stock Qwen3. |
| lmdeploy / TurboMind | C++ + Python | NVIDIA | Persistent batch; LRU of KV caches that collapse to token ids. |

Vesper's three templates, in order:

1. **FlashQwen** — C++ Qwen3 twin. Closed world. Bench after every kernel.
2. **Fastgen** — the objects we do not have yet (paged KV, graphs, host KV).
3. **ds4** — later: HIP without vLLM, expert cache, MTP draft/verify.

Do not start from candle / burn / tinygrad. Those are frameworks. That
is how the old Vesper training library happened.

## Where the complexity lives

```
small and fast          FlashQwen, Fastgen, llama2.c
                        one model, one loop, fused GEMV

clever machine model    FreeToken, ds4, KTransformers
                        GPU is a cache, host is the pool

local catalog           llama.cpp, Ollama, koboldcpp
                        GGUF + every backend + every architecture

production zoo          vLLM, SGLang, TRT-LLM
                        scheduler + HTTP + 200 models + PD disagg
```

Writing a quantized decode GEMV with the right workgroup is days-to-weeks
and is most of single-user tok/s. Writing a vLLM is years of serving
features. Mixing those up is how a new engine dies.

## How we squeeze performance

Decode on one GPU is:

```
tok/s ≈ achieved_bandwidth / bytes_touched_per_token
```

Order of work, not of hype (detail in [TOKS.md](TOKS.md)):

1. No alloc / no autograd on the hot path — already done.
2. Account the roofline so we do not chase a 27B dense "hundreds of tok/s."
3. Load weights. Quantize (Q8, then Q4). Fewer bytes per token.
4. HIP **GEMV**, not a GEMM used at M=1. Wave32. Right occupancy.
5. Fuse RMSNorm+QKV+RoPE and SwiGLU. Then HIP-graph the static decode step.
6. KV layout / KV quant only when context is actually KV-heavy.
7. Speculative decode / MTP — the only way a *dense* 20B+ looks fast.
8. MoE expert LRU + \(q^\star\) — the FreeToken numbers. Zero on dense.
9. Prefix / radix / Gated DeltaNet anchors — multi-turn agents.
10. Continuous batching / paged attention — multi-user. Defer.

On RDNA3, workgroup shape is a first-class trick: llama.cpp HIP often
loses decode to Vulkan because it launches fat wave32 blocks with LDS
reduces on small-K matvecs. Write the GEMV for the chip, or we will
lose to llama.cpp Vulkan on day one.

## What Vesper is, on this map

A small-host, HIP-first, MIT engine. CPU-correct Qwen3 dense loop
today. Next is weight load + HIP GEMV, using FlashQwen's ladder and
llama.cpp's decode contract (GGUF, fused quant GEMV, tiny dispatch).
FreeToken and ds4 are the later machine model, once the GEMV is real.

We are not a vLLM. We are not an Ollama. We will lose to llama.cpp on
coverage for a long time. We should try to beat it on one architecture,
one quant, one GPU class, with a number we can reproduce.
