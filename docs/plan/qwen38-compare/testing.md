# Compare plan. What CI can prove

The Cloud Agent VM has no GPU and must not download a 19 GB GGUF.
Most of this plan is R9700-only. CI still has to prove the scripts
do not invent numbers.

## Always on `ctest` (no GPU, no GGUF)

| Check | Proves |
| --- | --- |
| `DecodeReport` parse fixture | a known line round-trips. A truncated line is rejected. |
| `COMPARE_FIXTURE=1` llama.cpp HIP runner | the script parses a checked-in llama.cpp log snippet into a report line. |
| `COMPARE_FIXTURE=1` llama.cpp Vulkan runner | same for a Vulkan snippet. |
| `COMPARE_FIXTURE=1` Vesper runner | prints `status=unsupported` or a fixture report. Never a made-up tok/s. |
| `COMPARE_FIXTURE=1 compare.sh` | prints a three-row table from fixtures. |
| `artifact.env` exists and lists sha256, prompt, n, ctx | the pin file is in git. The GGUF is not. |

## R9700 only

| Check | Proves |
| --- | --- |
| `run_llamacpp_hip.sh` | real HIP tok/s for the pinned GGUF. |
| `run_llamacpp_vulkan.sh` | real Vulkan tok/s for the same file. |
| `run_vesper.sh` | real Vesper tok/s, or honest `unsupported`. |
| `compare_greedy.sh` | 32 greedy ids match, once Vesper generates. |
| `compare.sh` | the table a human can paste. |

## What this plan does not claim

A green `ctest` here does not mean Vesper runs Qwen3.8-27B.

A llama.cpp row does not mean Vesper is close.

A Vesper `unsupported` row is a passing script, not a passing engine.
