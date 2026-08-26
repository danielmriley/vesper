# Pinned compare artifact

One GGUF. Three rows. No download in CI.

| Field | Value |
| --- | --- |
| Repo | `ggml-org/Qwen3.8-27B-GGUF` |
| Revision | `97c30c65c8d9a3e73f9fdfb50f1d1a669e9a2827` |
| Filename | `Qwen3.8-27B-Q4_K_M.gguf` |
| Quant | `Q4_K_M` |
| SHA256 | `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| Prompt | `The capital of France is` |
| `n_predict` | 128 |
| Context | 4096 |
| MTP | off |

Source checkpoint is `Qwen/Qwen3.8-27B`. Architecture in the file is `qwen35`.

The pin was converted with `--no-mtp`, then quantized as Q4_K_M with attn/ssm projections forced to Q8_0 and `output.weight` to Q6_K. `convert.log` on that revision dumps the header:

| Key | Value |
| --- | --- |
| `qwen35.block_count` | 64 (no NextN block) |
| `qwen35.context_length` | 262144 |
| `qwen35.full_attention_interval` | 4 |
| `qwen35.attention.recurrent_layers` | absent; interval is the map |
| `qwen35.ssm.{conv_kernel,state_size,group_count,time_step_rank,inner_size}` | 4 / 128 / 16 / 48 / 6144 |
| `qwen35.rope.dimension_sections` | `[11, 11, 10, 0]` |
| `tokenizer.ggml.pre` | `qwen35` |
| `token_embd.weight` | Q4_K |
| `output.weight` | Q6_K |
| attn / GDN projections (`attn_*`, `ssm_alpha/beta/out`) | Q8_0 |
| `attn_output` | Q6_K |
| FFN | Q4_K |

Do not pin Unsloth `UD-*` files or uncensored forks. Do not commit the GGUF.

`scripts/compare-qwen38/artifact.env` exports the same fields. `COMPARE_BYTES_PER_TOKEN` is `18237132800`: packed linear weights in the official mix, not the 19 GB file size. All three table rows use that number for GB/s. On the R9700, set `COMPARE_GGUF` to the downloaded path and run `scripts/compare-qwen38/compare.sh`. The script hashes the file against this pin, then prints llama.cpp HIP, llama.cpp Vulkan, and Vesper HIP as one table. `COMPARE_FIXTURE=1` prints the same table with `unsupported` cells and does not touch the 19 GB file.

`--inspect` on this file prints `pin_header qwen38-27b-q4km yes` from the header alone. `--load-check` mmap-loads the tensors and prints `pin_weights qwen38-27b-q4km yes` when `linear_bytes` is `18237132800`. Compare-prompt ids from the file tokenizer are `760,6511,314,9338,369`. CPU greedy `--tokens 1` on that prompt emitted `11751` and decoded to `The capital of France is Paris`.
