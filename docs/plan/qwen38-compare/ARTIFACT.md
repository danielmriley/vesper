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

Do not pin Unsloth `UD-*` files or uncensored forks. Do not commit the GGUF.

`scripts/compare-qwen38/artifact.env` exports the same fields. On the R9700, set `COMPARE_GGUF` to the downloaded path and rerun `scripts/compare-qwen38/run.sh`.
