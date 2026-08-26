#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/compare-qwen38/artifact.env"

backend="${1:-}"
log="${2:-}"
if [[ -z "${backend}" || -z "${log}" || ! -f "${log}" ]]; then
  exit 1
fi

eval_line="$(grep -E 'eval time' "${log}" | grep -v 'prompt eval' | tail -n1 || true)"
prompt_line="$(grep -E 'prompt eval time' "${log}" | tail -n1 || true)"
if [[ -z "${eval_line}" ]]; then
  exit 1
fi

# llama.cpp common_perf_print uses "N runs" for decode. Older
# llama_print_timings and some forks still say "N tokens".
new_tokens="$(printf '%s\n' "${eval_line}" | sed -n 's/.*\/ *\([0-9][0-9]*\) \(runs\|tokens\).*/\1/p')"
decode_tps="$(printf '%s\n' "${eval_line}" | sed -n 's/.* \([0-9][0-9.]*\) tokens per second.*/\1/p')"
prompt_tokens=0
prefill_tps=0
if [[ -n "${prompt_line}" ]]; then
  prompt_tokens="$(printf '%s\n' "${prompt_line}" | sed -n 's/.*\/ *\([0-9][0-9]*\) tokens.*/\1/p')"
  prefill_tps="$(printf '%s\n' "${prompt_line}" | sed -n 's/.* \([0-9][0-9.]*\) tokens per second.*/\1/p')"
fi

if [[ -z "${new_tokens}" || -z "${decode_tps}" ]]; then
  exit 1
fi
if [[ -z "${prompt_tokens}" ]]; then
  prompt_tokens=0
fi
if [[ -z "${prefill_tps}" ]]; then
  prefill_tps=0
fi

bytes="${COMPARE_BYTES_PER_TOKEN:-0}"

achieved="$(awk -v b="${bytes}" -v t="${decode_tps}" 'BEGIN { printf "%.6f", (b * t) / 1e9 }')"

ids="-"
if grep -qE '^ids=' "${log}"; then
  ids="$(sed -n 's/^ids=//p' "${log}" | tail -n1)"
fi

printf 'engine=llamacpp backend=%s model=%s quant=%s arch=%s prompt_tokens=%s new_tokens=%s prefill_tps=%s decode_tps=%s bytes_per_token=%s achieved_gbs=%s peak_gbs=%s context=%s status=ok graphs=- ids=%s\n' \
  "${backend}" "${COMPARE_MODEL}" "${COMPARE_QUANT}" "${COMPARE_ARCH}" \
  "${prompt_tokens}" "${new_tokens}" "${prefill_tps}" "${decode_tps}" \
  "${bytes}" "${achieved}" "${COMPARE_PEAK_GBS}" "${COMPARE_CONTEXT}" \
  "${ids}"
