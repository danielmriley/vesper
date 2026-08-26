#!/usr/bin/env bash
# Same-file Qwen3.8-27B Q4_K_M table: llama.cpp HIP, llama.cpp Vulkan, Vesper HIP.
# CI: COMPARE_FIXTURE=1 (no GGUF, three unsupported rows).
# R9700: COMPARE_GGUF=/path/to/Qwen3.8-27B-Q4_K_M.gguf
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/compare-qwen38/artifact.env"
# shellcheck disable=SC1091
source "${ROOT}/scripts/compare-qwen38/check_pin.sh"

field() {
  local line="$1"
  local key="$2"
  printf '%s\n' "${line}" | sed -n "s/.*${key}=\\([^ ]*\\).*/\\1/p"
}

cell() {
  local line="$1"
  local key="$2"
  local status
  status="$(field "${line}" status)"
  if [[ "${status}" == "unsupported" ]]; then
    printf 'unsupported'
    return
  fi
  field "${line}" "${key}"
}

hip="$("${ROOT}/scripts/compare-qwen38/run_llamacpp.sh" hip)"
vulkan="$("${ROOT}/scripts/compare-qwen38/run_llamacpp.sh" vulkan)"
vesper="$(COMPARE_BACKEND=hip "${ROOT}/scripts/compare-qwen38/run_vesper.sh")"

vesper_commit="$(git -C "${ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
llama_commit="${LLAMA_CPP_COMMIT:-unknown}"
gguf_path="${COMPARE_GGUF:-missing}"
sha="${COMPARE_SHA256}"
if [[ "${COMPARE_FIXTURE:-}" != "1" ]] && compare_pin_ok; then
  gguf_path="${COMPARE_GGUF}"
fi

printf '# compare %s %s\n' "${COMPARE_MODEL}" "${COMPARE_QUANT}"
printf '# sha256 %s\n' "${sha}"
printf '# vesper %s\n' "${vesper_commit}"
printf '# llamacpp %s\n' "${llama_commit}"
printf '# prompt %s\n' "${COMPARE_PROMPT}"
printf '# n_predict %s\n' "${COMPARE_N_PREDICT}"
printf '# context %s\n' "${COMPARE_CONTEXT}"
printf '# gguf %s\n' "${gguf_path}"
printf '\n'
printf '| engine | backend | decode_tps | achieved_gbs | bytes_per_token | graphs | status | ids |\n'
printf '| --- | --- | --- | --- | --- | --- | --- | --- |\n'

print_row() {
  local line="$1"
  printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
    "$(field "${line}" engine)" \
    "$(field "${line}" backend)" \
    "$(cell "${line}" decode_tps)" \
    "$(cell "${line}" achieved_gbs)" \
    "$(cell "${line}" bytes_per_token)" \
    "$(cell "${line}" graphs)" \
    "$(field "${line}" status)" \
    "$(cell "${line}" ids)"
}

print_row "${hip}"
print_row "${vulkan}"
print_row "${vesper}"

h_status="$(field "${hip}" status)"
v_status="$(field "${vulkan}" status)"
s_status="$(field "${vesper}" status)"
if [[ "${h_status}" == "ok" && "${v_status}" == "ok" && "${s_status}" == "ok" ]]; then
  awk -v h="$(field "${hip}" decode_tps)" -v v="$(field "${vulkan}" decode_tps)" \
      -v s="$(field "${vesper}" decode_tps)" 'BEGIN {
    win = "llamacpp-hip"
    w = h + 0
    if (v + 0 > w) { win = "llamacpp-vulkan"; w = v + 0 }
    if (s + 0 > w) { win = "vesper-hip"; w = s + 0 }
    printf "\nwinner engine=%s decode_tps=%s\n", win, w
  }'
fi
