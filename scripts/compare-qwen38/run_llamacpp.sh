#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/compare-qwen38/artifact.env"

backend="${1:-hip}"
print_unsupported() {
  printf 'engine=llamacpp backend=%s model=%s quant=%s arch=%s prompt_tokens=0 new_tokens=0 prefill_tps=0 decode_tps=0 bytes_per_token=0 achieved_gbs=0 peak_gbs=%s context=%s status=unsupported graphs=- ids=-\n' \
    "${backend}" "${COMPARE_MODEL}" "${COMPARE_QUANT}" "${COMPARE_ARCH}" \
    "${COMPARE_PEAK_GBS}" "${COMPARE_CONTEXT}"
}

if [[ "${COMPARE_FIXTURE:-}" == "1" ]]; then
  print_unsupported
  exit 0
fi

# shellcheck disable=SC1091
source "${ROOT}/scripts/compare-qwen38/check_pin.sh"
if ! compare_pin_ok; then
  print_unsupported
  exit 0
fi

# shellcheck disable=SC1091
source "${ROOT}/scripts/compare-qwen38/resolve_llama_cli.sh"

cli=""
case "${backend}" in
  hip)
    cli="${LLAMA_CLI_HIP:-${LLAMA_CLI:-}}"
    ;;
  vulkan)
    cli="${LLAMA_CLI_VULKAN:-${LLAMA_CLI:-}}"
    ;;
  *)
    print_unsupported
    exit 0
    ;;
esac

if ! cli="$(resolve_llama_cli "${cli}")"; then
  print_unsupported
  exit 0
fi
if [[ ! -x "${cli}" ]]; then
  print_unsupported
  exit 0
fi

if [[ -z "${COMPARE_GGUF:-}" || ! -f "${COMPARE_GGUF}" ]]; then
  print_unsupported
  exit 0
fi

if [[ "${backend}" == "hip" && -z "${GPU_MAX_HW_QUEUES:-}" ]]; then
  export GPU_MAX_HW_QUEUES=1
fi

printf 'compare: llama.cpp %s using %s\n' "${backend}" "${cli}" >&2

log="$(mktemp)"
set +e
# Qwen3.8 GGUFs carry a chat template, so llama-completion (and older
# llama-cli) would enter conversation mode. -no-cnv keeps this a raw
# 128-token completion. --ignore-eos matches Vesper, which always
# emits n tokens. resolve_llama_cli prefers llama-completion because
# current llama-cli rejects -no-cnv.
"${cli}" \
  -m "${COMPARE_GGUF}" \
  -p "${COMPARE_PROMPT}" \
  -n "${COMPARE_N_PREDICT}" \
  -c "${COMPARE_CONTEXT}" \
  --temp 0 \
  --seed 1 \
  -ngl 99 \
  --no-display-prompt \
  --ignore-eos \
  -no-cnv \
  >"${log}" 2>&1
rc=$?
set -e

dump_llama_log() {
  printf 'compare: llama.cpp %s (%s) failed\n' "${backend}" "${cli}" >&2
  if grep -qE 'no-cnv|unrecognised option|unrecognized option' "${log}"; then
    printf 'compare: point LLAMA_CLI at llama-completion; llama-cli dropped -no-cnv\n' >&2
  fi
  tail -n 80 "${log}" >&2
}

if [[ "${rc}" -ne 0 ]]; then
  dump_llama_log
  rm -f "${log}"
  print_unsupported
  exit 0
fi

if ! "${ROOT}/scripts/compare-qwen38/parse_llamacpp.sh" "${backend}" "${log}"; then
  dump_llama_log
  rm -f "${log}"
  print_unsupported
  exit 0
fi
rm -f "${log}"
