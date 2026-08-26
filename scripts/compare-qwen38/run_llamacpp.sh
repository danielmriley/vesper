#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/compare-qwen38/artifact.env"

backend="${1:-hip}"
print_unsupported() {
  printf 'engine=llamacpp backend=%s model=%s quant=%s arch=%s prompt_tokens=0 new_tokens=0 prefill_tps=0 decode_tps=0 bytes_per_token=0 achieved_gbs=0 peak_gbs=%s context=%s status=unsupported ids=-\n' \
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

if [[ -z "${cli}" || ! -x "${cli}" ]]; then
  print_unsupported
  exit 0
fi

if [[ -z "${COMPARE_GGUF:-}" || ! -f "${COMPARE_GGUF}" ]]; then
  print_unsupported
  exit 0
fi

log="$(mktemp)"
set +e
# llama-completion (and older llama-cli) auto-enable conversation
# when the GGUF has a chat template. Qwen3.8 does. -no-cnv keeps
# this a raw 128-token completion. --ignore-eos matches Vesper,
# which always emits n tokens. Point LLAMA_CLI at llama-completion
# on current llama.cpp; new llama-cli rejects -no-cnv.
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

if [[ "${rc}" -ne 0 ]]; then
  rm -f "${log}"
  print_unsupported
  exit 0
fi

if ! "${ROOT}/scripts/compare-qwen38/parse_llamacpp.sh" "${backend}" "${log}"; then
  rm -f "${log}"
  print_unsupported
  exit 0
fi
rm -f "${log}"
