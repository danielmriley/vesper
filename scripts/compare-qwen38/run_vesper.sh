#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/compare-qwen38/artifact.env"

backend="${COMPARE_BACKEND:-hip}"
if [[ "${COMPARE_FIXTURE:-}" == "1" ]]; then
  backend="cpu"
fi

print_unsupported() {
  printf 'engine=vesper backend=%s model=%s quant=%s arch=%s prompt_tokens=0 new_tokens=0 prefill_tps=0 decode_tps=0 bytes_per_token=0 achieved_gbs=0 peak_gbs=%s context=%s status=unsupported\n' \
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

bin="${ROOT}/build/vesper-infer"
if [[ ! -x "${bin}" ]]; then
  print_unsupported
  exit 0
fi

log="$(mktemp)"
set +e
"${bin}" --model "${COMPARE_GGUF}" --prompt "${COMPARE_PROMPT}" \
  --tokens "${COMPARE_N_PREDICT}" --device "${backend}" \
  --context "${COMPARE_CONTEXT}" --report-only \
  >"${log}" 2>&1
rc=$?
set -e

if [[ "${rc}" -ne 0 ]]; then
  cat "${log}" >&2
  rm -f "${log}"
  print_unsupported
  exit 0
fi

line="$(grep -E '^engine=vesper ' "${log}" | tail -n1 || true)"
rm -f "${log}"
if [[ -z "${line}" ]]; then
  print_unsupported
  exit 0
fi
printf '%s\n' "${line}"
