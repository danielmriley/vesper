#!/usr/bin/env bash
# Source after artifact.env. compare_pin_ok returns 0 when COMPARE_GGUF matches the pin.

compare_pin_ok() {
  if [[ -z "${COMPARE_GGUF:-}" || ! -f "${COMPARE_GGUF}" ]]; then
    return 1
  fi
  if [[ "${COMPARE_SKIP_SHA:-}" == "1" ]]; then
    return 0
  fi
  if ! command -v sha256sum >/dev/null 2>&1; then
    echo "sha256sum missing; set COMPARE_SKIP_SHA=1 to skip the pin check" >&2
    return 1
  fi
  local got
  got="$(sha256sum "${COMPARE_GGUF}" | awk '{print $1}')"
  if [[ "${got}" != "${COMPARE_SHA256}" ]]; then
    echo "COMPARE_GGUF sha256 ${got} != pin ${COMPARE_SHA256}" >&2
    return 1
  fi
  return 0
}
