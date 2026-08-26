#!/usr/bin/env bash
# Map a user-supplied llama.cpp path to the binary that still accepts
# -no-cnv. Current llama-cli rejects that flag; llama-completion does not.

resolve_llama_given() {
  local cli="$1"
  if [[ -d "${cli}" ]]; then
    if [[ -x "${cli}/llama-completion" ]]; then
      printf '%s\n' "${cli}/llama-completion"
      return 0
    fi
    if [[ -x "${cli}/llama-cli" ]]; then
      printf '%s\n' "${cli}/llama-cli"
      return 0
    fi
    return 1
  fi
  local base
  base="$(basename -- "${cli}")"
  if [[ "${base}" == "llama-cli" ]]; then
    local dir
    dir="$(dirname -- "${cli}")"
    if [[ -x "${dir}/llama-completion" ]]; then
      printf '%s\n' "${dir}/llama-completion"
      return 0
    fi
  fi
  if [[ -x "${cli}" ]]; then
    printf '%s\n' "${cli}"
    return 0
  fi
  return 1
}

resolve_llama_cli() {
  local given="${1:-}"
  if [[ -n "${given}" ]]; then
    resolve_llama_given "${given}"
    return
  fi
  local found
  found="$(command -v llama-completion || true)"
  if [[ -n "${found}" && -x "${found}" ]]; then
    printf '%s\n' "${found}"
    return 0
  fi
  found="$(command -v llama-cli || true)"
  if [[ -n "${found}" ]]; then
    resolve_llama_given "${found}"
    return
  fi
  return 1
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
  resolve_llama_cli "${1:-}"
fi
