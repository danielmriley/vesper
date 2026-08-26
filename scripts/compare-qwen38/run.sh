#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"${ROOT}/scripts/compare-qwen38/run_llamacpp.sh" hip
"${ROOT}/scripts/compare-qwen38/run_llamacpp.sh" vulkan
COMPARE_BACKEND=hip "${ROOT}/scripts/compare-qwen38/run_vesper.sh"
