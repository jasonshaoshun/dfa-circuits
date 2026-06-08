#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python "$PROJECT_ROOT/paper_builders/build_figure1C_task_grid.py" \
  --results-dir "${RESULTS_DIR:-results}" \
  --out-dir "${OUT_DIR:-figures/figure1C_task_grid}" \
  --split "${SPLIT:-test}" \
  --absolute "${ABSOLUTE:-False}" \
  --ablation "${ABLATION:-patching}" \
  --level "${LEVEL:-node}" \
  --aligned-mode 'zero_shot' \
  --no-title \
  ${EXTRA_ARGS:-}
