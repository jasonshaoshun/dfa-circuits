#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python "$PROJECT_ROOT/paper_builders/build_figure2_cross_model_by_method.py" \
  --random-line-mode "${RANDOM_LINE_MODE:-family}" \
  --results-dir "${RESULTS_DIR:-results}" \
  --out-dir "${OUT_DIR:-figures/figure2_cross_model_by_method}" \
  --split "${SPLIT:-test}" \
  --absolute "${ABSOLUTE:-False}" \
  --ablation "${ABLATION:-patching}" \
  --level "${LEVEL:-node}" \
  --select-by "${METRIC_KEY:-area_under}" \
  --modes "${MODES:-zero_shot,in_distribution,near_distribution,best}" \
  ${DROP_MEAN_TASKS:+--drop-tasks-for-mean "$DROP_MEAN_TASKS"} \
  ${EXCLUDE_TASKS:+--exclude-tasks "$EXCLUDE_TASKS"} \
  ${MEAN_POLICY:+--mean-policy "$MEAN_POLICY"} \
  ${EXTRA_ARGS:-}
