#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="${RESULTS_DIR:-results}"
OUTPUT_DIR="${OUTPUT_DIR:-tables}"
METRIC="${METRIC:-area_under}"
SPLIT="${SPLIT:-test}"

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python "$PROJECT_ROOT/paper_builders/build_figure2_cross_model.py" \
  --tables-dir "$OUTPUT_DIR" \
  --out-dir "figures/figure2_cross_model" \
  --drop-tasks-for-mean arithmetic_addition \
  --mean-plot bar
