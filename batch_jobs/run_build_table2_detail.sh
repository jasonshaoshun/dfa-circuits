#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="${RESULTS_DIR:-results}"
OUTPUT_DIR="${OUTPUT_DIR:-tables}"
METRIC="${METRIC:-area_under}"
SPLIT="${SPLIT:-test}"

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python "$PROJECT_ROOT/paper_builders/build_table2_detail.py" \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --metric "$METRIC" \
  --split "$SPLIT"
