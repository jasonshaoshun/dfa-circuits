#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python "$PROJECT_ROOT/paper_builders/build_figure4_heatmap.py" \
  --results-dir results \
  --output-dir figures \
  --metric-key area_under \
  --ablation patching \
  --level node \
  --split test
