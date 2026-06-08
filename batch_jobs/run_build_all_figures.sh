#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$SCRIPT_DIR/run_figure1A.sh"
bash "$SCRIPT_DIR/run_figure1B.sh"
bash "$SCRIPT_DIR/run_figure1C.sh"
# bash "$SCRIPT_DIR/run_figure2_cross_model_by_method.sh"
# bash "$SCRIPT_DIR/run_figure4_heatmap.sh"