#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$SCRIPT_DIR/run_build_table1_leaderboard.sh"
# bash "$SCRIPT_DIR/run_build_table2_detail.sh"
# bash "$SCRIPT_DIR/run_build_table2_summary.sh"
# bash "$SCRIPT_DIR/run_build_table3_detail.sh"
# bash "$SCRIPT_DIR/run_build_table3_summary.sh"
# bash "$SCRIPT_DIR/run_build_table4_heatmap.sh"



# bash "$SCRIPT_DIR/run_build_figure2_cross_model.sh"
# bash "$SCRIPT_DIR/run_build_figure2_cross_model_by_method.sh"
