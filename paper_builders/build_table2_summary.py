
#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from typing import List
from table_utils2 import (
    TASKS, MODEL_PAIRS, METHODS, tex_escape, fmt_num, fmt_pct,
    load_records, build_index, get_baseline, best_aligned, get_gold, recovery_pct, write_lines
)

def build_task_table(task: str, task_name: str, idx, metric: str, out_tex: str, out_csv: str) -> None:
    # CSV (flattened)
    csv_lines: List[str] = []
    csv_lines.append("Method,Target Model Pair,Baseline,DFA (Best),Gold,Faithfulness Recovery Ratio (%)")
    for method_base, method_disp in METHODS:
        for src, tgt, pair_disp in MODEL_PAIRS:
            base = get_baseline(idx, method_base, task, src, tgt)
            ours = best_aligned(idx, method_base, task, src, tgt)
            gold = get_gold(idx, method_base, task, tgt)
            rec = recovery_pct(ours, gold)
            csv_lines.append(",".join([
                method_disp,
                pair_disp,
                "" if base is None else f"{base:.6f}",
                "" if ours is None else f"{ours:.6f}",
                "" if gold is None else f"{gold:.6f}",
                "" if rec is None else f"{rec:.3f}",
            ]))
    write_lines(out_csv, csv_lines)

    # LaTeX
    lines: List[str] = []
    lines.append("\\begin{table*}[!t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Target Model Pair} & \\textbf{Baseline} & \\textbf{DFA (Best)} & \\textbf{Gold} & \\textbf{Faithfulness Recovery Ratio} \\\\")
    lines.append("\\midrule")

    for m_i, (method_base, method_disp) in enumerate(METHODS):
        # multirow spanning 6 pairs
        for i, (src, tgt, pair_disp) in enumerate(MODEL_PAIRS):
            base = get_baseline(idx, method_base, task, src, tgt)
            ours = best_aligned(idx, method_base, task, src, tgt)
            gold = get_gold(idx, method_base, task, tgt)
            rec = recovery_pct(ours, gold)

            if i == 0:
                left = f"\\multirow{{{len(MODEL_PAIRS)}}}{{*}}{{{method_disp}}}"
                lines.append(f"{left} & {pair_disp} & {fmt_num(base)} & {fmt_num(ours)} & {fmt_num(gold)} & {fmt_pct(rec)} \\\\")
            else:
                lines.append(f"& {pair_disp} & {fmt_num(base)} & {fmt_num(ours)} & {fmt_num(gold)} & {fmt_pct(rec)} \\\\")
        if m_i != len(METHODS) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        f"\\caption{{Model scaling summary for {task_name}. Best aligned performance for this task using {tex_escape(metric)}. "
        f"Recovery\\% $= \\frac{{\\text{{Aligned(best)}}}}{{\\text{{Gold}}}}\\times 100$.}}"
    )
    lines.append(f"\\label{{tab:scaling-summary-{task.replace('_','-')}}}")
    lines.append("\\end{table*}")
    write_lines(out_tex, lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--output-dir", default="tables")
    ap.add_argument("--metric", default="area_under")
    ap.add_argument("--split", default="test")
    ap.add_argument("--absolute", action="store_true", default=False)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = load_records(args.results_dir, args.metric)
    idx = build_index(records, split=args.split, absolute=args.absolute)

    for task, task_name in TASKS:
        out_tex = os.path.join(args.output_dir, f"table2_summary_{task}.tex")
        out_csv = os.path.join(args.output_dir, f"table2_summary_{task}.csv")
        build_task_table(task, task_name, idx, args.metric, out_tex, out_csv)

if __name__ == "__main__":
    main()
