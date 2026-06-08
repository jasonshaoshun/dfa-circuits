
#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from typing import List
from table_utils2 import (
    TASKS, MODEL_PAIRS, METHODS, tex_escape, fmt_num, fmt_pct,
    load_records, build_index, get_baseline, best_aligned, get_gold, recovery_pct, write_lines
)

def build_one_table(task: str, task_name: str, method_base: str, method_disp: str,
                    idx, metric: str, out_tex: str, out_csv: str) -> None:
    # CSV
    csv_lines: List[str] = []
    csv_lines.append("Target Model Pair,Baseline,DFA (Best),Gold,Faithfulness Recovery Ratio (%)")
    for src, tgt, pair_disp in MODEL_PAIRS:
        base = get_baseline(idx, method_base, task, src, tgt)
        ours = best_aligned(idx, method_base, task, src, tgt)
        gold = get_gold(idx, method_base, task, tgt)
        rec = recovery_pct(ours, gold)
        csv_lines.append(",".join([
            pair_disp,
            "" if base is None else f"{base:.6f}",
            "" if ours is None else f"{ours:.6f}",
            "" if gold is None else f"{gold:.6f}",
            "" if rec is None else f"{rec:.3f}",
        ]))
    write_lines(out_csv, csv_lines)

    # LaTeX
    lines: List[str] = []
    lines.append("\\begin{table}[!t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Target Model Pair} & \\textbf{Baseline} & \\textbf{DFA (Best)} & \\textbf{Gold} & \\textbf{Recovery\\%} \\\\")
    lines.append("\\midrule")
    for src, tgt, pair_disp in MODEL_PAIRS:
        base = get_baseline(idx, method_base, task, src, tgt)
        ours = best_aligned(idx, method_base, task, src, tgt)
        gold = get_gold(idx, method_base, task, tgt)
        rec = recovery_pct(ours, gold)
        lines.append(f"{pair_disp} & {fmt_num(base)} & {fmt_num(ours)} & {fmt_num(gold)} & {fmt_pct(rec)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        f"\\caption{{Model scaling detail for {task_name} using {method_disp} (metric: {tex_escape(metric)}). "
        f"Recovery\\% $= \\frac{{\\text{{Aligned(best)}}}}{{\\text{{Gold}}}}\\times 100$.}}"
    )
    label = f"tab:scaling-detail-{task.replace('_','-')}-{method_base.lower().replace('_','-')}"
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
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

    # 6 tasks x 3 methods = 18 tables
    for task, task_name in TASKS:
        for method_base, method_disp in METHODS:
            out_tex = os.path.join(args.output_dir, f"table2_detail_{task}_{method_base}.tex")
            out_csv = os.path.join(args.output_dir, f"table2_detail_{task}_{method_base}.csv")
            build_one_table(task, task_name, method_base, method_disp, idx, args.metric, out_tex, out_csv)

if __name__ == "__main__":
    main()
