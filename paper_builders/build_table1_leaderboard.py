from typing import Dict, List, Optional

from table_utils import (
    METHODS, METHOD_DISPLAY, MODEL_PAIRS, TASKS,
    build_method_rows, make_index, load_records, parse_args_base,
    pair_display, row_average, rank_indices,
    write_csv, write_latex_lines, escape_latex, slugify, style_text,
)

BASELINE_METHOD = "EAP-IG-inputs"
ROW_ORDER = [
    ("gold", "Gold (Upper Bound)"),
    ("in", "DFA (In-Distribution)"),
    ("near", "DFA (Near-Distribution)"),
    ("far", "DFA (Far/Zero-Shot)"),
]


def main() -> None:
    args = parse_args_base("Build Table 1 leaderboard tables.")
    records = load_records(args.results_dir, args.metric_key)
    idx = make_index(records, args.split, args.ablation, args.level, args.absolute)

    for source_model, target_model in MODEL_PAIRS:
        method_rows: Dict[str, Dict[str, List[Optional[float]]]] = {
            method: build_method_rows(idx, method, source_model, target_model)
            for method in METHODS
        }
        baseline_vals = method_rows[BASELINE_METHOD]["random"]
        baseline_avg = row_average(baseline_vals)

        # Highlight within each METHOD block, across the 4 transfer-setting rows.
        # For each method and each column, compare [gold, in, near, far].
        marks: Dict[str, Dict[int, tuple]] = {}
        for method in METHODS:
            marks[method] = {}
            for col_idx in range(len(TASKS) + 1):
                vals: List[Optional[float]] = []
                for row_key, _label in ROW_ORDER:
                    task_vals = method_rows[method][row_key]
                    val = row_average(task_vals) if col_idx == len(TASKS) else task_vals[col_idx]
                    vals.append(val)
                marks[method][col_idx] = rank_indices(vals)

        csv_rows: List[List[str]] = []
        latex_lines: List[str] = [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\resizebox{\textwidth}{!}{",
            r"\begin{tabular}{llccccccc}",
            r"\toprule",
            r"\textbf{Method Category} & \textbf{Transfer Setting} & \textbf{IOI} & \textbf{MCQA} & \textbf{Arith +} & \textbf{Arith -} & \textbf{ARC-E} & \textbf{ARC-C} & \textbf{Average} \\",
            r"\midrule",
            "Baseline & Random Alignment & " + " & ".join([f"{v:.2f}" if v is not None else "-" for v in baseline_vals] + [f"{baseline_avg:.2f}" if baseline_avg is not None else "-"]) + r" \\",
            r"\midrule",
        ]
        csv_rows.append(["Baseline", "Random Alignment"] + [f"{v:.2f}" if v is not None else "" for v in baseline_vals] + [f"{baseline_avg:.2f}" if baseline_avg is not None else ""])

        for m_idx, method in enumerate(METHODS):
            display = METHOD_DISPLAY[method]
            for r_idx, (row_key, row_label) in enumerate(ROW_ORDER):
                raw_vals = list(method_rows[method][row_key])
                avg = row_average(raw_vals)
                row_index = r_idx  # 0=gold,1=in,2=near,3=far
                cell_texts: List[str] = []
                for col_idx in range(len(TASKS) + 1):
                    val = avg if col_idx == len(TASKS) else raw_vals[col_idx]
                    best_idx, second_idx = marks[method][col_idx]
                    text = "-" if val is None else f"{val:.2f}"
                    if val is not None:
                        text = style_text(text, bold=(best_idx == row_index), underline=(second_idx == row_index))
                    cell_texts.append(text)
                csv_rows.append([display, row_label] + [f"{v:.2f}" if v is not None else "" for v in raw_vals] + [f"{avg:.2f}" if avg is not None else ""])
                if r_idx == 0:
                    latex_lines.append(rf"\multirow{{4}}{{*}}{{{escape_latex(display)}}} & {escape_latex(row_label)} & " + " & ".join(cell_texts) + r" \\")
                else:
                    latex_lines.append("& " + escape_latex(row_label) + " & " + " & ".join(cell_texts) + r" \\")
            if m_idx != len(METHODS) - 1:
                latex_lines.append(r"\midrule")

        latex_lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\caption{{Circuit Recovery Performance ({escape_latex(pair_display(source_model, target_model))}). We report the faithfulness score using {escape_latex(args.metric_key)}. Near-distribution evaluation is only applicable to Arithmetic and ARC task pairs.}}",
            rf"\label{{tab:leaderboard-{slugify(source_model + '-to-' + target_model)}}}",
            r"\end{table*}",
        ]

        stem = f"table1_leaderboard_{source_model}_to_{target_model}"
        write_csv(f"{args.output_dir}/{stem}.csv", ["Method Category", "Transfer Setting"] + [label for _, label in TASKS] + ["Average"], csv_rows)
        write_latex_lines(f"{args.output_dir}/{stem}.tex", latex_lines)


if __name__ == "__main__":
    main()
