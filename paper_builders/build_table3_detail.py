
from typing import Dict, List, Optional

from table_utils import (
    METHODS, METHOD_DISPLAY, MODEL_PAIRS, TASKS,
    build_method_rows, load_records, make_index, parse_args_base,
    pair_display, row_average, rank_indices, style_text,
    write_csv, write_latex_lines, escape_latex, slugify
)

ROW_DEFS = [
    ("Random Alignment", "random"),
    ("Scrambled Input $s$", "scrambled"),
    ("Permuted $W$ Columns", "permuted"),
    ("Heuristic Depth-Mean", "heuristic_depth_mean"),
    ("Gold (Upper Bound)", "gold"),
    ("DFA (In-Distribution)", "in"),
    ("DFA (Near-Distribution)", "near"),
    ("DFA (Far/Zero-Shot)", "far"),
]


def main() -> None:
    args = parse_args_base("Build Table 3 detail tables.")
    records = load_records(args.results_dir, args.metric_key)
    idx = make_index(records, args.split, args.ablation, args.level, args.absolute)

    for source_model, target_model in MODEL_PAIRS:
        for method in METHODS:
            rows = build_method_rows(idx, method, source_model, target_model)

            # ranking across all 7 rows per numeric column incl average
            col_marks: Dict[int, tuple] = {}
            for col_idx in range(len(TASKS) + 1):
                vals: List[Optional[float]] = []
                for _label, row_key in ROW_DEFS:
                    base_vals = rows[row_key]
                    vals.append(row_average(base_vals) if col_idx == len(TASKS) else base_vals[col_idx])
                col_marks[col_idx] = rank_indices(vals)

            csv_rows: List[List[str]] = []
            latex_lines: List[str] = [
                r"\begin{table*}[!t]",
                r"\centering",
                r"\resizebox{\textwidth}{!}{",
                r"\begin{tabular}{llccccccc}",
                r"\toprule",
                r"\textbf{Method Category} & \textbf{Method Name} & \textbf{IOI} & \textbf{MCQA} & \textbf{Arith +} & \textbf{Arith -} & \textbf{ARC-E} & \textbf{ARC-C} & \textbf{Average} \\",
                r"\midrule",
            ]

            display = METHOD_DISPLAY[method]
            for row_idx, (row_label, row_key) in enumerate(ROW_DEFS):
                raw_vals = rows[row_key]
                avg = row_average(raw_vals)
                cell_texts = []
                for col_idx in range(len(TASKS) + 1):
                    val = avg if col_idx == len(TASKS) else raw_vals[col_idx]
                    best_idx, second_idx = col_marks[col_idx]
                    text = "-" if val is None else f"{val:.2f}"
                    if val is not None:
                        text = style_text(text, bold=(best_idx == row_idx), underline=(second_idx == row_idx))
                    cell_texts.append(text)
                csv_rows.append([display, row_label] + [f"{v:.2f}" if v is not None else "" for v in raw_vals] + [f"{avg:.2f}" if avg is not None else ""])
                if row_idx in (4, 5):
                    latex_lines.append(r"\midrule")
                latex_lines.append(
                    escape_latex(display) + " & " + row_label + " & " + " & ".join(cell_texts) + r" \\"
                )

            latex_lines += [
                r"\bottomrule",
                r"\end{tabular}",
                r"}",
                rf"\caption{{Validation and ablation study for {escape_latex(METHOD_DISPLAY[method])}, target model {escape_latex(target_model)} (source {escape_latex(source_model)}; metric: {escape_latex(args.metric_key)}).}}",
                rf"\label{{tab:ablation-detail-{slugify(method + '-' + source_model + '-to-' + target_model)}}}",
                r"\end{table*}",
            ]

            stem = f"table3_detail_{method}_{source_model}_to_{target_model}"
            write_csv(f"{args.output_dir}/{stem}.csv", ["Method Category", "Method Name"] + [label for _, label in TASKS] + ["Average"], csv_rows)
            write_latex_lines(f"{args.output_dir}/{stem}.tex", latex_lines)


if __name__ == "__main__":
    main()
