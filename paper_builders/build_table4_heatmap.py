
from typing import Dict, List, Optional

from table_utils import (
    METHODS, METHOD_DISPLAY, MODEL_PAIRS, TASKS,
    build_method_rows, load_records, make_index, parse_args_base,
    row_average, rank_indices, style_text, SHORT_TASK_LABEL,
    write_csv, write_latex_lines, escape_latex, slugify
)


def main() -> None:
    args = parse_args_base("Build Table 4 heatmap tables.")
    records = load_records(args.results_dir, args.metric_key)
    idx = make_index(records, args.split, args.ablation, args.level, args.absolute)

    task_names = [t for t, _ in TASKS]
    task_labels = [label for _, label in TASKS]

    for source_model, target_model in MODEL_PAIRS:
        for method in METHODS:
            # matrix[row train task][col eval task]
            matrix: List[List[Optional[float]]] = []
            for train_task in task_names:
                row: List[Optional[float]] = []
                for eval_task in task_names:
                    row.append(
                        idx.get((method, source_model, target_model, "diffalign", "none", f"TRAIN-{train_task}", eval_task, ""))
                    )
                matrix.append(row)

            avg_col = [row_average(row) for row in matrix]

            col_marks: Dict[int, tuple] = {}
            for col_idx in range(len(task_names) + 1):
                vals = avg_col if col_idx == len(task_names) else [matrix[r][col_idx] for r in range(len(task_names))]
                col_marks[col_idx] = rank_indices(vals)

            csv_rows: List[List[str]] = []
            latex_lines: List[str] = [
                r"\begin{table*}[!t]",
                r"\centering",
                r"\resizebox{\textwidth}{!}{",
                r"\begin{tabular}{lccccccc}",
                r"\toprule",
                "Training Source & Test: IOI & Test: MCQA & Test: Arith + & Test: Arith - & Test: ARC-E & Test: ARC-C & Average " + r"\\",
                r"\midrule",
            ]

            for r_idx, train_task in enumerate(task_names):
                numeric_cells: List[str] = []
                raw_vals = matrix[r_idx]
                for col_idx, val in enumerate(raw_vals):
                    best_idx, second_idx = col_marks[col_idx]
                    text = "-" if val is None else f"{val:.2f}"
                    if val is not None:
                        text = style_text(
                            text,
                            bold=(best_idx == r_idx),
                            underline=(second_idx == r_idx),
                            italic=(r_idx == col_idx),
                        )
                    numeric_cells.append(text)
                avg = avg_col[r_idx]
                best_idx, second_idx = col_marks[len(task_names)]
                avg_text = "-" if avg is None else f"{avg:.2f}"
                if avg is not None:
                    avg_text = style_text(avg_text, bold=(best_idx == r_idx), underline=(second_idx == r_idx))
                latex_lines.append(f"Train: {SHORT_TASK_LABEL[train_task]} & " + " & ".join(numeric_cells + [avg_text]) + r" \\")
                csv_rows.append([f"Train: {SHORT_TASK_LABEL[train_task]}"] + [f"{v:.2f}" if v is not None else "" for v in raw_vals] + [f"{avg:.2f}" if avg is not None else ""])

            latex_lines += [
                r"\bottomrule",
                r"\end{tabular}",
                r"}",
                rf"\caption{{Global Alignment Matrix (Task-to-Task Transfer). Faithfulness scores for {escape_latex(METHOD_DISPLAY[method])} on {escape_latex(source_model)} $\to$ {escape_latex(target_model)} using {escape_latex(args.metric_key)}. The diagonal represents in-distribution performance.}}",
                rf"\label{{tab:heatmap-{slugify(method + '-' + source_model + '-to-' + target_model)}}}",
                r"\end{table*}",
            ]

            stem = f"table4_heatmap_{method}_{source_model}_to_{target_model}"
            write_csv(f"{args.output_dir}/{stem}.csv", ["Training Source"] + [f"Test: {label}" for label in task_labels] + ["Average"], csv_rows)
            write_latex_lines(f"{args.output_dir}/{stem}.tex", latex_lines)


if __name__ == "__main__":
    main()
