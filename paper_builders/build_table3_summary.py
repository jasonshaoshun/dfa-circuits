from typing import Dict, List, Optional, Sequence, Tuple

from table_utils import (
    METHODS,
    METHOD_DISPLAY,
    MODEL_PAIRS,
    TASKS,
    build_method_rows,
    load_records,
    make_index,
    parse_args_base,
    style_text,
    write_csv,
    write_latex_lines,
    escape_latex,
    slugify,
)

# 3 control rows
CONTROL_ROWS: List[Tuple[str, str, str]] = [
    ("1. Random $W$ (Lower Bound)", "random", "Is task hard?"),
    ("2. Scrambled input $s$", "scrambled", "Is input used?"),
    ("3. Permuted $W$ columns", "permuted", "Is topology real?"),
    ("4. Heuristic Depth Mean", "heuristic_depth_mean", "Is proportional depth meaningful?")
]


def elemwise_max(*rows: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Elementwise max across rows (ignoring None)."""
    if not rows:
        return []
    n = len(rows[0])
    out: List[Optional[float]] = []
    for i in range(n):
        vals = [r[i] for r in rows if i < len(r) and r[i] is not None]
        out.append(max(vals) if vals else None)
    return out


def tie_aware_style_column(values: Sequence[Optional[float]]) -> List[str]:
    """
    Tie-aware styling for one column across rows:
      - Bold all rows tied for the maximum value.
      - Underline all rows tied for the 2nd DISTINCT value (strictly less than max).
    If there is no 2nd distinct value, underline none.
    """
    # Collect distinct numeric values
    numeric = [v for v in values if v is not None]
    if not numeric:
        return ["-" for _ in values]

    best_val = max(numeric)
    # Find 2nd distinct value (< best_val)
    below_best = [v for v in numeric if v < best_val]
    second_val = max(below_best) if below_best else None

    out: List[str] = []
    for v in values:
        if v is None:
            out.append("-")
            continue
        txt = f"{v:.2f}"
        is_best = (v == best_val)
        is_second = (second_val is not None and v == second_val)
        out.append(style_text(txt, bold=is_best, underline=is_second))
    return out


def main() -> None:
    args = parse_args_base("Build Table 3 summary tables (ablations + alignment).")
    records = load_records(args.results_dir, args.metric_key)
    idx = make_index(records, args.split, args.ablation, args.level, args.absolute)

    task_labels = [lbl for _t, lbl in TASKS]  # 6 tasks (matches detail)

    for source_model, target_model in MODEL_PAIRS:
        per_method: Dict[str, Dict[str, List[Optional[float]]]] = {
            method: build_method_rows(idx, method, source_model, target_model)
            for method in METHODS
        }

        latex_lines: List[str] = [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\resizebox{\textwidth}{!}{",
            r"\begin{tabular}{llcccccc}",
            r"\toprule",
            r"\textbf{Method} & \textbf{Setting / Control} & "
            + " & ".join([rf"\textbf{{{escape_latex(x)}}}" for x in task_labels])
            + r" \\",
            r"\midrule",
        ]

        # csv_header = ["Method", "Setting / Control"] + task_labels + ["Hypothesis Tested"]
        csv_header = ["Method", "Setting / Control"] + task_labels
        csv_rows: List[List[str]] = []

        for m_i, method in enumerate(METHODS):
            display_name = METHOD_DISPLAY.get(method, method)
            rows = per_method[method]

            # Ours variants
            in_dist = rows.get("in", [None] * len(TASKS))
            near = rows.get("near", [None] * len(TASKS))
            far = rows.get("far", [None] * len(TASKS))   # Alignment (Zero-shot)
            best = elemwise_max(in_dist, near, far)       # Alignment (Best)

            # 6 rows in the table, in order
            display_rows: List[Tuple[str, List[Optional[float]], str]] = [
                (CONTROL_ROWS[0][0], rows["random"], CONTROL_ROWS[0][2]),
                (CONTROL_ROWS[1][0], rows["scrambled"], CONTROL_ROWS[1][2]),
                (CONTROL_ROWS[2][0], rows["permuted"], CONTROL_ROWS[2][2]),
                (CONTROL_ROWS[3][0], rows["heuristic_depth_mean"], CONTROL_ROWS[3][2]),
                (r"\textbf{DFA (Zero-shot)}", far, "-"),
                (r"\textbf{DFA (Best)}", best, "-"),
            ]

            # Build tie-aware formatted matrix: 5 x 6
            formatted: List[List[str]] = [["" for _ in TASKS] for _ in range(len(display_rows))]
            for col_idx in range(len(TASKS)):
                col_vals = [rv[col_idx] for (_lab, rv, _hyp) in display_rows]
                col_fmt = tie_aware_style_column(col_vals)
                for r_idx in range(len(display_rows)):
                    formatted[r_idx][col_idx] = col_fmt[r_idx]

            # Emit first 3 (controls)
            for r_idx in range(4):
                row_label, row_vals, hyp = display_rows[r_idx]

                csv_rows.append(
                    [display_name, row_label]
                    + ["" if v is None else f"{v:.2f}" for v in row_vals[: len(TASKS)]]
                )

                if r_idx == 0:
                    latex_lines.append(
                        rf"\multirow{{6}}{{*}}{{{escape_latex(display_name)}}} & {row_label} & "
                        + " & ".join(formatted[r_idx])
                        + r" \\"
                    )
                else:
                    latex_lines.append(
                        "& " + row_label + " & " + " & ".join(formatted[r_idx]) + r" \\"
                    )

            # Separator before ours rows
            latex_lines.append(r"\addlinespace[4pt]")

            # Emit alignment rows
            for r_idx in [4, 5]:
                row_label, row_vals, _hyp = display_rows[r_idx]

                # CSV label (strip \textbf)
                csv_label = row_label.replace(r"\textbf{", "").replace("}", "")
                csv_rows.append(
                    [display_name, csv_label]
                    + ["" if v is None else f"{v:.2f}" for v in row_vals[: len(TASKS)]]
                )

                latex_lines.append(
                    "& " + row_label + " & " + " & ".join(formatted[r_idx]) + r" \\"
                )

            if m_i != len(METHODS) - 1:
                latex_lines.append(r"\midrule")

        latex_lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\caption{{Validation and Ablation Study for target model {escape_latex(target_model)} (source {escape_latex(source_model)}). We report the zero-shot faithfulness drop under various structural corruptions using CPR.}}",
            rf"\label{{tab:ablations-summary-{slugify(source_model + '-to-' + target_model)}}}",
            r"\end{table*}",
        ]

        stem = f"table3_summary_{source_model}_to_{target_model}"
        out_tex = f"{args.output_dir}/{stem}.tex"
        out_csv = f"{args.output_dir}/{stem}.csv"
        write_latex_lines(out_tex, latex_lines)
        write_csv(out_csv, csv_header, csv_rows)


if __name__ == "__main__":
    main()