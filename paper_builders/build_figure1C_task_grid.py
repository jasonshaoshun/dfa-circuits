#!/usr/bin/env python3
from __future__ import annotations

"""Figure 1C (v4) — {mode} Aligned vs. Gold Circuits Across Tasks.

Fixes requested:
- Removes the extra top 'method/src→tgt' mark that was appearing behind the legend.
- For Qwen-family pairs, drop arithmetic_addition by default (use --keep-qwen-addition to override).
- Legend encodes style only: Gold dashed black, Aligned solid.
- If fewer than 6 tasks remain, unused subplots are blank (axes off).
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_curve_utils_v2 import (
    BUDGET_PCTS, DEFAULT_TASKS, DEFAULT_METHOD_BASES, DEFAULT_MODEL_PAIRS,
    build_method_tag, choose_best_aligned, load_result, get_faithfulness_curve, curve_score, parse_list, apply_paper_style
)

TASK_PRETTY = {
    "ioi": "IOI",
    "mcqa": "MCQA",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "arithmetic_addition": "Addition",
    "arithmetic_subtraction": "Subtraction",
}
def pretty_task(t: str) -> str:
    return TASK_PRETTY.get(t, t.replace("_", "-").title())

TASK_COLORS = {
    "ioi": "#1b9e77",
    "mcqa": "#d95f02",
    "arithmetic_addition": "#7570b3",
    "arithmetic_subtraction": "#e7298a",
    "arc_easy": "#66a61e",
    "arc_challenge": "#e6ab02",
}

def _beautify(ax: plt.Axes):
    ax.grid(True, axis="y", alpha=0.20)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def aligned_mode_to_train_spec(mode: str, task: str, fixed: str) -> str:
    if mode == "zero_shot":
        return f"LOO-{task}"
    if mode == "in_dist":
        return f"TRAIN-{task}"
    if mode == "fixed":
        if not fixed:
            raise ValueError("--train-spec required for aligned-mode=fixed/near_dist")
        return fixed
    raise ValueError("aligned_mode_to_train_spec called for unsupported mode")

def mode_label(mode: str) -> str:
    return {
        "zero_shot": "Zero-shot",
        "in_dist": "In-distribution",
        "best": "Best",
        "fixed": "Fixed",
    }.get(mode, mode)

def is_qwen_pair(src: str, tgt: str) -> bool:
    return src.startswith("qwen") or tgt.startswith("qwen")


def safe_id(s: str) -> str:
    """Filename-safe identifier for model strings like 'qwen2.5-1.5b'."""
    s = s.replace(".", "p")
    s = s.replace("/", "-").replace(" ", "_")
    return s

def main():
    apply_paper_style()
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="figures/figure1C_task_grid")
    ap.add_argument("--split", default="test", choices=["train","validation","test"])
    ap.add_argument("--absolute", default="False", choices=["True","False"])
    ap.add_argument("--ablation", default="patching")
    ap.add_argument("--level", default="node")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--exclude-tasks", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--aligned-mode", default="zero_shot", choices=["zero_shot","in_dist","best","fixed","near_dist"])
    ap.add_argument("--train-spec", default="")
    ap.add_argument("--select-by", default="area_under", choices=["area_under","average","faithfulness@100"])
    ap.add_argument("--ticks", default="sparse", choices=["full","sparse"])
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--keep-qwen-addition", action="store_true")
    args = ap.parse_args()

    if args.aligned_mode == "near_dist":
        args.aligned_mode = "fixed"

    results_root = Path(args.results_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    split = args.split
    absolute = (args.absolute == "True")

    base_tasks = parse_list(args.tasks) if args.tasks else list(DEFAULT_TASKS)
    base_tasks = [t for t in base_tasks if t not in set(parse_list(args.exclude_tasks))]

    methods = parse_list(args.methods) if args.methods else list(DEFAULT_METHOD_BASES)

    pairs: List[Tuple[str,str]] = []
    if args.pairs:
        for s in parse_list(args.pairs):
            if ":" not in s:
                raise ValueError(f"Bad pair '{s}' (expected src:tgt)")
            a,b = s.split(":",1)
            pairs.append((a,b))
    else:
        pairs = list(DEFAULT_MODEL_PAIRS)

    xticks = list(BUDGET_PCTS) if args.ticks == "full" else [0.1, 0.5, 2, 10, 50, 100]
    gold_style = dict(color="black", linestyle="--", marker="o")

    for method_base in methods:
        for src, tgt in pairs:
            tasks = list(base_tasks)
            if (not args.keep_qwen_addition) and is_qwen_pair(src, tgt):
                tasks = [t for t in tasks if t != "arithmetic_addition"]
            tasks = tasks[:6]

            fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.8), sharex=True, sharey=True)
            axes = axes.flatten()

            for i in range(6):
                ax = axes[i]
                if i >= len(tasks):
                    ax.axis("off")
                    continue

                task = tasks[i]
                ax.set_xscale("log")
                ax.set_xticks(xticks)
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.tick_params(axis="x", labelrotation=25)
                _beautify(ax)

                gold = load_result(results_root, method_base, args.ablation, args.level, task, tgt, split, absolute)
                gold_y = get_faithfulness_curve(gold)
                cpr_gold = curve_score(gold, prefer=args.select_by)

                if args.aligned_mode == "best":
                    _, _, aligned = choose_best_aligned(
                        results_root, method_base, src, tgt, args.ablation, args.level,
                        task, split, absolute, train_mode="best", fixed_train_spec="", prefer=args.select_by
                    )
                else:
                    train_spec = aligned_mode_to_train_spec(args.aligned_mode, task, args.train_spec)
                    mt = build_method_tag(method_base, src, train_spec, "none")
                    aligned = load_result(results_root, mt, args.ablation, args.level, task, tgt, split, absolute)

                aligned_y = get_faithfulness_curve(aligned)
                cpr_aligned = curve_score(aligned, prefer=args.select_by)

                aligned_color = TASK_COLORS.get(task, "#d95f02")
                aligned_style = dict(color=aligned_color, linestyle="-", marker="o")

                if gold_y is not None:
                    ax.plot(BUDGET_PCTS, gold_y, **gold_style)
                if aligned_y is not None:
                    ax.plot(BUDGET_PCTS, aligned_y, **aligned_style)

                ax.set_title(pretty_task(task))

                cg = "?" if cpr_gold is None else f"{cpr_gold:.2f}"
                ca = "?" if cpr_aligned is None else f"{cpr_aligned:.2f}"
                ax.text(
                    0.03, 0.97,
                    f"CPR$_{{\\mathrm{{Gold}}}}$ = {cg}\nCPR$_{{\\mathrm{{Aligned}}}}$ = {ca}",
                    transform=ax.transAxes,
                    va="top", ha="left",
                    fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85)
                )

            for ax in axes[::3]:
                if ax.has_data():
                    ax.set_ylabel("Faithfulness (f)")
            for ax in axes[-3:]:
                if ax.has_data():
                    ax.set_xlabel("Proportion Node (k)")

            legend_handles = [
                Line2D([0],[0], color="black", linestyle="--", lw=2),
                Line2D([0],[0], color="#444444", linestyle="-", lw=2),
            ]
            fig.legend(legend_handles, ["Gold", "Aligned"], loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.04))

            if not args.no_title:
                fig.suptitle(f"{mode_label(args.aligned_mode)} Aligned vs. Gold Circuits Across Tasks", y=1.06)

            fig.tight_layout()
            out_prefix = out_dir / f"figure1C_{method_base}_{safe_id(src)}_to_{safe_id(tgt)}_{args.aligned_mode}"
            fig.savefig(str(out_prefix.with_suffix(".pdf")), bbox_inches="tight")
            fig.savefig(str(out_prefix.with_suffix(".png")), dpi=220, bbox_inches="tight")
            plt.close(fig)

    print(f"✓ Wrote Figure 1C (v4) to: {out_dir}")

if __name__ == "__main__":
    main()
