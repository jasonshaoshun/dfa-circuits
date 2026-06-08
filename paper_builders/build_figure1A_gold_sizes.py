#!/usr/bin/env python3
from __future__ import annotations

"""Figure 1A (v4) — GOLD faithfulness-vs-budget curves across model sizes.

Design:
- One panel per model family.
- Same style encoding for {small, medium, large} across panels.
- One legend for size encoding: 'Scale within family'
- Axis labels: x = Proportion Node (k), y = Faithfulness (f)
- Title format: '<PrettyTask>: Faithfulness vs. Circuit Size'

Data source: gold evaluation outputs
  results/{method_base}_{ablation}_{level}/{task-with-dashes}_{model}_{split}_abs-{absolute}.pkl
"""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_curve_utils_v2 import (
    BUDGET_PCTS, DEFAULT_TASKS, DEFAULT_METHOD_BASES,
    load_result, get_faithfulness_curve, parse_list, apply_paper_style
)
from model_config import FAMILY_DISPLAY, FAMILY_MODEL_BY_SIZE, pretty_model_size

# More visible purples (large not too light)
SIZE_STYLE = [
    dict(label="Small",  color="#54278f", linestyle="-"),   # dark purple
    dict(label="Medium", color="#756bb1", linestyle="--"),  # medium purple
    dict(label="Large",  color="#9e9ac8", linestyle=":"),   # light-but-visible purple
]

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

def _beautify(ax: plt.Axes):
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _size_label(model: str) -> str:
    return pretty_model_size(model)

def main():
    apply_paper_style()
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="figures/figure1A_gold_sizes")
    ap.add_argument("--split", default="test", choices=["train","validation","test"])
    ap.add_argument("--absolute", default="False", choices=["True","False"])
    ap.add_argument("--ablation", default="patching")
    ap.add_argument("--level", default="node")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--exclude-tasks", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("--ticks", default="sparse", choices=["full","sparse"])
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    split = args.split
    absolute = (args.absolute == "True")

    tasks = parse_list(args.tasks) if args.tasks else list(DEFAULT_TASKS)
    tasks = [t for t in tasks if t not in set(parse_list(args.exclude_tasks))]
    methods = parse_list(args.methods) if args.methods else list(DEFAULT_METHOD_BASES)

    xticks = list(BUDGET_PCTS) if args.ticks == "full" else [0.1, 0.5, 2, 10, 50, 100]

    for method_base in methods:
        for task in tasks:
            families = list(FAMILY_MODEL_BY_SIZE.items())
            fig, axes = plt.subplots(1, len(families), figsize=(4.8 * len(families), 3.4), sharey=True)
            axes = axes.ravel()

            for ax, (family, model_by_size) in zip(axes, families):
                missing = []
                for model, st in zip(model_by_size.values(), SIZE_STYLE):
                    d = load_result(results_root, method_base, args.ablation, args.level, task, model, split, absolute)
                    y = get_faithfulness_curve(d)
                    if y is None:
                        missing.append(model)
                        continue
                    ax.plot(BUDGET_PCTS, y, color=st["color"], linestyle=st["linestyle"], marker="o")
                ax.set_xscale("log")
                ax.set_xticks(xticks)
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.tick_params(axis="x", labelrotation=25)
                ax.set_xlabel("Proportion Node (k)")
                ax.set_title(FAMILY_DISPLAY[family])
                _beautify(ax)
                if missing:
                    ax.text(0.01, 0.02, "missing: " + ", ".join([_size_label(m) for m in missing]),
                            transform=ax.transAxes, fontsize=9, alpha=0.75)

            axes[0].set_ylabel("Faithfulness (f)")

            handles = [Line2D([0],[0], color=st["color"], linestyle=st["linestyle"], lw=2) for st in SIZE_STYLE]
            labels = [st["label"] for st in SIZE_STYLE]
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True,
                       bbox_to_anchor=(0.5, 1.04), title="Scale within family")

            if not args.no_title:
                fig.suptitle(f"{pretty_task(task)}: Faithfulness vs. Circuit Size", y=1.14)

            fig.tight_layout()
            out_prefix = out_dir / f"figure1A_{method_base}_{args.ablation}_{args.level}_{task}"
            fig.savefig(str(out_prefix.with_suffix(".pdf")), bbox_inches="tight")
            fig.savefig(str(out_prefix.with_suffix(".png")), dpi=220, bbox_inches="tight")
            plt.close(fig)

    print(f"✓ Wrote Figure 1A (v4) to: {out_dir}")

if __name__ == "__main__":
    main()
