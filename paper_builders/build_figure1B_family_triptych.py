#!/usr/bin/env python3
from __future__ import annotations

"""Figure 1B (v3) — Aligned vs Gold curves, grouped by model family.

For each (method_base, task, family) produce ONE figure with 3 subplots (the 3 pairs).
Each subplot shows only:
  - Gold (direct target)
  - Aligned (chosen by --aligned-mode)

Aligned selection:
  --aligned-mode zero_shot (default): use train_spec = LOO-{task}
  --aligned-mode in_dist:             use train_spec = TRAIN-{task}
  --aligned-mode best:                choose best ctrl=none across all train_specs (max CPR by default)
  --aligned-mode fixed/near_dist:     use --train-spec

Annotations per subplot (upper-left):
  CPR_Gold and CPR_Aligned (CPR == area_under in pkls).

Axis labels:
  x = Proportion Node (k)
  y = Faithfulness (f)

Subplot titles:
  e.g. 'LLaMA-3 1B → 3B'
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_curve_utils_v2 import (
    BUDGET_PCTS, DEFAULT_TASKS, DEFAULT_METHOD_BASES,
    build_method_tag, choose_best_aligned, load_result, get_faithfulness_curve, curve_score, parse_list, apply_paper_style
)
from model_config import FAMILY_DISPLAY, FAMILY_PAIRS, pretty_model_size

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
    ax.grid(True, axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _model_size(m: str) -> str:
    return pretty_model_size(m)

def subplot_title(family: str, src: str, tgt: str) -> str:
    fam_name = FAMILY_DISPLAY[family]
    return f"{fam_name} {_model_size(src)} → {_model_size(tgt)}"

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

def main():
    apply_paper_style()
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="figures/figure1B_family_triptych")
    ap.add_argument("--split", default="test", choices=["train","validation","test"])
    ap.add_argument("--absolute", default="False", choices=["True","False"])
    ap.add_argument("--ablation", default="patching")
    ap.add_argument("--level", default="node")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--exclude-tasks", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("--family", default="all", choices=["all", *FAMILY_PAIRS.keys()])
    ap.add_argument("--aligned-mode", default="zero_shot", choices=["zero_shot","in_dist","best","fixed","near_dist"])
    ap.add_argument("--train-spec", default="")
    ap.add_argument("--select-by", default="area_under", choices=["area_under","average","faithfulness@100"])
    ap.add_argument("--ticks", default="sparse", choices=["full","sparse"])
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args()

    if args.aligned_mode == "near_dist":
        args.aligned_mode = "fixed"

    results_root = Path(args.results_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    split = args.split
    absolute = (args.absolute == "True")

    tasks = parse_list(args.tasks) if args.tasks else list(DEFAULT_TASKS)
    tasks = [t for t in tasks if t not in set(parse_list(args.exclude_tasks))]
    methods = parse_list(args.methods) if args.methods else list(DEFAULT_METHOD_BASES)

    families: List[Tuple[str, List[Tuple[str,str]]]] = []
    for family, pairs in FAMILY_PAIRS.items():
        if args.family in ("all", family):
            families.append((family, list(pairs)))

    xticks = list(BUDGET_PCTS) if args.ticks == "full" else [0.1, 0.5, 2, 10, 50, 100]

    gold_style = dict(color="black", linestyle="--", marker="o")
    aligned_style = dict(color="#d95f02", linestyle="-", marker="o")

    for method_base in methods:
        for task in tasks:
            for fam, pairs in families:
                fig, axes = plt.subplots(1, len(pairs), figsize=(3.75 * len(pairs), 3.4), sharey=True)
                axes = axes.ravel()
                for ax, (src, tgt) in zip(axes, pairs):
                    gold = load_result(results_root, method_base, args.ablation, args.level, task, tgt, split, absolute)
                    gold_y = get_faithfulness_curve(gold)
                    cpr_gold = curve_score(gold, prefer=args.select_by)

                    if args.aligned_mode == "best":
                        train_spec, _, aligned = choose_best_aligned(
                            results_root, method_base, src, tgt, args.ablation, args.level,
                            task, split, absolute, train_mode="best", fixed_train_spec="", prefer=args.select_by
                        )
                    else:
                        train_spec = aligned_mode_to_train_spec(args.aligned_mode, task, args.train_spec)
                        mt = build_method_tag(method_base, src, train_spec, "none")
                        aligned = load_result(results_root, mt, args.ablation, args.level, task, tgt, split, absolute)

                    aligned_y = get_faithfulness_curve(aligned)
                    cpr_aligned = curve_score(aligned, prefer=args.select_by)

                    ax.set_xscale("log")
                    ax.set_xticks(xticks)
                    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    ax.tick_params(axis="x", labelrotation=25)
                    _beautify(ax)

                    if gold_y is not None:
                        ax.plot(BUDGET_PCTS, gold_y, **gold_style)
                    if aligned_y is not None:
                        ax.plot(BUDGET_PCTS, aligned_y, **aligned_style)

                    ax.set_title(subplot_title(fam, src, tgt))
                    ax.set_xlabel("Proportion Node (k)")

                    cg = "?" if cpr_gold is None else f"{cpr_gold:.2f}"
                    ca = "?" if cpr_aligned is None else f"{cpr_aligned:.2f}"
                    ax.text(
                        0.03, 0.97,
                        f"CPR$_{{\\mathrm{{Gold}}}}$ = {cg}\nCPR$_{{\\mathrm{{Aligned}}}}$ = {ca}",
                        transform=ax.transAxes,
                        va="top", ha="left",
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85)
                    )

                axes[0].set_ylabel("Faithfulness (f)")

                # One simple legend
                handles = [
                    Line2D([0],[0], color="black", linestyle="--", lw=2),
                    Line2D([0],[0], color="#d95f02", linestyle="-", lw=2),
                ]
                fig.legend(handles, ["Gold", "Aligned"], loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.05))

                if not args.no_title:
                    fig.suptitle(f"{pretty_task(task)}: Faithfulness vs. Circuit Size", y=1.16)

                fig.tight_layout()
                out_prefix = out_dir / f"figure1B_{method_base}_{task}_{fam}_{args.aligned_mode}"
                fig.savefig(str(out_prefix.with_suffix(".pdf")), bbox_inches="tight")
                fig.savefig(str(out_prefix.with_suffix(".png")), dpi=220, bbox_inches="tight")
                plt.close(fig)

    print(f"✓ Wrote Figure 1B (v3) to: {out_dir}")

if __name__ == "__main__":
    main()
