#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figure_curve_utils_v2 import (
    DEFAULT_TASKS,
    curve_score,
    choose_best_aligned,
)

from table_utils import load_records, make_index, query_value
from model_config import (
    FAMILY_DISPLAY,
    FAMILY_ORDER,
    family_for_pair_display,
    model_by_size,
)

FAMILY_ORDER_ITEMS = tuple(FAMILY_ORDER.items())

# CSV uses display names; results dirs use method_base
METHOD_DISPLAY_TO_BASE = {
    "EAP": "EAP",
    "EAP-IG-Inputs": "EAP-IG-inputs",
    "EAP-IG-Activations": "EAP-IG-activations",
}

METHOD_ORDER = ["EAP", "EAP-IG-Activations", "EAP-IG-Inputs"]

# Short legend labels only
METHOD_LEGEND_LABELS = {
    "EAP": "EAP",
    "EAP-IG-Activations": "EAP-IG-activations",
    "EAP-IG-Inputs": "EAP-IG-inputs",
}

# Green shades
METHOD_COLORS = {
    "EAP": "#b7e4c7",
    "EAP-IG-Activations": "#52b788",
    "EAP-IG-Inputs": "#1b4332",
}

TASK_PRETTY = {
    "ioi": "IOI",
    "mcqa": "MCQA",
    "arithmetic_addition": "Addition",
    "arithmetic_subtraction": "Subtraction",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
}


def apply_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
    })


def parse_list(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[\s,]+", s.strip())
    return [p for p in parts if p]


def _beautify(ax: plt.Axes):
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def pair_tick(src: float, tgt: float) -> str:
    def f(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return str(x).rstrip("0").rstrip(".")
    return f"{f(src)}→{f(tgt)}"


def parse_pair_from_csv(pair_disp: str) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    fam = family_for_pair_display(pair_disp)
    if fam == "other":
        fam = None
    nums = re.findall(r"(\d+(?:\.\d+)?)\s*B", pair_disp)
    if len(nums) >= 2:
        return fam, float(nums[0]), float(nums[1])
    return fam, None, None


def compute_gapclosed(b: float, ours: float, gold: float) -> float:
    denom = gold - b
    if denom is None or abs(denom) < 1e-12:
        return np.nan
    return (ours - b) / denom


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def task_label(task: str) -> str:
    return TASK_PRETTY.get(task, task.replace("_", "-"))


def load_table2_csvs(tables_dir: Path) -> Dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(tables_dir.glob("table2_summary_*.csv")):
        task = p.stem.replace("table2_summary_", "")
        df = pd.read_csv(p)
        for c in ["Baseline", "Ours (Best)", "Gold"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        out[task] = df
    return out


def baseline_gold_maps(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    baseline = {}
    gold = {}
    for _, row in df.iterrows():
        method_disp = str(row["Method"])
        fam, src, tgt = parse_pair_from_csv(str(row["Target Model Pair"]))
        if fam is None or src is None or tgt is None:
            continue
        baseline[(method_disp, fam, src, tgt)] = float(row["Baseline"]) if np.isfinite(row["Baseline"]) else np.nan
        gold[(method_disp, fam, src, tgt)] = float(row["Gold"]) if np.isfinite(row["Gold"]) else np.nan
    return baseline, gold


def get_src_tgt_models(fam: str, src_size: float, tgt_size: float) -> Tuple[str, str]:
    if fam not in FAMILY_ORDER:
        raise ValueError(f"Unknown family: {fam}")
    return model_by_size(fam, src_size, tgt_size)


def aligned_value_for_mode(
    results_root: Path,
    method_disp: str,
    task: str,
    fam: str,
    src_size: float,
    tgt_size: float,
    split: str,
    absolute: bool,
    ablation: str,
    level: str,
    mode: str,
    select_by: str,
) -> float:
    method_base = METHOD_DISPLAY_TO_BASE.get(method_disp, method_disp)
    src_model, tgt_model = get_src_tgt_models(fam, src_size, tgt_size)

    if mode == "in_distribution":
        train_mode = "train"
        train_spec = ""
    elif mode == "zero_shot":
        train_mode = "loo"
        train_spec = ""
    elif mode == "near_distribution":
        train_mode = "fixed"
        train_spec = "All"
    elif mode == "best":
        train_mode = "best"
        train_spec = ""
    else:
        raise ValueError(f"Unknown mode: {mode}")

    _, _, d = choose_best_aligned(
        results_root=results_root,
        method_base=method_base,
        src_model=src_model,
        tgt_model=tgt_model,
        ablation=ablation,
        level=level,
        task=task,
        split=split,
        absolute=absolute,
        train_mode=train_mode,
        fixed_train_spec=train_spec,
        prefer=select_by,
    )

    if d is None:
        return np.nan
    v = curve_score(d, prefer=select_by)
    return float(v) if v is not None else np.nan


def baseline_gold_from_idx(
    idx,
    method_disp: str,
    task: str,
    fam: str,
    src_size: float,
    tgt_size: float,
) -> Tuple[float, float]:
    method_base = METHOD_DISPLAY_TO_BASE.get(method_disp, method_disp)
    src_model, tgt_model = get_src_tgt_models(fam, src_size, tgt_size)

    baseline = query_value(
        idx,
        method_base,
        src_model,
        tgt_model,
        "diffalign",
        "random_W",
        f"LOO-{task}",
        task,
    )
    gold = query_value(
        idx,
        method_base,
        "",
        tgt_model,
        "gold",
        "",
        "",
        task,
    )

    baseline = float(baseline) if baseline is not None and np.isfinite(baseline) else np.nan
    gold = float(gold) if gold is not None and np.isfinite(gold) else np.nan
    return baseline, gold


def random_recovery_value(
    idx,
    method_disp: str,
    task: str,
    fam: str,
    src_size: float,
    tgt_size: float,
) -> float:
    method_base = METHOD_DISPLAY_TO_BASE.get(method_disp, method_disp)
    src_model, tgt_model = get_src_tgt_models(fam, src_size, tgt_size)

    rand = query_value(
        idx,
        method_base,
        src_model,
        tgt_model,
        "diffalign",
        "random_W",
        f"LOO-{task}",
        task,
    )
    gold = query_value(
        idx,
        method_base,
        "",
        tgt_model,
        "gold",
        "",
        "",
        task,
    )

    if rand is None or gold is None:
        return np.nan
    if not np.isfinite(rand) or not np.isfinite(gold) or abs(gold) < 1e-12:
        return np.nan
    return float(rand / gold)


def pair_random_recovery_segment(
    idx,
    task: str,
    fam: str,
    pair: Tuple[float, float],
) -> float:
    src_size, tgt_size = pair
    vals = []
    for method_disp in METHOD_ORDER:
        v = random_recovery_value(idx, method_disp, task, fam, src_size, tgt_size)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def pair_random_recovery_segment_mean(
    idx,
    tasks: List[str],
    fam: str,
    pair: Tuple[float, float],
) -> float:
    src_size, tgt_size = pair
    vals = []
    for task in tasks:
        for method_disp in METHOD_ORDER:
            v = random_recovery_value(idx, method_disp, task, fam, src_size, tgt_size)
            if np.isfinite(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def family_random_recovery_line(idx, task: str, fam: str) -> float:
    order = FAMILY_ORDER[fam]
    vals = []
    for (src_size, tgt_size) in order:
        for method_disp in METHOD_ORDER:
            v = random_recovery_value(idx, method_disp, task, fam, src_size, tgt_size)
            if np.isfinite(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def family_random_recovery_line_mean(idx, tasks: List[str], fam: str) -> float:
    order = FAMILY_ORDER[fam]
    vals = []
    for task in tasks:
        for (src_size, tgt_size) in order:
            for method_disp in METHOD_ORDER:
                v = random_recovery_value(idx, method_disp, task, fam, src_size, tgt_size)
                if np.isfinite(v):
                    vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def global_random_recovery_line(idx, task: str) -> float:
    vals = []
    for fam, order in FAMILY_ORDER_ITEMS:
        for (src_size, tgt_size) in order:
            for method_disp in METHOD_ORDER:
                v = random_recovery_value(idx, method_disp, task, fam, src_size, tgt_size)
                if np.isfinite(v):
                    vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def global_random_recovery_line_mean(idx, tasks: List[str]) -> float:
    vals = []
    for task in tasks:
        for fam, order in FAMILY_ORDER_ITEMS:
            for (src_size, tgt_size) in order:
                for method_disp in METHOD_ORDER:
                    v = random_recovery_value(idx, method_disp, task, fam, src_size, tgt_size)
                    if np.isfinite(v):
                        vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def method_legend_handles():
    return [
        Patch(
            facecolor=METHOD_COLORS[m],
            edgecolor="none",
            label=METHOD_LEGEND_LABELS[m],
        )
        for m in METHOD_ORDER
    ]


def draw_random_reference(
    ax: plt.Axes,
    fam: str,
    order: List[Tuple[float, float]],
    x_positions,
    random_ref,
    mode: str,
):
    if mode == "none" or random_ref is None:
        return

    if mode == "family":
        rv = random_ref.get(fam, np.nan)
        if np.isfinite(rv):
            ax.axhline(
                rv,
                linestyle=":",
                linewidth=1.0,
                color="black",
                alpha=1.0,
                zorder=4,
            )

    elif mode == "global":
        rv = random_ref.get("global", np.nan)
        if np.isfinite(rv):
            ax.axhline(
                rv,
                linestyle=":",
                linewidth=1.0,
                color="black",
                alpha=1.0,
                zorder=4,
            )

    elif mode == "pair":
        seg_half = 0.39
        fam_dict = random_ref.get(fam, {})
        for xi, pair in zip(x_positions, order):
            rv = fam_dict.get(pair, np.nan)
            if np.isfinite(rv):
                ax.hlines(
                    rv,
                    xi - seg_half,
                    xi + seg_half,
                    linestyles=":",
                    linewidth=1.0,
                    colors="black",
                    alpha=1.0,
                    zorder=4,
                )


def _finalize_figure(
    fig,
    axes,
    out_path: Path,
    no_title: bool = False,
    has_suptitle: bool = True,
):
    legend_y = 0.975 if has_suptitle and not no_title else 0.995
    top = 0.78 if has_suptitle and not no_title else 0.86

    fig.legend(
        handles=method_legend_handles(),
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, legend_y),
        borderpad=0.25,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.18,
        top=top,
        wspace=0.08,
    )

    safe_mkdir(out_path.parent)
    fig.savefig(str(out_path.with_suffix(".pdf")))
    fig.savefig(str(out_path.with_suffix(".png")), dpi=220)
    plt.close(fig)


def plot_task(metric: str, task: str, vals, out_path: Path, random_ref=None, random_line_mode="none"):
    fig, axes = plt.subplots(1, len(FAMILY_ORDER_ITEMS), figsize=(4.7 * len(FAMILY_ORDER_ITEMS), 3.8), sharey=True)
    axes = axes.ravel()

    for ax, (fam, order) in zip(axes, FAMILY_ORDER_ITEMS):
        x = np.arange(len(order))
        width = 0.24
        offsets = [-width, 0.0, width]

        for off, method_disp in zip(offsets, METHOD_ORDER):
            y = [vals.get(fam, {}).get(pair, {}).get(method_disp, np.nan) for pair in order]
            ax.bar(
                x + off,
                y,
                width=width,
                color=METHOD_COLORS[method_disp],
                label=method_disp,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([pair_tick(s, t) for (s, t) in order])
        ax.set_title(FAMILY_DISPLAY[fam], pad=2)
        _beautify(ax)

        if metric == "recovery":
            draw_random_reference(ax, fam, order, x, random_ref, random_line_mode)

    if metric == "recovery":
        fig.suptitle(f"Recovery Ratio on {task_label(task)} by Method", y=0.99)
        axes[0].set_ylabel("Faithfulness recovery ratio")
        for ax in axes:
            ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)
    else:
        fig.suptitle(f"Gap Closed on {task_label(task)} by Method", y=0.99)
        axes[0].set_ylabel("Faithfulness gap closed")
        for ax in axes:
            ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)

    _finalize_figure(fig, axes, out_path, no_title=False, has_suptitle=True)


def plot_mean(
    metric: str,
    mean_std,
    no_title: bool,
    out_path: Path,
    n_tasks_used: int,
    random_ref=None,
    random_line_mode="none",
):
    fig, axes = plt.subplots(1, len(FAMILY_ORDER_ITEMS), figsize=(4.7 * len(FAMILY_ORDER_ITEMS), 3.8), sharey=True)
    axes = axes.ravel()

    for ax, (fam, order) in zip(axes, FAMILY_ORDER_ITEMS):
        x = np.arange(len(order))
        width = 0.24
        offsets = [-width, 0.0, width]

        for off, method_disp in zip(offsets, METHOD_ORDER):
            means = []
            stds = []
            for pair in order:
                m, s = mean_std.get(fam, {}).get(pair, {}).get(method_disp, (np.nan, np.nan))
                means.append(m)
                stds.append(s)

            ax.bar(
                x + off,
                means,
                width=width,
                yerr=stds,
                capsize=3,
                color=METHOD_COLORS[method_disp],
                label=method_disp,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([pair_tick(s, t) for (s, t) in order])
        ax.set_title(FAMILY_DISPLAY[fam], pad=2)
        _beautify(ax)

        if metric == "recovery":
            draw_random_reference(ax, fam, order, x, random_ref, random_line_mode)

    if metric == "recovery":
        if not no_title:
            fig.suptitle("Mean Recovery Ratio Across Tasks by Method", y=0.99)
        axes[0].set_ylabel("Faithfulness recovery ratio")
        for ax in axes:
            ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)
    else:
        if not no_title:
            fig.suptitle("Mean Gap Closed Across Tasks by Method", y=0.99)
        axes[0].set_ylabel("Faithfulness gap closed")
        for ax in axes:
            ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)

    _finalize_figure(fig, axes, out_path, no_title=no_title, has_suptitle=True)


def main():
    apply_style()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default="tables", help="Dir with table2_summary_<task>.csv")
    ap.add_argument("--results-dir", default="results", help="Dir with evaluation pkls")
    ap.add_argument("--out-dir", default="figures/figure2_cross_model_by_method", help="Base output dir")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--absolute", default="False", choices=["True", "False"])
    ap.add_argument("--ablation", default="patching")
    ap.add_argument("--level", default="node")
    ap.add_argument("--select-by", default="area_under", choices=["area_under", "average", "faithfulness@100"])
    ap.add_argument(
        "--modes",
        default="zero_shot,in_distribution,near_distribution,best",
        help="Comma list from {zero_shot,in_distribution,near_distribution,best}",
    )
    ap.add_argument("--exclude-tasks", default="", help="Drop tasks entirely.")
    ap.add_argument("--drop-tasks-for-mean", default="", help="Drop tasks only for mean.")
    ap.add_argument(
        "--mean-policy",
        default="per_pair",
        choices=["per_pair", "intersection"],
        help="Mean aggregation when some tasks missing.",
    )
    ap.add_argument(
        "--random-line-mode",
        default="family",
        choices=["none", "family", "pair", "global"],
        help="How to draw the random baseline on recovery plots.",
    )
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args()

    tables_dir = Path(args.tables_dir)
    results_root = Path(args.results_dir)
    out_base = Path(args.out_dir)
    split = args.split
    absolute = (args.absolute == "True")
    modes = parse_list(args.modes)

    exclude = set(parse_list(args.exclude_tasks))
    drop_mean = set(parse_list(args.drop_tasks_for_mean))

    records = load_records(str(results_root), args.select_by)
    idx = make_index(records, split, args.ablation, args.level, absolute)

    task_to_df = load_table2_csvs(tables_dir)

    if task_to_df:
        tasks = [t for t in task_to_df.keys() if t not in exclude]
        tasks = [t for t in tasks if t in DEFAULT_TASKS] + [t for t in tasks if t not in DEFAULT_TASKS]
    else:
        tasks = [t for t in DEFAULT_TASKS if t not in exclude]

    if not tasks:
        raise RuntimeError(
            f"No tasks available. exclude={sorted(exclude)}; "
            f"found_csv_tasks={sorted(task_to_df.keys()) if task_to_df else []}"
        )

    baseline_map = {}
    gold_map = {}
    for t in tasks:
        if t in task_to_df:
            b, g = baseline_gold_maps(task_to_df[t])
            baseline_map[t] = b
            gold_map[t] = g
        else:
            baseline_map[t] = {}
            gold_map[t] = {}

    for mode in modes:
        out_dir = out_base / mode
        safe_mkdir(out_dir)

        per_task_metric_vals = {"recovery": [], "gapclosed": []}

        for t in tasks:
            vals_recovery = {fam: {} for fam in FAMILY_ORDER}
            vals_gap = {fam: {} for fam in FAMILY_ORDER}

            for fam, order in FAMILY_ORDER_ITEMS:
                for (src_size, tgt_size) in order:
                    pair = (src_size, tgt_size)
                    for method_disp in METHOD_ORDER:
                        base = baseline_map[t].get((method_disp, fam, src_size, tgt_size), np.nan)
                        gold = gold_map[t].get((method_disp, fam, src_size, tgt_size), np.nan)

                        if not np.isfinite(base) or not np.isfinite(gold):
                            base_fallback, gold_fallback = baseline_gold_from_idx(
                                idx=idx,
                                method_disp=method_disp,
                                task=t,
                                fam=fam,
                                src_size=src_size,
                                tgt_size=tgt_size,
                            )
                            if not np.isfinite(base):
                                base = base_fallback
                            if not np.isfinite(gold):
                                gold = gold_fallback

                        ours = aligned_value_for_mode(
                            results_root=results_root,
                            method_disp=method_disp,
                            task=t,
                            fam=fam,
                            src_size=src_size,
                            tgt_size=tgt_size,
                            split=split,
                            absolute=absolute,
                            ablation=args.ablation,
                            level=args.level,
                            mode=mode,
                            select_by=args.select_by,
                        )

                        rec = ours / gold if np.isfinite(ours) and np.isfinite(gold) and abs(gold) > 1e-12 else np.nan
                        gap = compute_gapclosed(base, ours, gold) if np.isfinite(base) and np.isfinite(ours) and np.isfinite(gold) else np.nan

                        vals_recovery.setdefault(fam, {}).setdefault(pair, {})[method_disp] = rec
                        vals_gap.setdefault(fam, {}).setdefault(pair, {})[method_disp] = gap

            if args.random_line_mode == "family":
                random_ref = {
                    fam: family_random_recovery_line(idx, t, fam)
                    for fam in FAMILY_ORDER
                }
            elif args.random_line_mode == "pair":
                random_ref = {
                    fam: {
                        pair: pair_random_recovery_segment(idx, t, fam, pair)
                        for pair in order
                    }
                    for fam, order in FAMILY_ORDER_ITEMS
                }
            elif args.random_line_mode == "global":
                random_ref = {"global": global_random_recovery_line(idx, t)}
            else:
                random_ref = None

            plot_task(
                "recovery",
                t,
                vals_recovery,
                out_dir / f"figure2_recovery_{t}_by_method",
                random_ref=random_ref,
                random_line_mode=args.random_line_mode,
            )
            plot_task(
                "gapclosed",
                t,
                vals_gap,
                out_dir / f"figure2_gapclosed_{t}_by_method",
                random_ref=None,
                random_line_mode="none",
            )

            per_task_metric_vals["recovery"].append((t, vals_recovery))
            per_task_metric_vals["gapclosed"].append((t, vals_gap))

        tasks_for_mean = [t for t in tasks if t not in drop_mean]
        if not tasks_for_mean:
            raise RuntimeError("No tasks remain for mean after drop-tasks-for-mean.")

        for metric in ("recovery", "gapclosed"):
            vals_list = [vals for (t, vals) in per_task_metric_vals[metric] if t in tasks_for_mean]

            if args.mean_policy == "intersection":
                kept = []
                for vals in vals_list:
                    ok = True
                    for fam, order in FAMILY_ORDER_ITEMS:
                        for pair in order:
                            for method_disp in METHOD_ORDER:
                                v = vals.get(fam, {}).get(pair, {}).get(method_disp, np.nan)
                                if not np.isfinite(v):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        kept.append(vals)
                vals_list = kept

            mean_std = {fam: {} for fam in FAMILY_ORDER}
            for fam, order in FAMILY_ORDER_ITEMS:
                for pair in order:
                    mean_std[fam][pair] = {}
                    for method_disp in METHOD_ORDER:
                        coll = []
                        for vals in vals_list:
                            v = vals.get(fam, {}).get(pair, {}).get(method_disp, np.nan)
                            if np.isfinite(v):
                                coll.append(v)
                        if coll:
                            mean_std[fam][pair][method_disp] = (float(np.mean(coll)), float(np.std(coll)))
                        else:
                            mean_std[fam][pair][method_disp] = (np.nan, np.nan)

            if metric == "recovery":
                if args.random_line_mode == "family":
                    random_ref_mean = {
                        fam: family_random_recovery_line_mean(idx, tasks_for_mean, fam)
                        for fam in FAMILY_ORDER
                    }
                elif args.random_line_mode == "pair":
                    random_ref_mean = {
                        fam: {
                            pair: pair_random_recovery_segment_mean(idx, tasks_for_mean, fam, pair)
                            for pair in order
                        }
                        for fam, order in FAMILY_ORDER_ITEMS
                    }
                elif args.random_line_mode == "global":
                    random_ref_mean = {"global": global_random_recovery_line_mean(idx, tasks_for_mean)}
                else:
                    random_ref_mean = None
            else:
                random_ref_mean = None

            plot_mean(
                metric,
                mean_std,
                no_title=args.no_title,
                out_path=out_dir / f"figure2_{metric}_mean_over_tasks_by_method",
                n_tasks_used=len(vals_list),
                random_ref=random_ref_mean,
                random_line_mode=(args.random_line_mode if metric == "recovery" else "none"),
            )

    print(f"✓ Wrote Figure2-by-method to: {out_base} (modes: {modes})")


if __name__ == "__main__":
    main()
