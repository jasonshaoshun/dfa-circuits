#!/usr/bin/env python3
from __future__ import annotations

"""Figure 2 (Cross-Model Scaling) generator (from Table 2 summary CSVs).

Reads the CSVs produced by build_table2_summary.py (table2_summary_<task>.csv),
computes:
  - Recovery ratio  = Ours / Gold
  - GapClosed       = (Ours - Baseline) / (Gold - Baseline)

Then for EACH (task, model_pair) it selects "Best across methods" by taking the max
metric value across methods (EAP, EAP-IG-Activations, EAP-IG-Inputs).

Outputs:
  - 6 figures for GapClosed (one per task)
  - 6 figures for Recovery (one per task)
  - 1 mean figure over tasks for GapClosed (mean±std across tasks)
  - 1 mean figure over tasks for Recovery (mean±std across tasks)

Each figure has 2 panels: LLaMA family (left) and Qwen family (right).

New options:
  --exclude-tasks:        drop tasks entirely (no per-task figs; excluded from mean)
  --drop-tasks-for-mean:  exclude tasks only from mean-over-tasks figures
  --mean-policy:          intersection (default) or per_pair
      - intersection: only tasks with complete coverage across ALL 6 pairs contribute to the mean
      - per_pair: mean for each pair uses whatever tasks are available for that pair

Styling:
  Defaults are set to be more "paper-friendly":
    - smaller suptitle (or disabled with --no-suptitle)
    - lollipop plot for mean figures (cleaner than big bars)
    - light y-grid, no top/right spines
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Set

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_config import FAMILY_DISPLAY, FAMILY_ORDER, family_for_pair_display

FAMILY_ORDER_ITEMS = tuple(FAMILY_ORDER.items())

METHOD_SHORT = {
    "EAP-IG-Inputs": "IG-In",
    "EAP-IG-Activations": "IG-Act",
    "EAP": "EAP",
}

TASK_NAME_MAP = {
    "ioi": "IOI",
    "mcqa": "MCQA",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "arithmetic_addition": "Addition",
    "arithmetic_subtraction": "Subtraction",
    "addition": "Addition",
    "subtraction": "Subtraction",
}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.5,
    })


def _parse_list(s: str) -> Set[str]:
    """Parse comma/space-separated list into a set of strings."""
    if not s:
        return set()
    parts = re.split(r"[\s,]+", s.strip())
    return {p for p in parts if p}


def parse_sizes_from_pair_display(pair_disp: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract src/tgt sizes from strings like 'LLaMA-3 (1B $\\to$ 3B)' or 'Qwen-2.5 (0.5B -> 3B)'."""
    nums = re.findall(r"(\d+(?:\.\d+)?)\s*B", pair_disp)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    nums2 = re.findall(r"(\d+(?:\.\d+)?)", pair_disp)
    if len(nums2) >= 2:
        try:
            return float(nums2[0]), float(nums2[1])
        except Exception:
            return None, None
    return None, None


def family_and_sizes(pair_disp: str) -> Tuple[str, Optional[float], Optional[float]]:
    fam = family_for_pair_display(pair_disp)
    src, tgt = parse_sizes_from_pair_display(pair_disp)
    return fam, src, tgt


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Baseline", "Ours (Best)", "Gold"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["recovery"] = df["Ours (Best)"] / df["Gold"]
    denom = (df["Gold"] - df["Baseline"])
    df["gapclosed"] = (df["Ours (Best)"] - df["Baseline"]) / denom
    df.loc[denom.abs() < 1e-12, "gapclosed"] = np.nan
    return df


def select_best_across_methods(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """For each Target Model Pair, select the row with maximum metric value across methods."""
    df2 = df.dropna(subset=[metric]).copy()
    best_idx = df2.groupby("Target Model Pair")[metric].idxmax()
    best = df2.loc[best_idx].copy()
    best["method_short"] = best["Method"].map(METHOD_SHORT).fillna(best["Method"])
    best["value"] = best[metric]
    return best


def make_task_title(task_id: str) -> str:
    if task_id in TASK_NAME_MAP:
        return TASK_NAME_MAP[task_id]
    for k, v in TASK_NAME_MAP.items():
        if task_id.startswith(k):
            return v
    return task_id.replace("_", "-").upper()


def pair_tick(src: float, tgt: float) -> str:
    def f(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return str(x).rstrip("0").rstrip(".")
    return f"{f(src)}→{f(tgt)}"


def collect_values(best: pd.DataFrame) -> Dict[str, Dict[Tuple[float, float], Tuple[float, str]]]:
    """Return {family: {(src,tgt): (value, best_method_short)}}"""
    out: Dict[str, Dict[Tuple[float, float], Tuple[float, str]]] = {fam: {} for fam in FAMILY_ORDER}
    for _, row in best.iterrows():
        fam, src, tgt = family_and_sizes(str(row["Target Model Pair"]))
        if fam not in FAMILY_ORDER or src is None or tgt is None:
            continue
        out[fam][(float(src), float(tgt))] = (float(row["value"]), str(row["method_short"]))
    return out


def _beautify_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_two_panel_task(
    values: Dict[str, Dict[Tuple[float,float], Tuple[float,str]]],
    metric: str,
    title: str,
    out_prefix: Path,
    ylim: Tuple[float, float],
    annotate_method: bool,
    fmt: str,
    no_suptitle: bool,
    task_plot: str,
) -> None:
    """Per-task figure."""
    fig, axes = plt.subplots(1, len(FAMILY_ORDER_ITEMS), figsize=(4.5 * len(FAMILY_ORDER_ITEMS), 3.0), sharey=True)
    axes = axes.ravel()

    for ax, (fam, order) in zip(axes, FAMILY_ORDER_ITEMS):
        xs = np.arange(len(order))
        ys, ann = [], []
        for pair in order:
            if pair in values.get(fam, {}):
                v, m = values[fam][pair]
                ys.append(v); ann.append(m)
            else:
                ys.append(np.nan); ann.append("")

        ys = np.array(ys, dtype=float)

        if task_plot == "line":
            ax.plot(xs, ys, marker="o")
        elif task_plot == "lollipop":
            # lollipop: stem from 0 to point
            for x, y in zip(xs, ys):
                if np.isfinite(y):
                    ax.vlines(x, 0.0, y, linewidth=2.0, alpha=0.9)
            ax.plot(xs, ys, marker="o", linestyle="None")
        else:
            raise ValueError("Unknown --task-plot")

        ax.set_xticks(xs)
        ax.set_xticklabels([pair_tick(s, t) for (s, t) in order])
        ax.axhline(1.0, linestyle="--", linewidth=1.0)
        ax.set_title(FAMILY_DISPLAY[fam])
        ax.set_ylim(*ylim)
        _beautify_axis(ax)

        if annotate_method:
            for x, y, m in zip(xs, ys, ann):
                if np.isfinite(y) and m:
                    ax.annotate(m, (x, y), textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=9)

    ylabel = "Gap closed" if metric == "gapclosed" else "Recovery (Ours / Gold)"
    axes[0].set_ylabel(ylabel)

    if not no_suptitle:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_prefix.with_suffix(f".{fmt}")), bbox_inches="tight")
    if fmt != "png":
        fig.savefig(str(out_prefix.with_suffix(".png")), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_two_panel_mean(
    mean_std: Dict[str, Dict[Tuple[float,float], Tuple[float,float]]],
    metric: str,
    title: str,
    subtitle: str,
    out_prefix: Path,
    ylim: Tuple[float, float],
    fmt: str,
    no_suptitle: bool,
    mean_plot: str,
) -> None:
    """Mean-over-tasks figure."""
    fig, axes = plt.subplots(1, len(FAMILY_ORDER_ITEMS), figsize=(4.5 * len(FAMILY_ORDER_ITEMS), 3.0), sharey=True)
    axes = axes.ravel()

    for ax, (fam, order) in zip(axes, FAMILY_ORDER_ITEMS):
        xs = np.arange(len(order))
        means, stds = [], []
        for pair in order:
            m, s = mean_std.get(fam, {}).get(pair, (np.nan, np.nan))
            means.append(m); stds.append(s)
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        if mean_plot == "bar":
            ax.bar(xs, means, yerr=stds, capsize=3)
        elif mean_plot == "lollipop":
            # cleaner: errorbar points + stems
            ax.errorbar(xs, means, yerr=stds, fmt="o", capsize=3, linestyle="None")
            for x, y in zip(xs, means):
                if np.isfinite(y):
                    ax.vlines(x, 0.0, y, linewidth=2.0, alpha=0.6)
        else:
            raise ValueError("Unknown --mean-plot")

        ax.set_xticks(xs)
        ax.set_xticklabels([pair_tick(s, t) for (s, t) in order])
        ax.axhline(1.0, linestyle="--", linewidth=1.0)
        ax.set_title(FAMILY_DISPLAY[fam])
        ax.set_ylim(*ylim)
        _beautify_axis(ax)

    ylabel = "Gap closed" if metric == "gapclosed" else "Recovery (Ours / Gold)"
    axes[0].set_ylabel(ylabel)

    if not no_suptitle:
        fig.suptitle(title, y=1.02)
    # subtitle as small text below (paper-friendly)
    fig.text(0.5, -0.02, subtitle, ha="center", va="top", fontsize=10)

    fig.tight_layout()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_prefix.with_suffix(f".{fmt}")), bbox_inches="tight")
    if fmt != "png":
        fig.savefig(str(out_prefix.with_suffix(".png")), dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_mean_std(
    per_task_values: Dict[str, Dict[str, Dict[str, Dict[Tuple[float,float], Tuple[float,str]]]]],
    metric: str,
    task_ids_for_mean: List[str],
    mean_policy: str,
) -> Tuple[Dict[str, Dict[Tuple[float,float], Tuple[float,float]]], List[str]]:
    """
    Returns (mean_std, tasks_used).

    mean_policy:
      - intersection: only include tasks that have finite values for all configured pairs
      - per_pair: compute mean per pair using whatever tasks have finite values for that pair
    """
    if mean_policy not in ("intersection", "per_pair"):
        raise ValueError("mean_policy must be intersection or per_pair")

    order_all = FAMILY_ORDER_ITEMS

    tasks_used = list(task_ids_for_mean)

    if mean_policy == "intersection":
        good_tasks = []
        for t in task_ids_for_mean:
            ok = True
            for fam, order in order_all:
                for pair in order:
                    v = per_task_values[metric][t].get(fam, {}).get(pair, (np.nan, ""))[0]
                    if not np.isfinite(v):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                good_tasks.append(t)
        tasks_used = good_tasks

    mean_std: Dict[str, Dict[Tuple[float,float], Tuple[float,float]]] = {fam: {} for fam in FAMILY_ORDER}
    for fam, order in order_all:
        for pair in order:
            vals = []
            for t in tasks_used if mean_policy == "intersection" else task_ids_for_mean:
                v = per_task_values[metric][t].get(fam, {}).get(pair, (np.nan, ""))[0]
                if np.isfinite(v):
                    vals.append(v)
            if vals:
                mean_std[fam][pair] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                mean_std[fam][pair] = (np.nan, np.nan)

    return mean_std, tasks_used


def main():
    apply_paper_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default="tables", help="Directory containing table2_summary_<task>.csv files.")
    ap.add_argument("--out-dir", default="figures/figure2_cross_model", help="Output directory.")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"], help="Primary output format.")
    ap.add_argument("--exclude-tasks", default="", help="Comma/space-separated task ids to skip entirely.")
    ap.add_argument("--drop-tasks-for-mean", default="", help="Comma/space-separated task ids to exclude only from mean figures.")
    ap.add_argument("--mean-policy", default="intersection", choices=["intersection", "per_pair"],
                    help="How to compute mean-over-tasks when some tasks are missing for some pairs.")
    ap.add_argument("--no-annotate-method", action="store_true", help="Disable method-short annotations on per-task points.")
    ap.add_argument("--no-suptitle", action="store_true", help="Disable the big title (recommended for paper figures; use caption instead).")
    ap.add_argument("--task-plot", default="line", choices=["line", "lollipop"], help="Per-task plot style.")
    ap.add_argument("--mean-plot", default="lollipop", choices=["lollipop", "bar"], help="Mean plot style.")
    args = ap.parse_args()

    exclude_tasks = _parse_list(args.exclude_tasks)
    drop_tasks_for_mean = _parse_list(args.drop_tasks_for_mean)

    tables_dir = Path(args.tables_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(tables_dir.glob("table2_summary_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found: {tables_dir}/table2_summary_*.csv")

    per_task: Dict[str, Dict[str, Dict[str, Dict[Tuple[float,float], Tuple[float,str]]]]] = {
        "gapclosed": {},
        "recovery": {},
    }

    summary_rows = []
    global_vals = {"gapclosed": [], "recovery": []}
    task_ids: List[str] = []

    for p in csv_paths:
        task_id = p.stem.replace("table2_summary_", "")
        if task_id in exclude_tasks:
            continue

        df = pd.read_csv(p)
        df = compute_metrics(df)

        task_ids.append(task_id)

        for metric in ("gapclosed", "recovery"):
            best = select_best_across_methods(df, metric)
            values = collect_values(best)
            per_task[metric][task_id] = values

            for fam in FAMILY_ORDER:
                for (src, tgt), (v, mshort) in values.get(fam, {}).items():
                    if np.isfinite(v):
                        global_vals[metric].append(v)
                    summary_rows.append({
                        "task": task_id,
                        "metric": metric,
                        "family": fam,
                        "src_B": src,
                        "tgt_B": tgt,
                        "pair": f"{src}B->{tgt}B",
                        "best_value": v,
                        "best_method": mshort,
                    })

    if not task_ids:
        raise RuntimeError("After --exclude-tasks, no tasks remain.")

    # robust y-lims per metric (consistent across figures)
    ylims = {}
    for metric in ("gapclosed", "recovery"):
        arr = np.array(global_vals[metric], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ylims[metric] = (0.0, 1.05)
        else:
            hi = float(np.quantile(arr, 0.98))
            hi = max(1.05, hi * 1.10)
            hi = min(2.0, hi)
            ylims[metric] = (0.0, hi)

    # Write summary CSV (best-per-task-per-pair)
    pd.DataFrame(summary_rows).to_csv(out_dir / "figure2_cross_model_best_by_task_pair.csv", index=False)

    # Per-task figures
    for task_id in task_ids:
        task_title = make_task_title(task_id)
        for metric in ("gapclosed", "recovery"):
            title = f"Figure 2 — {task_title} — Best across methods"
            out_prefix = out_dir / f"figure2_cross_model_{metric}_{task_id}"
            plot_two_panel_task(
                values=per_task[metric][task_id],
                metric=metric,
                title=title,
                out_prefix=out_prefix,
                ylim=ylims[metric],
                annotate_method=(not args.no_annotate_method),
                fmt=args.format,
                no_suptitle=args.no_suptitle,
                task_plot=args.task_plot,
            )

    # Mean-over-tasks figures
    task_ids_for_mean = [t for t in task_ids if t not in drop_tasks_for_mean]
    if not task_ids_for_mean:
        raise RuntimeError("After --drop-tasks-for-mean, no tasks remain for mean figures.")

    for metric in ("gapclosed", "recovery"):
        mean_std, tasks_used = compute_mean_std(
            per_task_values=per_task,
            metric=metric,
            task_ids_for_mean=task_ids_for_mean,
            mean_policy=args.mean_policy,
        )

        # Subtitle tells exactly which tasks contributed
        tasks_used_str = ", ".join(tasks_used) if tasks_used else "(none)"
        subtitle = f"Mean±std over {len(tasks_used)} tasks ({args.mean_policy} policy). Tasks: {tasks_used_str}"

        title = "Figure 2 — Cross-Model Scaling — Mean over tasks — Best across methods"
        out_prefix = out_dir / f"figure2_cross_model_{metric}_mean_over_tasks"
        plot_two_panel_mean(
            mean_std=mean_std,
            metric=metric,
            title=title,
            subtitle=subtitle,
            out_prefix=out_prefix,
            ylim=ylims[metric],
            fmt=args.format,
            no_suptitle=args.no_suptitle,
            mean_plot=args.mean_plot,
        )

    print(f"✓ Wrote figures to: {out_dir}")
    print(f"✓ Summary CSV: {out_dir / 'figure2_cross_model_best_by_task_pair.csv'}")
    print(f"✓ Mean policy: {args.mean_policy}")
    print(f"✓ Excluded tasks (all): {sorted(exclude_tasks)}")
    print(f"✓ Dropped tasks (mean only): {sorted(drop_tasks_for_mean)}")


if __name__ == "__main__":
    main()
