# mib-circuit-track/build_figure4_heatmap.py
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from table_utils import (
    METHODS,
    METHOD_DISPLAY,
    MODEL_PAIRS,
    TASKS,
    SHORT_TASK_LABEL,
    load_records,
    make_index,
    parse_args_base,
    slugify,
)
from model_config import family_for_model, FAMILY_DISPLAY, pretty_model_size

# ----------------------------
# Family + compact titles
# ----------------------------

def detect_family(model_name: str) -> str:
    return family_for_model(model_name)


def _extract_size(model_name: str) -> str:
    return pretty_model_size(model_name)


def _extract_family_display(model_name: str) -> str:
    fam = family_for_model(model_name)
    return FAMILY_DISPLAY.get(fam, model_name)


def pretty_pair_compact(source_model: str, target_model: str) -> str:
    fam = _extract_family_display(source_model)
    s_sz = _extract_size(source_model)
    t_sz = _extract_size(target_model)
    return f"{fam} {s_sz} $\\rightarrow$ {t_sz}"


# ----------------------------
# Data extraction
# ----------------------------

def build_matrix(
    idx,
    method: str,
    source_model: str,
    target_model: str,
    task_names: List[str],
) -> np.ndarray:
    """matrix[row=train_task][col=eval_task] as float array with NaNs."""
    mat = np.full((len(task_names), len(task_names)), np.nan, dtype=float)
    for i, train_task in enumerate(task_names):
        for j, eval_task in enumerate(task_names):
            v = idx.get(
                (
                    method,
                    source_model,
                    target_model,
                    "diffalign",
                    "none",
                    f"TRAIN-{train_task}",
                    eval_task,
                    "",
                )
            )
            if v is not None:
                mat[i, j] = float(v)
    return mat


def _shared_vmin_vmax(mats: List[np.ndarray]) -> Tuple[float, float]:
    finite_vals = []
    for m in mats:
        v = m[np.isfinite(m)]
        if v.size:
            finite_vals.append(v)
    if not finite_vals:
        return 0.0, 1.0
    all_vals = np.concatenate(finite_vals)
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


# ----------------------------
# Plotting
# ----------------------------

def plot_three_panel(
    matrices: List[np.ndarray],
    subtitles: List[str],
    task_labels_y: List[str],
    task_labels_x: List[str],
    x_col_indices: List[int],
    outpath_base: str,
) -> None:
    """
    1 figure with 3 subplots (1x3), shared color scale.
    - No figure-level title.
    - Larger ticks.
    - Qwen can exclude a column on x-axis (evaluation) while keeping y intact.
    - Axis labels bold+large, positioned to avoid overlap.
    - Colorbar label: Faithfulness (CPR).
    - Cell annotations show CPR values (area_under).
    """
    assert len(matrices) == 3
    assert len(subtitles) == 3

    # Slice columns if x_col_indices excludes something
    sliced = [m[:, x_col_indices] for m in matrices]
    vmin, vmax = _shared_vmin_vmax(sliced)

    # higher -> darker (reverse of default viridis)
    # cmap = plt.get_cmap("viridis_r").copy()
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(1, 3, figsize=(18.8, 6.8))

    # Your current layout preference
    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.20, top=0.88, wspace=0.25)

    # Fonts
    tick_fs = 15
    title_fs = 20
    cell_fs = 11

    axis_label_fs = 18
    axis_label_pad = 22  # more separation from ticks
    tick_pad = 10

    im_last = None
    for k, ax in enumerate(axes):
        arr = np.ma.masked_invalid(sliced[k])
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        im_last = im

        ax.set_title(subtitles[k], fontsize=title_fs, pad=12)

        # X ticks (evaluation)
        ax.set_xticks(np.arange(len(task_labels_x)))
        ax.set_xticklabels(task_labels_x, rotation=25, ha="right")
        ax.tick_params(axis="x", labelsize=tick_fs, pad=tick_pad)

        # Y ticks (training)
        ax.set_yticks(np.arange(len(task_labels_y)))
        if k == 0:
            ax.set_yticklabels(task_labels_y)
            ax.tick_params(axis="y", labelsize=tick_fs, pad=tick_pad)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", labelsize=tick_fs, pad=tick_pad)

        # Cell annotations
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = sliced[k][i, j]
                txt = "-" if not np.isfinite(v) else f"{v:.2f}"
                # ax.text(j, i, txt, ha="center", va="center", fontsize=cell_fs)
                text_color = "white" if np.isfinite(v) and v > (vmin + vmax) / 2 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=cell_fs, color=text_color)

    # Axis labels: put y-label on left subplot, x-label on middle subplot
    # (prevents overlaps with big ticks)
    axes[0].set_ylabel("Training task", fontsize=axis_label_fs, fontweight="bold", labelpad=axis_label_pad)
    axes[1].set_xlabel("Evaluation task", fontsize=axis_label_fs, fontweight="bold", labelpad=axis_label_pad)

    # Shared colorbar on the right
    cbar = fig.colorbar(im_last, ax=axes, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("Faithfulness (CPR)", rotation=90, fontsize=16, fontweight="bold", labelpad=12)
    cbar.ax.tick_params(labelsize=12)

    fig.savefig(outpath_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(outpath_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args_base("Build Figure 4 grouped heatmaps (3 subplots per family).")

    records = load_records(args.results_dir, args.metric_key)
    idx = make_index(records, args.split, args.ablation, args.level, args.absolute)

    task_names = [t for t, _ in TASKS]
    y_labels = [SHORT_TASK_LABEL[t] for t in task_names]  # training ticks always full

    # Find which task is Arith +
    arith_plus_idx = None
    for i, t in enumerate(task_names):
        if t in ("arithmetic_addition", "arith_add", "arith+", "arith_addition"):
            arith_plus_idx = i
            break

    # # Group pairs by family
    # family_pairs: Dict[str, List[Tuple[str, str]]] = {}
    # for src, tgt in MODEL_PAIRS:
    #     fam = detect_family(src)
    #     family_pairs.setdefault(fam, []).append((src, tgt))

    # Group pairs by family
    family_pairs: Dict[str, List[Tuple[str, str]]] = {}
    for src, tgt in MODEL_PAIRS:
        fam = detect_family(src)
        family_pairs.setdefault(fam, []).append((src, tgt))


    out_dir = os.path.join(args.output_dir, "figure4_heatmaps_grouped")
    os.makedirs(out_dir, exist_ok=True)

    for fam, pairs in family_pairs.items():
        if fam == "other":
            continue

        pairs_sorted = pairs
        if len(pairs_sorted) != 3:
            raise ValueError(
                f"Expected 3 model pairs for family '{fam}', got {len(pairs_sorted)}: {pairs_sorted}"
            )

        # X axis (evaluation) task labels:
        # - LLaMA: keep all 6
        # - Qwen: drop Arith + from columns only, keep rows intact
        if fam == "qwen" and arith_plus_idx is not None:
            x_indices = [i for i in range(len(task_names)) if i != arith_plus_idx]
            x_labels = [y_labels[i] for i in x_indices]
        else:
            x_indices = list(range(len(task_names)))
            x_labels = y_labels

        for method in METHODS:
            mats: List[np.ndarray] = []
            subs: List[str] = []
            for src, tgt in pairs_sorted:
                mats.append(build_matrix(idx, method, src, tgt, task_names))
                subs.append(pretty_pair_compact(src, tgt))  # <-- compact title

            stem = f"fig4_heatmap_grouped_{slugify(method)}_{fam}"
            outpath_base = os.path.join(out_dir, stem)

            plot_three_panel(
                matrices=mats,
                subtitles=subs,
                task_labels_y=y_labels,
                task_labels_x=x_labels,
                x_col_indices=x_indices,
                outpath_base=outpath_base,
            )

    print(f"[OK] Wrote grouped heatmaps to: {out_dir}")


if __name__ == "__main__":
    main()
