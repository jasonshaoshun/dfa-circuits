#!/usr/bin/env python3
"""
Utilities for Figure 1 curves, aligned with your pipeline scripts.

Key conventions (from your scripts):

- circuits.sh (gold circuits):
    circuits/{method}_{ablation}_{level}/{task-with-dashes}_{model}/importances.json
  fileciteturn27file1

- evaluation.sh (evaluation outputs):
    results/{method_tag}_{ablation}_{level}/{task-with-dashes}_{model}_{split}_abs-{absolute}.pkl
  where:
    gold: method_tag = method_base (e.g., "EAP-IG-inputs")
    aligned: method_tag = "{method_base}__kind-diffalign__src-{src}__train-{train_spec}__ctrl-{control}"
  fileciteturn28file0

- run_evaluation.py saves a dict containing 'faithfulnesses' (length 10) and usually 'area_under'
  fileciteturn27file4
"""

from __future__ import annotations
import re
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from model_config import DEFAULT_MODEL_PAIRS, FAMILY_PAIRS

BUDGET_FRACS: Tuple[float, ...] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
BUDGET_PCTS: Tuple[float, ...] = tuple(100.0 * x for x in BUDGET_FRACS)

DEFAULT_TASKS: Tuple[str, ...] = (
    "ioi",
    "mcqa",
    "arithmetic_addition",
    "arithmetic_subtraction",
    "arc_easy",
    "arc_challenge",
)

DEFAULT_METHOD_BASES: Tuple[str, ...] = (
    "EAP",
    "EAP-IG-inputs",
    "EAP-IG-activations",
)

GPT2_PAIRS: Tuple[Tuple[str, str], ...] = FAMILY_PAIRS["gpt2"]
LLAMA_PAIRS: Tuple[Tuple[str, str], ...] = FAMILY_PAIRS["llama"]
QWEN_PAIRS: Tuple[Tuple[str, str], ...] = FAMILY_PAIRS["qwen"]
GEMMA_PAIRS: Tuple[Tuple[str, str], ...] = FAMILY_PAIRS["gemma"]


def apply_paper_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
    })


def task_to_fname(task: str) -> str:
    return task.replace("_", "-")


def results_dir_name(method_tag: str, ablation: str, level: str) -> str:
    return f"{method_tag}_{ablation}_{level}"


def result_pkl_path(
    results_root: Path,
    method_tag: str,
    ablation: str,
    level: str,
    task: str,
    model: str,
    split: str,
    absolute: bool,
) -> Path:
    d = results_root / results_dir_name(method_tag, ablation, level)
    fname = f"{task_to_fname(task)}_{model}_{split}_abs-{absolute}.pkl"
    return d / fname


def load_result(
    results_root: Path,
    method_tag: str,
    ablation: str,
    level: str,
    task: str,
    model: str,
    split: str,
    absolute: bool,
) -> Optional[dict]:
    p = result_pkl_path(results_root, method_tag, ablation, level, task, model, split, absolute)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def get_faithfulness_curve(d: Optional[dict]) -> Optional[np.ndarray]:
    if d is None or "faithfulnesses" not in d:
        return None
    arr = np.array(d["faithfulnesses"], dtype=float)
    if arr.ndim != 1 or arr.size != len(BUDGET_PCTS):
        return None
    return arr


def curve_score(d: Optional[dict], prefer: str = "area_under") -> Optional[float]:
    if d is None:
        return None
    if prefer in d and isinstance(d[prefer], (float, int)):
        return float(d[prefer])
    y = get_faithfulness_curve(d)
    if y is None:
        return None
    if prefer == "faithfulness@100":
        return float(y[-1])
    if prefer == "average":
        return float(np.mean(y))
    x = np.array(BUDGET_PCTS, dtype=float) / BUDGET_PCTS[-1]
    return float(np.trapz(y, x))


def build_method_tag(method_base: str, src_model: str, train_spec: str, control: str) -> str:
    # matches evaluation.sh method_tag construction
    return f"{method_base}__kind-diffalign__src-{src_model}__train-{train_spec}__ctrl-{control}"


_ALIGN_DIR_RE = re.compile(
    r"^(?P<method>.+?)__kind-diffalign__src-(?P<src>[^_]+)__train-(?P<train>.+?)__ctrl-(?P<ctrl>.+)$"
)


def list_aligned_method_tags(
    results_root: Path,
    method_base: str,
    src_model: str,
    ablation: str,
    level: str,
    control: str = "none",
    train_glob: str = "*",
) -> List[Tuple[str, str]]:
    pattern = f"{method_base}__kind-diffalign__src-{src_model}__train-{train_glob}__ctrl-{control}_{ablation}_{level}"
    hits = sorted([p for p in results_root.glob(pattern) if p.is_dir()])
    out: List[Tuple[str, str]] = []
    suffix = f"_{ablation}_{level}"
    for h in hits:
        dn = h.name
        if not dn.endswith(suffix):
            continue
        method_tag = dn[: -len(suffix)]
        m = _ALIGN_DIR_RE.match(method_tag)
        if not m:
            continue
        out.append((m.group("train"), method_tag))
    return out


def choose_best_aligned(
    results_root: Path,
    method_base: str,
    src_model: str,
    tgt_model: str,
    ablation: str,
    level: str,
    task: str,
    split: str,
    absolute: bool,
    train_mode: str = "best",  # best|train|loo|fixed
    fixed_train_spec: str = "",
    prefer: str = "area_under",
) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
    if train_mode == "fixed":
        if not fixed_train_spec:
            raise ValueError("train_mode=fixed requires fixed_train_spec")
        candidates = [(fixed_train_spec, build_method_tag(method_base, src_model, fixed_train_spec, "none"))]
    elif train_mode == "train":
        ts = f"TRAIN-{task}"
        candidates = [(ts, build_method_tag(method_base, src_model, ts, "none"))]
    elif train_mode == "loo":
        ts = f"LOO-{task}"
        candidates = [(ts, build_method_tag(method_base, src_model, ts, "none"))]
    elif train_mode == "best":
        candidates = list_aligned_method_tags(results_root, method_base, src_model, ablation, level, control="none", train_glob="*")
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    best_d = None
    best_s = None
    best_train = None
    best_tag = None
    for train_spec, method_tag in candidates:
        d = load_result(results_root, method_tag, ablation, level, task, tgt_model, split, absolute)
        s = curve_score(d, prefer=prefer)
        if s is None:
            continue
        if best_s is None or s > best_s:
            best_s = s
            best_d = d
            best_train = train_spec
            best_tag = method_tag
    return best_train, best_tag, best_d


def parse_list(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[\s,]+", s.strip())
    return [p for p in parts if p]
