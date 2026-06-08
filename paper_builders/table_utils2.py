
#!/usr/bin/env python3
from __future__ import annotations
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

from model_config import DEFAULT_MODEL_PAIRS, pair_display

TASKS: List[Tuple[str, str]] = [
    ("ioi", "IOI"),
    ("mcqa", "MCQA"),
    ("arithmetic_addition", "Arith +"),
    ("arithmetic_subtraction", "Arith -"),
    ("arc_easy", "ARC-E"),
    ("arc_challenge", "ARC-C"),
]

MODEL_PAIRS: List[Tuple[str, str, str]] = [
    (src, tgt, pair_display(src, tgt))
    for src, tgt in DEFAULT_MODEL_PAIRS
]

METHODS: List[Tuple[str, str]] = [
    ("EAP-IG-inputs", "NAP-IG-Inputs"),
    ("EAP-IG-activations", "NAP-IG-Activations"),
    ("EAP", "NAP"),
]

ABLAT = "patching"
LEVEL = "node"

def tex_escape(s: str) -> str:
    return s.replace("_", "\\_")

def fmt_num(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.2f}"

def fmt_pct(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.1f}\\%"

def load_records(results_dir: str, metric_key: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for root, _, files in os.walk(results_dir):
        for fn in files:
            if not fn.endswith(".pkl"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if metric_key not in payload:
                continue
            meta = payload.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            recs.append({
                "path": path,
                "value": float(payload[metric_key]),
                "method_base": meta.get("method_base", ""),
                "kind": meta.get("kind", ""),             # 'gold' or 'diffalign'
                "control": meta.get("control", ""),       # 'none', 'random_W', ...
                "train_spec": meta.get("train_spec", ""), # 'LOO-ioi', 'TRAIN-ioi', etc.
                "task": meta.get("task", ""),
                "model": meta.get("model", ""),
                "source_model": meta.get("source_model", ""),
                "ablation": meta.get("ablation", ""),
                "level": meta.get("level", ""),
                "split": meta.get("split", ""),
                "absolute": bool(meta.get("absolute", False)),
            })
    return recs

def build_index(records: List[Dict[str, Any]], *, split: str, absolute: bool) -> Dict[Tuple[str,str,str,str,str,str,str], float]:
    """
    Key:
      (method_base, kind, control, train_spec, eval_task, source_model, target_model)
    """
    idx: Dict[Tuple[str,str,str,str,str,str,str], float] = {}
    for r in records:
        if r["split"] != split:
            continue
        if r["absolute"] != absolute:
            continue
        if r["ablation"] != ABLAT or r["level"] != LEVEL:
            continue
        key = (
            r["method_base"], r["kind"], r["control"], r["train_spec"],
            r["task"], r["source_model"], r["model"]
        )
        idx[key] = r["value"]  # last one wins
    return idx

def near_train_spec(eval_task: str) -> Optional[str]:
    if eval_task == "arithmetic_addition":
        return "TRAIN-arithmetic_subtraction"
    if eval_task == "arithmetic_subtraction":
        return "TRAIN-arithmetic_addition"
    if eval_task == "arc_easy":
        return "TRAIN-arc_challenge"
    if eval_task == "arc_challenge":
        return "TRAIN-arc_easy"
    return None

def get_gold(idx, method_base: str, eval_task: str, target_model: str) -> Optional[float]:
    return idx.get((method_base, "gold", "", "", eval_task, "", target_model))

def get_baseline(idx, method_base: str, eval_task: str, source_model: str, target_model: str) -> Optional[float]:
    return idx.get((method_base, "diffalign", "random_W", f"LOO-{eval_task}", eval_task, source_model, target_model))

def get_far(idx, method_base: str, eval_task: str, source_model: str, target_model: str) -> Optional[float]:
    return idx.get((method_base, "diffalign", "none", f"LOO-{eval_task}", eval_task, source_model, target_model))

def get_in(idx, method_base: str, eval_task: str, source_model: str, target_model: str) -> Optional[float]:
    return idx.get((method_base, "diffalign", "none", f"TRAIN-{eval_task}", eval_task, source_model, target_model))

def get_near(idx, method_base: str, eval_task: str, source_model: str, target_model: str) -> Optional[float]:
    ts = near_train_spec(eval_task)
    if ts is None:
        return None
    return idx.get((method_base, "diffalign", "none", ts, eval_task, source_model, target_model))

def best_aligned(idx, method_base: str, eval_task: str, source_model: str, target_model: str) -> Optional[float]:
    vals = [get_far(idx, method_base, eval_task, source_model, target_model),
            get_in(idx, method_base, eval_task, source_model, target_model),
            get_near(idx, method_base, eval_task, source_model, target_model)]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None

def recovery_pct(aligned: Optional[float], gold: Optional[float]) -> Optional[float]:
    if aligned is None or gold is None or gold == 0:
        return None
    return (aligned / gold) * 100.0

def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
