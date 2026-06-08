
import argparse
import csv
import os
import pickle
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from model_config import DEFAULT_MODEL_PAIRS, pair_display, pretty_model_size

MODEL_PAIRS: List[Tuple[str, str]] = list(DEFAULT_MODEL_PAIRS)

METHODS: List[str] = [
    "EAP-IG-inputs",
    "EAP",
    "EAP-IG-activations",
]

METHOD_DISPLAY: Dict[str, str] = {
    "EAP-IG-inputs": "NAP-IG-Inputs",
    "EAP": "NAP",
    "EAP-IG-activations": "NAP-IG-Activations",
}

TASKS: List[Tuple[str, str]] = [
    ("ioi", "IOI"),
    ("mcqa", "MCQA"),
    ("arithmetic_addition", "Arith +"),
    ("arithmetic_subtraction", "Arith -"),
    ("arc_easy", "ARC-E"),
    ("arc_challenge", "ARC-C"),
]

SHORT_TASK_LABEL: Dict[str, str] = {
    "ioi": "IOI",
    "mcqa": "MCQA",
    "arithmetic_addition": "Arith +",
    "arithmetic_subtraction": "Arith -",
    "arc_easy": "ARC-E",
    "arc_challenge": "ARC-C",
}

NEAR_DIST_SOURCE: Dict[str, str] = {
    "arithmetic_addition": "TRAIN-arithmetic_subtraction",
    "arithmetic_subtraction": "TRAIN-arithmetic_addition",
    "arc_easy": "TRAIN-arc_challenge",
    "arc_challenge": "TRAIN-arc_easy",
}

SUMMARY_TASKS: List[Tuple[str, str]] = [
    ("ioi", "IOI"),
    ("mcqa", "MCQA"),
    ("arithmetic_addition", "Arith +"),
    ("arc_easy", "ARC-E"),
]

LATEX_REPL = {
    "&": r"\&",
    "%": r"\%",
    "_": r"\_",
    "#": r"\#",
}


def parse_args_base(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-dir", default="tables")
    p.add_argument("--metric-key", default="area_under")
    p.add_argument("--split", default="test")
    p.add_argument("--ablation", default="patching")
    p.add_argument("--level", default="node")
    p.add_argument("--absolute", action="store_true")
    return p.parse_args()


def escape_latex(s: str) -> str:
    out = s
    for a, b in LATEX_REPL.items():
        out = out.replace(a, b)
    return out


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return safe_float(x[0])
    return None


def parse_packed_method_tag(method_tag: str) -> Dict[str, str]:
    parts = method_tag.split("__")
    info: Dict[str, str] = {
        "method_base": parts[0] if parts else "",
        "kind": "gold",
        "source_model": "",
        "train_spec": "",
        "control": "",
    }
    for segment in parts[1:]:
        if "-" not in segment:
            continue
        key, value = segment.split("-", 1)
        if key == "kind":
            info["kind"] = value
        elif key == "src":
            info["source_model"] = value
        elif key == "train":
            info["train_spec"] = value
        elif key == "ctrl":
            info["control"] = value
        else:
            info[key] = value
    return info


def load_records(results_dir: str, metric_key: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for root, _, files in os.walk(results_dir):
        for filename in files:
            if not filename.endswith(".pkl"):
                continue
            path = os.path.join(root, filename)
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
            except Exception as e:
                print(f"Warning: failed to read {path}: {e}")
                continue
            if not isinstance(payload, dict):
                continue
            value = safe_float(payload.get(metric_key))
            if value is None:
                continue
            meta = payload.get("meta", {})
            method_tag = meta.get("method_tag", "")
            parsed = parse_packed_method_tag(method_tag) if method_tag else {
                "method_base": "",
                "kind": "unknown",
                "source_model": "",
                "train_spec": "",
                "control": "",
            }
            records.append({
                "path": path,
                "value": value,
                "method_tag": method_tag,
                "method_base": meta.get("method_base", parsed.get("method_base", "")),
                "kind": meta.get("kind", parsed.get("kind", "unknown")),
                "source_model": meta.get("source_model", parsed.get("source_model", "")),
                "train_spec": meta.get("train_spec", parsed.get("train_spec", "")),
                "control": meta.get("control", parsed.get("control", "")),
                "ablation": meta.get("ablation", ""),
                "level": meta.get("level", ""),
                "task": meta.get("task", ""),
                "model": meta.get("model", ""),
                "split": meta.get("split", ""),
                "absolute": bool(meta.get("absolute", False)),
            })
    return records


def make_index(
    records: List[Dict[str, Any]],
    split: str,
    ablation: str,
    level: str,
    absolute: bool,
) -> Dict[Tuple[str, str, str, str, str, str, str, str], float]:
    idx: Dict[Tuple[str, str, str, str, str, str, str, str], float] = {}
    for r in records:
        if r["split"] != split:
            continue
        if r["ablation"] != ablation or r["level"] != level:
            continue
        if bool(r["absolute"]) != bool(absolute):
            continue
        key = (
            r["method_base"],
            r["source_model"],
            r["model"],
            r["kind"],
            r["control"],
            r["train_spec"],
            r["task"],
            r["path"],  # uniqueness safeguard
        )
        idx[key] = r["value"]
    # collapse by dropping path, last one wins but deterministically by sorted path
    collapsed: Dict[Tuple[str, str, str, str, str, str, str, str], float] = {}
    for key in sorted(idx.keys()):
        method_base, source_model, model, kind, control, train_spec, task, _path = key
        collapsed[(method_base, source_model, model, kind, control, train_spec, task, "")] = idx[key]
    return collapsed


def query_value(
    idx: Dict[Tuple[str, str, str, str, str, str, str, str], float],
    method_base: str,
    source_model: str,
    target_model: str,
    kind: str,
    control: str,
    train_spec: str,
    task: str,
) -> Optional[float]:
    return idx.get((method_base, source_model, target_model, kind, control, train_spec, task, ""))


def row_average(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def format_plain(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:.2f}"


def style_text(text: str, bold: bool = False, underline: bool = False, italic: bool = False) -> str:
    out = text
    if italic:
        out = r"\textit{" + out + "}"
    if underline:
        out = r"\underline{" + out + "}"
    if bold:
        out = r"\textbf{" + out + "}"
    return out


def rank_indices(values: Sequence[Optional[float]]) -> Tuple[Optional[int], Optional[int]]:
    indexed = [(i, v) for i, v in enumerate(values) if v is not None]
    if not indexed:
        return None, None
    indexed.sort(key=lambda x: (-x[1], x[0]))
    best_idx = indexed[0][0]
    best_val = indexed[0][1]
    second_idx: Optional[int] = None
    for i, v in indexed[1:]:
        if v < best_val:
            second_idx = i
            break
    return best_idx, second_idx


def styled_numeric_cells(
    values: Sequence[Optional[float]],
    best_idx: Optional[int],
    second_idx: Optional[int],
    italic_idx: Optional[int] = None,
) -> List[str]:
    out: List[str] = []
    for i, v in enumerate(values):
        if v is None:
            out.append("-")
            continue
        text = f"{v:.2f}"
        out.append(
            style_text(
                text,
                bold=(best_idx == i),
                underline=(second_idx == i),
                italic=(italic_idx == i),
            )
        )
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def write_latex_lines(path: str, lines: Sequence[str]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def build_method_rows(
    idx: Dict[Tuple[str, str, str, str, str, str, str, str], float],
    method: str,
    source_model: str,
    target_model: str,
) -> Dict[str, List[Optional[float]]]:
    gold = [query_value(idx, method, "", target_model, "gold", "", "", task) for task, _ in TASKS]
    ind = [query_value(idx, method, source_model, target_model, "diffalign", "none", f"TRAIN-{task}", task) for task, _ in TASKS]
    near: List[Optional[float]] = []
    for task, _ in TASKS:
        train_spec = NEAR_DIST_SOURCE.get(task)
        if train_spec is None:
            near.append(None)
        else:
            near.append(query_value(idx, method, source_model, target_model, "diffalign", "none", train_spec, task))
    far = [query_value(idx, method, source_model, target_model, "diffalign", "none", f"LOO-{task}", task) for task, _ in TASKS]
    rand = [query_value(idx, method, source_model, target_model, "diffalign", "random_W", f"LOO-{task}", task) for task, _ in TASKS]
    scrambled = [query_value(idx, method, source_model, target_model, "diffalign", "scrambled_s", f"LOO-{task}", task) for task, _ in TASKS]
    permuted = [query_value(idx, method, source_model, target_model, "diffalign", "permuted_W", f"LOO-{task}", task) for task, _ in TASKS]
    heuristic_depth_mean = [query_value(idx, method, source_model, target_model, "diffalign", "heuristic_depth_mean", f"LOO-{task}", task) for task, _ in TASKS]
    return {
        "gold": gold,
        "in": ind,
        "near": near,
        "far": far,
        "random": rand,
        "scrambled": scrambled,
        "permuted": permuted,
        "heuristic_depth_mean": heuristic_depth_mean,
    }
