from __future__ import annotations

from typing import Dict, Tuple

Family = str
ModelPair = Tuple[str, str]
SizePair = Tuple[float, float]

FAMILY_DISPLAY: Dict[Family, str] = {
    "gpt2": "GPT-2",
    "llama": "Llama-3",
    "qwen": "Qwen-2.5",
    "gemma": "Gemma-2",
}

FAMILY_MODEL_BY_SIZE: Dict[Family, Dict[float, str]] = {
    "gpt2": {0.124: "gpt2-small", 0.355: "gpt2-medium", 0.774: "gpt2-large"},
    "llama": {1.0: "llama3-1b", 3.0: "llama3-3b", 8.0: "llama3-8b"},
    "qwen": {0.5: "qwen2.5-0.5b", 1.5: "qwen2.5-1.5b", 3.0: "qwen2.5-3b"},
    "gemma": {2.0: "gemma2-2b", 9.0: "gemma2-9b", 27.0: "gemma2-27b"},
}

FAMILY_ORDER: Dict[Family, Tuple[SizePair, ...]] = {
    "gpt2": ((0.124, 0.355), (0.355, 0.774), (0.124, 0.774)),
    "llama": ((1.0, 3.0), (3.0, 8.0), (1.0, 8.0)),
    "qwen": ((0.5, 1.5), (1.5, 3.0), (0.5, 3.0)),
    "gemma": ((2.0, 9.0), (9.0, 27.0), (2.0, 27.0)),
}

FAMILY_PAIRS: Dict[Family, Tuple[ModelPair, ...]] = {
    family: tuple(
        (FAMILY_MODEL_BY_SIZE[family][src], FAMILY_MODEL_BY_SIZE[family][tgt])
        for src, tgt in order
    )
    for family, order in FAMILY_ORDER.items()
}

DEFAULT_MODEL_PAIRS: Tuple[ModelPair, ...] = tuple(
    pair
    for family in FAMILY_ORDER
    for pair in FAMILY_PAIRS[family]
)


def family_order_items() -> Tuple[Tuple[Family, Tuple[SizePair, ...]], ...]:
    return tuple(FAMILY_ORDER.items())


def family_for_model(model_name: str) -> Family:
    s = model_name.lower()
    if "gpt2" in s:
        return "gpt2"
    if "llama" in s:
        return "llama"
    if "qwen" in s:
        return "qwen"
    if "gemma" in s:
        return "gemma"
    return "other"


def family_for_pair_display(pair_display: str) -> Family:
    s = pair_display.lower()
    if "gpt" in s:
        return "gpt2"
    if "llama" in s:
        return "llama"
    if "qwen" in s:
        return "qwen"
    if "gemma" in s:
        return "gemma"
    return "other"


def model_by_size(family: Family, src_size: float, tgt_size: float) -> ModelPair:
    return FAMILY_MODEL_BY_SIZE[family][src_size], FAMILY_MODEL_BY_SIZE[family][tgt_size]


def pretty_model_size(model_name: str) -> str:
    gpt2_sizes = {
        "gpt2": "0.124B",
        "gpt2-small": "0.124B",
        "gpt2-medium": "0.355B",
        "gpt2-large": "0.774B",
    }
    if model_name in gpt2_sizes:
        return gpt2_sizes[model_name]

    x = model_name
    for prefix in ("llama3-", "qwen2.5-", "gemma2-"):
        x = x.replace(prefix, "")
    for family in ("llama3", "qwen2.5", "gemma2"):
        x = x.replace(family, "")
    x = x.replace("b", "B")
    return x


def pair_display(source: str, target: str) -> str:
    family = FAMILY_DISPLAY.get(family_for_model(source), family_for_model(source))
    return f"{family} ({pretty_model_size(source)} $\\to$ {pretty_model_size(target)})"


def pair_tick(src: float, tgt: float) -> str:
    def fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return str(x).rstrip("0").rstrip(".")

    return f"{fmt(src)}->{fmt(tgt)}"
