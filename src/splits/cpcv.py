# src/splits/cpcv.py
from __future__ import annotations
from pathlib import Path
import yaml
from typing import List, Dict, Tuple

Segment = Tuple[str, str]
Fold = Dict[str, List[Segment]]  # {"train": [(start,end),...], "test": [(start,end),...]}

def iter_windows_from_yaml(path: str | Path) -> List[Fold]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    folds = cfg.get("folds", [])
    out: List[Fold] = []
    for fold in folds:
        train = [(s, e) for s, e in fold.get("train", [])]
        test  = [(s, e) for s, e in fold.get("test", [])]
        out.append({"train": train, "test": test})
    return out
