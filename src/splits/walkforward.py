# src/splits/walkforward.py
from __future__ import annotations
from datetime import date
from dateutil.relativedelta import relativedelta
from pathlib import Path
import yaml
from typing import List, Dict, Tuple

Segment = Tuple[str, str]
Fold = Dict[str, List[Segment]]

def iter_windows_from_yaml(path: str | Path) -> List[Fold]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    start = _to_date(cfg["start"])
    end   = _to_date(cfg["end"])
    train_years = int(cfg["train_years"])
    test_years  = int(cfg["test_months"])
    step_months  = int(cfg.get("step_months", test_years))

    folds: List[Fold] = []
    cur = start
    while True:
        train_start = cur
        train_end   = train_start + relativedelta(months=+train_years) - relativedelta(days=+1)
        test_start  = train_end + relativedelta(days=+1)
        test_end    = test_start + relativedelta(months=+test_years) - relativedelta(days=+1)
        if test_end > end:
            break
        folds.append({
            "train": [(train_start.isoformat(), train_end.isoformat())],
            "test" : [(test_start.isoformat(),  test_end .isoformat())],
        })
        cur = cur + relativedelta(months=+step_months)
    return folds

def _to_date(s: str) -> date:
    y, m, d = map(int, s.split("-"))
    return date(y, m, d)
