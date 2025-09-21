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
    first_train_years = int(cfg["first_train_years"])   # z.B. 4
    test_years        = int(cfg["test_years"])          # z.B. 1
    step_years        = int(cfg.get("step_years", 1))   # z.B. 1

    folds: List[Fold] = []
    # Start: erstes Trainingsende (expanding)
    train_start = start
    train_end   = train_start + relativedelta(years=+first_train_years) - relativedelta(days=+1)

    while True:
        test_start = train_end + relativedelta(days=+1)
        test_end   = test_start + relativedelta(years=+test_years) - relativedelta(days=+1)
        if test_end > end:
            break

        folds.append({
            "train": [(train_start.isoformat(), train_end.isoformat())],
            "test" : [(test_start.isoformat(),  test_end .isoformat())],
        })

        # expanding: nur train_end schiebt sich nach vorne
        train_end = train_end + relativedelta(years=+step_years)

    return folds

from datetime import date, datetime

def _to_date(x) -> date:
    """Robuste Normalisierung: akzeptiert str, datetime, date, pandas/NumPy-ähnliche Typen."""
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    # z.B. pandas.Timestamp / numpy.datetime64:
    if hasattr(x, "to_pydatetime"):
        return x.to_pydatetime().date()
    s = str(x)
    y, m, d = map(int, s.split("-"))
    return date(y, m, d)

