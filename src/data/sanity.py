# src/data/sanity.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd

from src.utils.paths import CLEAN_DIR, DATA_DIR
from src.validation.check_cpcv_data import (
    normalize_datetime_index, check_datetime_index, compare_year_splits_to_master, CheckResult
)
from src.utils.parquet_io import load_parquet

def check_data() -> Tuple[bool, CheckResult, CheckResult]:
    feats = load_parquet(CLEAN_DIR / "features_v1.parquet")
    rf    = load_parquet(CLEAN_DIR / "riskfree.parquet")

    feats, _ = normalize_datetime_index(feats)
    rf, _    = normalize_datetime_index(rf)

    r_master = check_datetime_index(feats, "features_v1")
    r_rf     = check_datetime_index(rf, "riskfree")

    r_cpcv = compare_year_splits_to_master(
        feats, Path(DATA_DIR) / "cpcv" / "years" / "features", label="CPCV/features"
    )
    r_wf = compare_year_splits_to_master(
        feats, Path(DATA_DIR) / "walk_forward" / "years" / "features", label="WF/features"
    )
    # (optional) riskfree-Jahre analog prüfen

    ok = r_master.ok and r_rf.ok and r_cpcv.ok and r_wf.ok
    # kombiniere bei Bedarf Messages / gib sie im CLI aus
    combined = CheckResult(ok=ok, errors=r_master.errors+r_rf.errors+r_cpcv.errors+r_wf.errors,
                           warns=r_master.warns+r_rf.warns+r_cpcv.warns+r_wf.warns)
    return ok, combined, r_cpcv  # tuple, falls du differenziert reagieren willst

if __name__ == "__main__":
    ok, combined, _ = check_data()
    for w in combined.warns:  print("[WARN]", w)
    for e in combined.errors: print("[FAIL]", e)
    print("OK" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
