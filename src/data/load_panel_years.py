# src/data/load_panel.py
from pathlib import Path
import pandas as pd

from src.utils.parquet_io import load_parquet
# Repo-Root unabhängig vom cwd bestimmen (…/REPO)
ROOT = Path(__file__).resolve().parents[2]
CLEAN_ROOT = ROOT / "data" / "clean"

def load_panel_years(years: list[int]) -> pd.DataFrame:
    dfs = []
    for y in years:
        p = CLEAN_ROOT / "cpcv" / f"{y}" /f"{y}_features.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Fehlt: {p} (cwd={Path.cwd()})")
        dfs.append(load_parquet(p))
    df = pd.concat(dfs).sort_index()
    assert df.index.is_unique
    return df

def load_panel_wf(features_source: str) -> pd.DataFrame:
    """
    Lädt ein komplettes Feature-Panel (für Walk-Forward/expanding)
    und lässt später segmentweise per Datum schneiden.
    Beispiel: features_source="features_v1_raw_z"
    """
    p = CLEAN_ROOT / f"{features_source}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Fehlt: {p} (cwd={Path.cwd()})")
    df = load_parquet(p).sort_index()
    assert df.index.is_unique
    return df